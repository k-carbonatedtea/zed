mod actions;
pub(crate) mod autoscroll;
pub(crate) mod scroll_amount;

use crate::editor_settings::ScrollBeyondLastLine;
use crate::{
    Anchor, DisplayPoint, DisplayRow, Editor, EditorEvent, EditorMode, EditorSettings,
    MultiBufferSnapshot, RowExt, SelectionEffects, SizingBehavior, ToPoint,
    display_map::{DisplaySnapshot, LongRow, MeasureRequest, ToDisplayPoint},
    hover_popover::hide_hover,
    persistence::EditorDb,
};
pub use autoscroll::{Autoscroll, AutoscrollStrategy};
use core::fmt::Debug;
use gpui::{
    Along, App, AppContext as _, Axis, Context, Entity, EntityId, OngoingScroll, Pixels, Task,
    TouchPhase, Window, point,
};
use language::language_settings::{AllLanguageSettings, SoftWrap};
use language::{Bias, Point};
use multi_buffer::MultiBufferRow;
pub use scroll_amount::ScrollAmount;
use settings::Settings;
use std::{cmp::Ordering, ops::RangeInclusive, sync::Arc, time::Duration};
use text::Selection;
use ui::scrollbars::ScrollbarAutoHide;
use util::ResultExt;
use workspace::{ItemId, WorkspaceId};

const SCROLLBAR_SHOW_INTERVAL: Duration = Duration::from_secs(1);

pub struct WasScrolled(pub(crate) bool);

struct ProvisionalAutoscroll {
    request: (Autoscroll, bool),
    disjoint: Arc<[Selection<Anchor>]>,
    pending: Option<Selection<Anchor>>,
    anchor: ScrollAnchor,
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct SerializedScrollPosition {
    pub top_row: u32,
    pub offset: gpui::Point<ScrollOffset>,
    pub horizontal: Option<(Point, ScrollPixelOffset)>,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct HorizontalScrollAnchor {
    anchor: Anchor,
    offset: ScrollPixelOffset,
    display_map_id: EntityId,
}

pub type ScrollOffset = f64;
pub type ScrollPixelOffset = f64;
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ScrollAnchor {
    pub offset: gpui::Point<ScrollOffset>,
    pub anchor: Anchor,
}

impl ScrollAnchor {
    pub(super) fn new() -> Self {
        Self {
            offset: gpui::Point::default(),
            anchor: Anchor::Min,
        }
    }

    pub fn scroll_position(&self, snapshot: &DisplaySnapshot) -> gpui::Point<ScrollOffset> {
        self.offset.apply_along(Axis::Vertical, |offset| {
            if self.anchor == Anchor::Min {
                0.
            } else {
                let scroll_top = self.anchor.to_display_point(snapshot).row().as_f64();
                (offset + scroll_top).max(0.)
            }
        })
    }

    pub fn top_row(&self, buffer: &MultiBufferSnapshot) -> u32 {
        self.anchor.to_point(buffer).row
    }
}

/// In the split diff view, the two sides share a ScrollAnchor using this struct.
/// Either side can set a ScrollAnchor that points to its own multibuffer, and we store the ID of the display map
/// that the last-written anchor came from so that we know how to resolve it to a DisplayPoint.
///
/// For normal editors, this just acts as a wrapper around a ScrollAnchor.
#[derive(Clone, Copy, Debug)]
pub struct SharedScrollAnchor {
    pub scroll_anchor: ScrollAnchor,
    pub display_map_id: Option<EntityId>,
    horizontal: Option<HorizontalScrollAnchor>,
    local: bool,
}

impl SharedScrollAnchor {
    pub fn scroll_position(&self, snapshot: &DisplaySnapshot) -> gpui::Point<ScrollOffset> {
        let snapshot = if let Some(display_map_id) = self.display_map_id
            && display_map_id != snapshot.display_map_id
        {
            let companion_snapshot = snapshot
                .companion_snapshot()
                .expect("shared scroll anchor references a non native display map, but snapshot has no companion");
            assert_eq!(
                companion_snapshot.display_map_id, display_map_id,
                "shared scroll anchor display map should match the snapshot's split companion"
            );
            companion_snapshot
        } else {
            snapshot
        };

        self.scroll_anchor.scroll_position(snapshot)
    }

    pub fn scroll_top_row(&self, snapshot: &DisplaySnapshot) -> DisplayRow {
        let snapshot = if let Some(display_map_id) = self.display_map_id
            && display_map_id != snapshot.display_map_id
        {
            let companion_snapshot = snapshot
                .companion_snapshot()
                .expect("shared scroll anchor references a non native display map, but snapshot has no companion");
            assert_eq!(
                companion_snapshot.display_map_id, display_map_id,
                "shared scroll anchor display map should match the snapshot's split companion"
            );
            companion_snapshot
        } else {
            snapshot
        };

        self.scroll_anchor.anchor.to_display_point(snapshot).row()
    }

    fn horizontal_scroll_left(&self, snapshot: &DisplaySnapshot) -> Option<ScrollPixelOffset> {
        let (long_row, column, offset) = self.horizontal_anchor_position(snapshot)?;
        Some(long_row.x_for_column(column) - offset)
    }

    fn horizontal_anchor_position(
        &self,
        snapshot: &DisplaySnapshot,
    ) -> Option<(LongRow, u32, ScrollPixelOffset)> {
        let horizontal = self.horizontal?;
        let point = self.horizontal_display_point(snapshot)?;
        let long_row = snapshot.long_row(point.row())?;
        let column = point.column().min(long_row.len());
        Some((long_row, column, horizontal.offset))
    }

    fn native_horizontal(&self, snapshot: &DisplaySnapshot) -> Option<HorizontalScrollAnchor> {
        let horizontal = self.horizontal?;
        if horizontal.display_map_id == snapshot.display_map_id {
            return Some(horizontal);
        }
        let point = self.horizontal_display_point(snapshot)?;
        Some(HorizontalScrollAnchor {
            anchor: snapshot
                .buffer_snapshot()
                .anchor_before(point.to_offset(snapshot, Bias::Left)),
            offset: horizontal.offset,
            display_map_id: snapshot.display_map_id,
        })
    }

    fn horizontal_display_point(&self, snapshot: &DisplaySnapshot) -> Option<DisplayPoint> {
        let horizontal = self.horizontal?;
        if horizontal.display_map_id == snapshot.display_map_id {
            return Some(horizontal.anchor.to_display_point(snapshot));
        }
        let companion = snapshot
            .companion_snapshot()
            .filter(|companion| companion.display_map_id == horizontal.display_map_id)?;
        let companion_point = horizontal.anchor.to_point(companion.buffer_snapshot());
        if snapshot.is_block_line(companion_point.to_display_point(companion).row()) {
            return None;
        }
        let (point, past_line_end) =
            companion_display_point(snapshot, companion, companion_point, Bias::Left);
        (!past_line_end).then_some(point)
    }
}

#[derive(Copy, Clone, Default, PartialEq, Eq)]
pub enum ScrollbarThumbState {
    #[default]
    Idle,
    Hovered,
    Dragging,
}

#[derive(PartialEq, Eq)]
pub struct ActiveScrollbarState {
    axis: Axis,
    thumb_state: ScrollbarThumbState,
}

impl ActiveScrollbarState {
    pub fn new(axis: Axis, thumb_state: ScrollbarThumbState) -> Self {
        ActiveScrollbarState { axis, thumb_state }
    }

    pub fn thumb_state_for_axis(&self, axis: Axis) -> Option<ScrollbarThumbState> {
        (self.axis == axis).then_some(self.thumb_state)
    }
}

pub struct ScrollManager {
    pub(crate) vertical_scroll_margin: ScrollOffset,
    anchor: Entity<SharedScrollAnchor>,
    /// Value to be used for clamping the x component of the SharedScrollAnchor's offset.
    ///
    /// We store this outside the SharedScrollAnchor so that the two sides of a split diff can share
    /// a horizontal scroll offset that may be out of range for one of the editors (when one side is wider than the other).
    /// Each side separately clamps the x component using its own scroll_max_x when reading from the SharedScrollAnchor.
    scroll_max_x: Option<f64>,
    ongoing: OngoingScroll,
    /// The second element indicates whether the autoscroll request is local
    /// (true) or remote (false). Local requests are initiated by user actions,
    /// while remote requests come from external sources.
    autoscroll_request: Option<(Autoscroll, bool)>,
    provisional_autoscroll: Option<ProvisionalAutoscroll>,
    horizontal_scroll_x: Option<(HorizontalScrollAnchor, ScrollOffset)>,
    last_autoscroll: Option<(
        gpui::Point<ScrollOffset>,
        ScrollOffset,
        ScrollOffset,
        AutoscrollStrategy,
    )>,
    show_scrollbars: bool,
    hide_scrollbar_task: Option<Task<()>>,
    active_scrollbar: Option<ActiveScrollbarState>,
    visible_line_count: Option<f64>,
    visible_column_count: Option<f64>,
    forbid_vertical_scroll: bool,
    minimap_thumb_state: Option<ScrollbarThumbState>,
    persisted_position: Option<(Option<WorkspaceId>, SerializedScrollPosition)>,
    _save_scroll_position_task: Task<()>,
}

impl ScrollManager {
    pub fn new(cx: &mut Context<Editor>) -> Self {
        let anchor = cx.new(|_| SharedScrollAnchor {
            scroll_anchor: ScrollAnchor::new(),
            display_map_id: None,
            horizontal: None,
            local: true,
        });
        ScrollManager {
            vertical_scroll_margin: EditorSettings::get_global(cx).vertical_scroll_margin,
            anchor,
            scroll_max_x: None,
            ongoing: OngoingScroll::default(),
            autoscroll_request: None,
            provisional_autoscroll: None,
            horizontal_scroll_x: None,
            show_scrollbars: true,
            hide_scrollbar_task: None,
            active_scrollbar: None,
            last_autoscroll: None,
            visible_line_count: None,
            visible_column_count: None,
            forbid_vertical_scroll: false,
            minimap_thumb_state: None,
            persisted_position: None,
            _save_scroll_position_task: Task::ready(()),
        }
    }

    pub fn set_native_display_map_id(
        &mut self,
        display_map_id: EntityId,
        cx: &mut Context<Editor>,
    ) {
        self.anchor.update(cx, |shared, _| {
            if shared.display_map_id.is_none() {
                shared.display_map_id = Some(display_map_id);
            }
        });
    }

    pub fn clone_state(
        &mut self,
        other: &Self,
        other_snapshot: &DisplaySnapshot,
        my_snapshot: &DisplaySnapshot,
        cx: &mut Context<Editor>,
    ) {
        let native = other.native_shared_anchor(other_snapshot, cx);
        let cloned = SharedScrollAnchor {
            display_map_id: Some(my_snapshot.display_map_id),
            horizontal: native.horizontal.map(|horizontal| HorizontalScrollAnchor {
                display_map_id: my_snapshot.display_map_id,
                ..horizontal
            }),
            ..native
        };
        self.anchor.update(cx, |shared, _| *shared = cloned);
        self.ongoing = other.ongoing;
        self.provisional_autoscroll.take();
        self.cache_native_horizontal(&cloned);
    }

    pub fn offset(&self, cx: &App) -> gpui::Point<f64> {
        self.effective_offset(self.anchor.read(cx))
    }

    fn effective_offset(&self, shared: &SharedScrollAnchor) -> gpui::Point<f64> {
        let mut offset = shared.scroll_anchor.offset;
        if let Some((horizontal, x)) = self.horizontal_scroll_x
            && shared.horizontal == Some(horizontal)
        {
            offset.x = x;
        }
        if let Some(max_x) = self.scroll_max_x {
            offset.x = offset.x.min(max_x);
        }
        offset
    }

    /// Get a ScrollAnchor whose `anchor` field is guaranteed to point into the multibuffer for the provided snapshot.
    ///
    /// For normal editors, this just retrieves the internal ScrollAnchor and is lossless. When the editor is part of a split diff,
    /// we may need to translate the anchor to point to the "native" multibuffer first. That translation is lossy,
    /// so this method should be used sparingly---if you just need a scroll position or display point, call the appropriate helper method instead,
    /// since they can losslessly handle the case where the ScrollAnchor was last set from the other side.
    pub fn native_anchor(&self, snapshot: &DisplaySnapshot, cx: &App) -> ScrollAnchor {
        let shared = self.anchor.read(cx);

        let mut result = if let Some(display_map_id) = shared.display_map_id
            && display_map_id != snapshot.display_map_id
        {
            let companion_snapshot = snapshot
                .companion_snapshot()
                .expect("shared scroll anchor references a non native display map, but the snapshot has no companion");
            assert_eq!(
                companion_snapshot.display_map_id, display_map_id,
                "shared scroll anchor display map should match the companion used for native anchor conversion"
            );

            let mut display_point = shared
                .scroll_anchor
                .anchor
                .to_display_point(companion_snapshot);
            *display_point.column_mut() = 0;
            let buffer_point = snapshot.display_point_to_point(display_point, Bias::Left);
            let anchor = snapshot.buffer_snapshot().anchor_before(buffer_point);
            ScrollAnchor {
                anchor,
                offset: shared.scroll_anchor.offset,
            }
        } else {
            shared.scroll_anchor
        };

        result.offset = self.effective_offset(shared);
        result
    }

    pub fn shared_scroll_anchor(&self, cx: &App) -> SharedScrollAnchor {
        let mut shared = *self.anchor.read(cx);
        shared.scroll_anchor.offset = self.effective_offset(&shared);
        shared
    }

    pub fn scroll_top_row(&self, snapshot: &DisplaySnapshot, cx: &App) -> DisplayRow {
        self.anchor.read(cx).scroll_top_row(snapshot)
    }

    pub fn scroll_anchor_entity(&self) -> Entity<SharedScrollAnchor> {
        self.anchor.clone()
    }

    pub fn set_shared_scroll_anchor(&mut self, entity: Entity<SharedScrollAnchor>) {
        self.anchor = entity;
    }

    pub fn unshare_scroll_anchor(&mut self, snapshot: &DisplaySnapshot, cx: &mut Context<Editor>) {
        let native = self.native_shared_anchor(snapshot, cx);
        self.anchor = cx.new(|_| native);
        self.cache_native_horizontal(&native);
    }

    pub(crate) fn make_scroll_anchor_native(
        &mut self,
        snapshot: &DisplaySnapshot,
        cx: &mut Context<Editor>,
    ) {
        let native = self.native_shared_anchor(snapshot, cx);
        self.anchor.update(cx, |shared, _| *shared = native);
        self.cache_native_horizontal(&native);
    }

    fn native_shared_anchor(&self, snapshot: &DisplaySnapshot, cx: &App) -> SharedScrollAnchor {
        let shared = self.anchor.read(cx);
        SharedScrollAnchor {
            scroll_anchor: self.native_anchor(snapshot, cx),
            display_map_id: Some(snapshot.display_map_id),
            horizontal: shared.native_horizontal(snapshot),
            local: shared.local,
        }
    }

    fn cache_native_horizontal(&mut self, native: &SharedScrollAnchor) {
        self.horizontal_scroll_x = native
            .horizontal
            .map(|horizontal| (horizontal, native.scroll_anchor.offset.x));
    }

    pub fn filter_scroll_delta(
        &mut self,
        delta: &mut gpui::Point<Pixels>,
        touch_phase: TouchPhase,
    ) {
        self.ongoing.filter(delta, touch_phase);
    }

    pub fn scroll_position(
        &self,
        snapshot: &DisplaySnapshot,
        cx: &App,
    ) -> gpui::Point<ScrollOffset> {
        let shared = self.anchor.read(cx);
        let mut pos = shared.scroll_position(snapshot);
        pos.x = self.effective_offset(shared).x;
        pos
    }

    fn set_scroll_position(
        &mut self,
        scroll_position: gpui::Point<ScrollOffset>,
        map: &DisplaySnapshot,
        scroll_beyond_last_line: ScrollBeyondLastLine,
        local: bool,
        autoscroll: bool,
        window: &mut Window,
        cx: &mut Context<Editor>,
    ) -> WasScrolled {
        let scroll_top = scroll_position.y.max(0.);
        let scroll_top = match scroll_beyond_last_line {
            ScrollBeyondLastLine::OnePage => scroll_top,
            ScrollBeyondLastLine::Off => {
                if let Some(height_in_lines) = self.visible_line_count {
                    let max_row = map.max_point().row().as_f64();
                    scroll_top.min(max_row - height_in_lines + 1.).max(0.)
                } else {
                    scroll_top
                }
            }
            ScrollBeyondLastLine::VerticalScrollMargin => {
                if let Some(height_in_lines) = self.visible_line_count {
                    let max_row = map.max_point().row().as_f64();
                    scroll_top
                        .min(max_row - height_in_lines + 1. + self.vertical_scroll_margin)
                        .max(0.)
                } else {
                    scroll_top
                }
            }
        };
        let scroll_top_row = DisplayRow(scroll_top as u32);
        let scroll_top_buffer_point = map
            .clip_point(
                DisplayPoint::new(scroll_top_row, scroll_position.x as u32),
                Bias::Left,
            )
            .to_point(map);
        let top_anchor = map.buffer_snapshot().anchor_before(scroll_top_buffer_point);

        self.set_anchor(
            ScrollAnchor {
                anchor: top_anchor,
                offset: point(
                    scroll_position.x.max(0.),
                    scroll_top - top_anchor.to_display_point(map).row().as_f64(),
                ),
            },
            map,
            local,
            autoscroll,
            window,
            cx,
        )
    }

    fn set_anchor(
        &mut self,
        anchor: ScrollAnchor,
        display_map: &DisplaySnapshot,
        local: bool,
        autoscroll: bool,
        window: &mut Window,
        cx: &mut Context<Editor>,
    ) -> WasScrolled {
        let adjusted_anchor = if self.forbid_vertical_scroll {
            let current = self.anchor.read(cx);
            ScrollAnchor {
                offset: gpui::Point::new(anchor.offset.x, current.scroll_anchor.offset.y),
                anchor: current.scroll_anchor.anchor,
            }
        } else {
            anchor
        };

        self.scroll_max_x.take();
        self.autoscroll_request.take();

        let current = self.anchor.read(cx);
        if current.scroll_anchor == adjusted_anchor {
            return WasScrolled(false);
        }

        self.anchor.update(cx, |shared, _| {
            shared.scroll_anchor = adjusted_anchor;
            shared.display_map_id = Some(display_map.display_map_id);
            shared.horizontal = None;
            shared.local = local;
        });
        self.horizontal_scroll_x = None;
        cx.emit(EditorEvent::ScrollPositionChanged { local, autoscroll });
        self.show_scrollbars(window, cx);
        cx.notify();

        WasScrolled(true)
    }

    pub(crate) fn serialized_position(
        &self,
        snapshot: &DisplaySnapshot,
        cx: &App,
    ) -> SerializedScrollPosition {
        let anchor = self.native_anchor(snapshot, cx);
        let buffer = snapshot.buffer_snapshot();
        SerializedScrollPosition {
            top_row: anchor.top_row(buffer),
            offset: anchor.offset,
            horizontal: self
                .anchor
                .read(cx)
                .native_horizontal(snapshot)
                .map(|horizontal| (horizontal.anchor.to_point(buffer), horizontal.offset)),
        }
    }

    fn persist_position(
        &mut self,
        position: SerializedScrollPosition,
        workspace_id: Option<WorkspaceId>,
        cx: &mut Context<Editor>,
    ) -> bool {
        let persisted = Some((workspace_id, position));
        if self.persisted_position == persisted {
            return false;
        }
        self.persisted_position = persisted;
        let Some(workspace_id) = workspace_id else {
            return true;
        };
        let item_id = cx.entity().entity_id().as_u64() as ItemId;
        let executor = cx.background_executor().clone();
        let db = EditorDb::global(cx);
        self._save_scroll_position_task = cx.background_executor().spawn(async move {
            executor.timer(Duration::from_millis(10)).await;
            log::debug!(
                "Saving scroll position for item {item_id:?} in workspace {workspace_id:?}"
            );
            db.save_scroll_position(
                item_id,
                workspace_id,
                position.top_row,
                position.offset.x,
                position.offset.y,
                position.horizontal.map(|(point, _)| point.row),
                position.horizontal.map(|(point, _)| point.column),
                position.horizontal.map(|(_, offset)| offset),
            )
            .await
            .log_err();
        });
        true
    }

    pub fn show_scrollbars(&mut self, window: &mut Window, cx: &mut Context<Editor>) {
        if !self.show_scrollbars {
            self.show_scrollbars = true;
            cx.notify();
        }

        if cx.default_global::<ScrollbarAutoHide>().should_hide() {
            self.hide_scrollbar_task = Some(cx.spawn_in(window, async move |editor, cx| {
                cx.background_executor()
                    .timer(SCROLLBAR_SHOW_INTERVAL)
                    .await;
                editor
                    .update(cx, |editor, cx| {
                        editor.scroll_manager.show_scrollbars = false;
                        cx.notify();
                    })
                    .log_err();
            }));
        } else {
            self.hide_scrollbar_task = None;
        }
    }

    pub fn scrollbars_visible(&self) -> bool {
        self.show_scrollbars
    }

    pub fn has_autoscroll_request(&self) -> bool {
        self.autoscroll_request.is_some()
    }

    pub fn take_autoscroll_request(&mut self) -> Option<(Autoscroll, bool)> {
        self.autoscroll_request.take()
    }

    pub(crate) fn restore_autoscroll_request(&mut self, request: (Autoscroll, bool)) {
        self.autoscroll_request = Some(request);
    }

    pub(crate) fn retain_provisional_autoscroll(
        &mut self,
        request: (Autoscroll, bool),
        disjoint: Arc<[Selection<Anchor>]>,
        pending: Option<Selection<Anchor>>,
        cx: &App,
    ) {
        let request = match request {
            (Autoscroll::Next, local) => match &self.last_autoscroll {
                Some((_, _, _, strategy)) => (Autoscroll::Strategy(*strategy, None), local),
                None => return,
            },
            request => request,
        };
        self.provisional_autoscroll = Some(ProvisionalAutoscroll {
            request,
            disjoint,
            pending,
            anchor: self.anchor.read(cx).scroll_anchor,
        });
    }

    pub(crate) fn take_provisional_autoscroll(
        &mut self,
        disjoint: &Arc<[Selection<Anchor>]>,
        pending: Option<&Selection<Anchor>>,
        cx: &App,
    ) -> Option<(Autoscroll, bool)> {
        let provisional = self.provisional_autoscroll.take()?;
        (Arc::ptr_eq(&provisional.disjoint, disjoint)
            && provisional.pending.as_ref() == pending
            && provisional.anchor == self.anchor.read(cx).scroll_anchor)
            .then_some(provisional.request)
    }

    pub(crate) fn horizontal_anchor_measure_request(
        &self,
        snapshot: &DisplaySnapshot,
        cx: &App,
    ) -> Option<MeasureRequest> {
        let (long_row, column, _) = self.anchor.read(cx).horizontal_anchor_position(snapshot)?;
        long_row.measure_request(&long_row.window_around(column), &[])
    }

    pub(crate) fn resolve_horizontal_anchor(
        &mut self,
        snapshot: &DisplaySnapshot,
        em_layout_width: Pixels,
        cx: &mut Context<Editor>,
    ) -> bool {
        let Some(is_native) = self.resolve_horizontal_scroll_x(snapshot, em_layout_width, cx)
        else {
            return false;
        };
        if is_native {
            cx.emit(EditorEvent::ScrollPositionChanged {
                local: self.is_local_position(cx),
                autoscroll: false,
            });
        }
        true
    }

    pub(crate) fn is_local_position(&self, cx: &App) -> bool {
        self.anchor.read(cx).local
    }

    fn resolve_horizontal_scroll_x(
        &mut self,
        snapshot: &DisplaySnapshot,
        em_layout_width: Pixels,
        cx: &mut Context<Editor>,
    ) -> Option<bool> {
        let em_layout_width = ScrollPixelOffset::from(em_layout_width);
        let shared = *self.anchor.read(cx);
        let horizontal = shared.horizontal?;
        if em_layout_width <= 0. {
            return None;
        }
        let previous_x = self.effective_offset(&shared).x;
        let is_native = horizontal.display_map_id == snapshot.display_map_id;
        let Some(scroll_left) = shared.horizontal_scroll_left(snapshot) else {
            self.horizontal_scroll_x
                .take_if(|(resolved, _)| *resolved == horizontal)?;
            return (self.effective_offset(&shared).x != previous_x).then_some(is_native);
        };
        let x = (scroll_left / em_layout_width).max(0.);
        self.horizontal_scroll_x = Some((horizontal, x));
        if is_native && shared.scroll_anchor.offset.x != x {
            self.anchor
                .update(cx, |shared, _| shared.scroll_anchor.offset.x = x);
        }
        (self.effective_offset(&shared).x != previous_x).then_some(is_native)
    }

    pub(crate) fn store_horizontal_anchor(
        &mut self,
        anchor: Option<(Anchor, ScrollPixelOffset)>,
        snapshot: &DisplaySnapshot,
        cx: &mut Context<Editor>,
    ) {
        let shared = *self.anchor.read(cx);
        if shared
            .display_map_id
            .is_some_and(|writer| writer != snapshot.display_map_id)
        {
            return;
        }
        if let Some(current) = shared.horizontal {
            let resolved = self
                .horizontal_scroll_x
                .is_some_and(|(resolved, _)| resolved == current);
            if resolved || current.display_map_id != snapshot.display_map_id {
                return;
            }
        }
        let horizontal = anchor.map(|(anchor, offset)| HorizontalScrollAnchor {
            anchor,
            offset,
            display_map_id: snapshot.display_map_id,
        });
        if horizontal == shared.horizontal {
            return;
        }
        let offset = self.effective_offset(&shared);
        self.anchor
            .update(cx, |shared, _| shared.horizontal = horizontal);
        self.horizontal_scroll_x = horizontal.map(|horizontal| (horizontal, offset.x));
    }

    fn restore_anchor(
        &mut self,
        saved: SharedScrollAnchor,
        display_map: &DisplaySnapshot,
        window: &mut Window,
        cx: &mut Context<Editor>,
    ) -> WasScrolled {
        let is_resolvable = |display_map_id: EntityId| {
            display_map_id == display_map.display_map_id
                || display_map
                    .companion_snapshot()
                    .is_some_and(|companion| companion.display_map_id == display_map_id)
        };
        if saved
            .display_map_id
            .is_some_and(|display_map_id| !is_resolvable(display_map_id))
        {
            return WasScrolled(false);
        }
        let current = *self.anchor.read(cx);
        let mut restored = SharedScrollAnchor {
            horizontal: saved
                .horizontal
                .filter(|horizontal| is_resolvable(horizontal.display_map_id)),
            local: true,
            ..saved
        };
        if self.forbid_vertical_scroll {
            restored.scroll_anchor = ScrollAnchor {
                anchor: current.scroll_anchor.anchor,
                offset: point(saved.scroll_anchor.offset.x, current.scroll_anchor.offset.y),
            };
            restored.display_map_id = current.display_map_id;
        }

        self.scroll_max_x.take();
        self.autoscroll_request.take();

        if current.scroll_anchor == restored.scroll_anchor
            && current.display_map_id == restored.display_map_id
            && current.horizontal == restored.horizontal
        {
            return WasScrolled(false);
        }

        self.anchor.update(cx, |shared, _| *shared = restored);
        self.horizontal_scroll_x = None;
        cx.emit(EditorEvent::ScrollPositionChanged {
            local: true,
            autoscroll: false,
        });
        self.show_scrollbars(window, cx);
        cx.notify();

        WasScrolled(true)
    }

    pub fn active_scrollbar_state(&self) -> Option<&ActiveScrollbarState> {
        self.active_scrollbar.as_ref()
    }

    pub fn dragging_scrollbar_axis(&self) -> Option<Axis> {
        self.active_scrollbar
            .as_ref()
            .filter(|scrollbar| scrollbar.thumb_state == ScrollbarThumbState::Dragging)
            .map(|scrollbar| scrollbar.axis)
    }

    pub fn any_scrollbar_dragged(&self) -> bool {
        self.active_scrollbar
            .as_ref()
            .is_some_and(|scrollbar| scrollbar.thumb_state == ScrollbarThumbState::Dragging)
    }

    pub fn set_hovered_scroll_thumb_axis(&mut self, axis: Axis, cx: &mut Context<Editor>) {
        self.update_active_scrollbar_state(
            Some(ActiveScrollbarState::new(
                axis,
                ScrollbarThumbState::Hovered,
            )),
            cx,
        );
    }

    pub fn set_dragged_scroll_thumb_axis(&mut self, axis: Axis, cx: &mut Context<Editor>) {
        self.update_active_scrollbar_state(
            Some(ActiveScrollbarState::new(
                axis,
                ScrollbarThumbState::Dragging,
            )),
            cx,
        );
    }

    pub fn reset_scrollbar_state(&mut self, cx: &mut Context<Editor>) {
        self.update_active_scrollbar_state(None, cx);
    }

    fn update_active_scrollbar_state(
        &mut self,
        new_state: Option<ActiveScrollbarState>,
        cx: &mut Context<Editor>,
    ) {
        if self.active_scrollbar != new_state {
            self.active_scrollbar = new_state;
            cx.notify();
        }
    }

    pub fn set_is_hovering_minimap_thumb(&mut self, hovered: bool, cx: &mut Context<Editor>) {
        self.update_minimap_thumb_state(
            Some(if hovered {
                ScrollbarThumbState::Hovered
            } else {
                ScrollbarThumbState::Idle
            }),
            cx,
        );
    }

    pub fn set_is_dragging_minimap(&mut self, cx: &mut Context<Editor>) {
        self.update_minimap_thumb_state(Some(ScrollbarThumbState::Dragging), cx);
    }

    pub fn hide_minimap_thumb(&mut self, cx: &mut Context<Editor>) {
        self.update_minimap_thumb_state(None, cx);
    }

    pub fn is_dragging_minimap(&self) -> bool {
        self.minimap_thumb_state
            .is_some_and(|state| state == ScrollbarThumbState::Dragging)
    }

    fn update_minimap_thumb_state(
        &mut self,
        thumb_state: Option<ScrollbarThumbState>,
        cx: &mut Context<Editor>,
    ) {
        if self.minimap_thumb_state != thumb_state {
            self.minimap_thumb_state = thumb_state;
            cx.notify();
        }
    }

    pub fn minimap_thumb_state(&self) -> Option<ScrollbarThumbState> {
        self.minimap_thumb_state
    }

    pub fn clamp_scroll_left(&mut self, max: f64, cx: &App) -> bool {
        self.scroll_max_x = None;
        let current_x = self.effective_offset(self.anchor.read(cx)).x;
        self.scroll_max_x = Some(max);
        current_x > max
    }

    pub fn set_forbid_vertical_scroll(&mut self, forbid: bool) {
        self.forbid_vertical_scroll = forbid;
    }

    pub fn forbid_vertical_scroll(&self) -> bool {
        self.forbid_vertical_scroll
    }
}

impl Editor {
    pub fn has_autoscroll_request(&self) -> bool {
        self.scroll_manager.has_autoscroll_request()
    }

    pub fn set_forbid_vertical_scroll(&mut self, forbid: bool) {
        self.scroll_manager.set_forbid_vertical_scroll(forbid);
    }

    pub fn scroll_top_row(&self, snapshot: &DisplaySnapshot, cx: &App) -> DisplayRow {
        self.scroll_manager.scroll_top_row(snapshot, cx)
    }

    pub fn vertical_scroll_margin(&self) -> usize {
        self.scroll_manager.vertical_scroll_margin as usize
    }

    pub(crate) fn scroll_beyond_last_line(&self, cx: &App) -> ScrollBeyondLastLine {
        match self.mode {
            EditorMode::Minimap { .. }
            | EditorMode::Full {
                sizing_behavior: SizingBehavior::Default,
                ..
            } => EditorSettings::get_global(cx).scroll_beyond_last_line,

            EditorMode::Full { .. } | EditorMode::SingleLine | EditorMode::AutoHeight { .. } => {
                ScrollBeyondLastLine::Off
            }
        }
    }

    pub fn set_vertical_scroll_margin(&mut self, margin_rows: usize, cx: &mut Context<Self>) {
        self.scroll_manager.vertical_scroll_margin = margin_rows as f64;
        cx.notify();
    }

    pub fn visible_line_count(&self) -> Option<f64> {
        self.scroll_manager.visible_line_count
    }

    pub fn visible_row_count(&self) -> Option<u32> {
        self.visible_line_count()
            .map(|line_count| line_count as u32 - 1)
    }

    pub fn visible_column_count(&self) -> Option<f64> {
        self.scroll_manager.visible_column_count
    }

    pub(crate) fn set_visible_line_count(
        &mut self,
        lines: f64,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        match self.scroll_manager.visible_line_count.replace(lines) {
            None => self.update_data_on_scroll(false, window, cx),
            Some(previous_lines) if previous_lines < lines => {
                self.update_data_on_scroll(true, window, cx)
            }
            Some(_) => {}
        }
    }

    pub(crate) fn set_visible_column_count(&mut self, columns: f64) -> bool {
        self.scroll_manager
            .visible_column_count
            .replace(columns)
            .is_some_and(|previous_columns| previous_columns < columns)
    }

    pub fn apply_scroll_delta(
        &mut self,
        scroll_delta: gpui::Point<f32>,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        let mut delta = scroll_delta;
        if self.scroll_manager.forbid_vertical_scroll {
            delta.y = 0.0;
        }
        let display_map = self.display_map.update(cx, |map, cx| map.snapshot(cx));
        let position = self.scroll_manager.scroll_position(&display_map, cx) + delta.map(f64::from);
        self.set_scroll_position_taking_display_map(position, true, false, display_map, window, cx);
    }

    pub fn set_scroll_position(
        &mut self,
        scroll_position: gpui::Point<ScrollOffset>,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) -> WasScrolled {
        let mut position = scroll_position;
        if self.scroll_manager.forbid_vertical_scroll {
            let current_position = self.scroll_position(cx);
            position.y = current_position.y;
        }
        self.set_scroll_position_internal(position, true, false, window, cx)
    }

    /// Scrolls so that `row` is at the top of the editor view.
    pub fn set_scroll_top_row(
        &mut self,
        row: DisplayRow,
        window: &mut Window,
        cx: &mut Context<Editor>,
    ) {
        let snapshot = self.snapshot(window, cx).display_snapshot;
        let new_screen_top = DisplayPoint::new(row, 0);
        let new_screen_top = new_screen_top.to_offset(&snapshot, Bias::Left);
        let new_anchor = snapshot.buffer_snapshot().anchor_before(new_screen_top);

        self.set_scroll_anchor(
            ScrollAnchor {
                anchor: new_anchor,
                offset: Default::default(),
            },
            window,
            cx,
        );
    }

    pub(crate) fn set_scroll_position_internal(
        &mut self,
        scroll_position: gpui::Point<ScrollOffset>,
        local: bool,
        autoscroll: bool,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) -> WasScrolled {
        let map = self.display_map.update(cx, |map, cx| map.snapshot(cx));
        let was_scrolled = self.set_scroll_position_taking_display_map(
            scroll_position,
            local,
            autoscroll,
            map,
            window,
            cx,
        );

        was_scrolled
    }

    fn set_scroll_position_taking_display_map(
        &mut self,
        scroll_position: gpui::Point<ScrollOffset>,
        local: bool,
        autoscroll: bool,
        display_map: DisplaySnapshot,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) -> WasScrolled {
        hide_hover(self, cx);

        self.edit_prediction_preview
            .set_previous_scroll_position(None);

        let adjusted_position = if self.scroll_manager.forbid_vertical_scroll {
            let current_position = self.scroll_manager.scroll_position(&display_map, cx);
            gpui::Point::new(scroll_position.x, current_position.y)
        } else {
            scroll_position
        };
        let scroll_beyond_last_line = self.scroll_beyond_last_line(cx);
        self.scroll_manager.set_scroll_position(
            adjusted_position,
            &display_map,
            scroll_beyond_last_line,
            local,
            autoscroll,
            window,
            cx,
        )
    }

    pub fn scroll_position(&self, cx: &mut Context<Self>) -> gpui::Point<ScrollOffset> {
        let display_map = self.display_map.update(cx, |map, cx| map.snapshot(cx));
        self.scroll_manager.scroll_position(&display_map, cx)
    }

    pub fn set_scroll_anchor(
        &mut self,
        scroll_anchor: ScrollAnchor,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        hide_hover(self, cx);
        let display_map = self.display_map.update(cx, |map, cx| map.snapshot(cx));
        self.scroll_manager
            .set_anchor(scroll_anchor, &display_map, true, false, window, cx);
    }

    pub(crate) fn persist_scroll_position(
        &mut self,
        snapshot: &DisplaySnapshot,
        cx: &mut Context<Self>,
    ) {
        let position = self.scroll_manager.serialized_position(snapshot, cx);
        let workspace_id = self.workspace.as_ref().and_then(|workspace| workspace.1);
        if self
            .scroll_manager
            .persist_position(position, workspace_id, cx)
            && self.scroll_manager.is_local_position(cx)
        {
            self.update_restoration_data(cx, move |data| data.scroll_position = position);
        }
    }

    pub(crate) fn restore_scroll_position(
        &mut self,
        position: SerializedScrollPosition,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        let anchor = self
            .buffer()
            .read(cx)
            .snapshot(cx)
            .anchor_before(Point::new(position.top_row, 0));
        self.restore_native_scroll_position(
            ScrollAnchor {
                anchor,
                offset: position.offset,
            },
            position.horizontal,
            window,
            cx,
        );
    }

    pub(crate) fn restore_native_scroll_position(
        &mut self,
        scroll_anchor: ScrollAnchor,
        horizontal: Option<(Point, ScrollPixelOffset)>,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        let buffer = self.buffer().read(cx).snapshot(cx);
        let display_map_id = self.display_map.entity_id();
        let saved = SharedScrollAnchor {
            scroll_anchor,
            display_map_id: Some(display_map_id),
            horizontal: horizontal.map(|(point, offset)| HorizontalScrollAnchor {
                anchor: buffer.anchor_before(buffer.clip_point(point, Bias::Left)),
                offset,
                display_map_id,
            }),
            local: true,
        };
        self.restore_scroll_anchor(saved, window, cx);
    }

    pub(crate) fn save_scroll_anchor(&self, cx: &App) -> SharedScrollAnchor {
        self.scroll_manager.shared_scroll_anchor(cx)
    }

    pub(crate) fn restore_scroll_anchor(
        &mut self,
        saved: SharedScrollAnchor,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) -> WasScrolled {
        hide_hover(self, cx);
        self.edit_prediction_preview
            .set_previous_scroll_position(None);
        let display_map = self.display_map.update(cx, |map, cx| map.snapshot(cx));
        self.scroll_manager
            .restore_anchor(saved, &display_map, window, cx)
    }

    pub(crate) fn set_scroll_anchor_remote(
        &mut self,
        scroll_anchor: ScrollAnchor,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        hide_hover(self, cx);
        let buffer_snapshot = self.buffer().read(cx).snapshot(cx);
        if !scroll_anchor.anchor.is_valid(&buffer_snapshot) {
            log::warn!("Invalid scroll anchor: {:?}", scroll_anchor);
            return;
        }
        let display_map = self.display_map.update(cx, |map, cx| map.snapshot(cx));
        self.scroll_manager
            .set_anchor(scroll_anchor, &display_map, false, false, window, cx);
    }

    pub fn scroll_screen(
        &mut self,
        amount: &ScrollAmount,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        if matches!(self.mode, EditorMode::SingleLine) {
            cx.propagate();
            return;
        }

        if self.take_rename(true, window, cx).is_some() {
            return;
        }

        let mut current_position = self.scroll_position(cx);
        let Some(visible_line_count) = self.visible_line_count() else {
            return;
        };
        let Some(mut visible_column_count) = self.visible_column_count() else {
            return;
        };
        if let Some((em_layout_width, em_advance)) = self.em_widths(cx) {
            visible_column_count *= em_advance / em_layout_width;
        }

        // If the user has a preferred line length, and has the editor
        // configured to wrap at the preferred line length, or bounded to it,
        // use that value over the visible column count. This was mostly done so
        // that tests could actually be written for vim's `z l`, `z h`, `z
        // shift-l` and `z shift-h` commands, as there wasn't a good way to
        // configure the editor to only display a certain number of columns. If
        // that ever happens, this could probably be removed.
        let settings = AllLanguageSettings::get_global(cx);
        if matches!(settings.defaults.soft_wrap, SoftWrap::Bounded)
            && (settings.defaults.preferred_line_length as f64) < visible_column_count
        {
            visible_column_count = settings.defaults.preferred_line_length as f64;
        }

        // If the scroll position is currently at the left edge of the document
        // (x == 0.0) and the intent is to scroll right, the gutter's margin
        // should first be added to the current position, otherwise the cursor
        // will end at the column position minus the margin, which looks off.
        if current_position.x == 0.0
            && amount.columns(visible_column_count) > 0.
            && let Some(last_position_map) = &self.last_position_map
        {
            current_position.x +=
                f64::from(self.gutter_dimensions.margin / last_position_map.em_layout_width);
        }
        let new_position = current_position
            + point(
                amount.columns(visible_column_count),
                amount.lines(visible_line_count),
            );
        self.set_scroll_position(new_position, window, cx);
    }

    pub fn scroll_screen_with_cursor_margin(
        &mut self,
        amount: &ScrollAmount,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        self.scroll_screen(amount, window, cx);

        let Some(visible_line_count) = self.visible_line_count() else {
            return;
        };
        let display_snapshot = self.display_map.update(cx, |map, cx| map.snapshot(cx));
        let top = self.scroll_manager.scroll_top_row(&display_snapshot, cx);
        let vertical_scroll_margin =
            (self.vertical_scroll_margin() as u32).min(visible_line_count as u32 / 2);

        let max_point = display_snapshot.max_point();
        let min_row = if top.0 == 0 {
            DisplayRow(0)
        } else {
            DisplayRow(top.0 + vertical_scroll_margin)
        };
        let max_row = if top.0 + visible_line_count as u32 >= max_point.row().0 {
            max_point.row()
        } else {
            DisplayRow(
                (top.0 + visible_line_count as u32).saturating_sub(1 + vertical_scroll_margin),
            )
        };

        self.change_selections(
            SelectionEffects::no_scroll().nav_history(false),
            window,
            cx,
            |s| {
                s.move_with(&mut |map, selection| {
                    let head = selection.head();
                    let new_row = if head.row() < min_row {
                        min_row
                    } else if head.row() > max_row {
                        max_row
                    } else {
                        head.row()
                    };
                    if new_row != head.row() {
                        let new_head =
                            map.clip_point(DisplayPoint::new(new_row, head.column()), Bias::Left);
                        selection.collapse_to(new_head, selection.goal);
                    }
                })
            },
        );
    }

    /// Returns an ordering. The newest selection is:
    ///     Ordering::Equal => on screen
    ///     Ordering::Less => above or to the left of the screen
    ///     Ordering::Greater => below or to the right of the screen
    pub fn newest_selection_on_screen(&self, window: &mut Window, cx: &mut App) -> Ordering {
        let snapshot = self.display_map.update(cx, |map, cx| map.snapshot(cx));
        let newest_head = self
            .selections
            .newest_anchor()
            .head()
            .to_display_point(&snapshot);
        let screen_top = self.scroll_manager.scroll_top_row(&snapshot, cx);
        let text_layout_details = self.text_layout_details(window, cx);
        let column = newest_head.column();
        let horizontal_position = |visible_columns: RangeInclusive<u32>| {
            if column < *visible_columns.start() {
                Ordering::Less
            } else if column > *visible_columns.end() {
                Ordering::Greater
            } else {
                Ordering::Equal
            }
        };
        let horizontal = match (
            self.visible_x_range(&snapshot, cx),
            self.visible_column_count(),
        ) {
            (Some(visible_x), _) => match snapshot.visible_display_columns(
                newest_head.row(),
                visible_x.clone(),
                &text_layout_details,
            ) {
                Some(visible_columns) => horizontal_position(visible_columns),
                None => {
                    let head_x = snapshot.x_for_display_point(newest_head, &text_layout_details);
                    if head_x < *visible_x.start() {
                        Ordering::Less
                    } else {
                        Ordering::Greater
                    }
                }
            },
            (None, visible_column_count) => {
                let left_column = self.scroll_manager.scroll_position(&snapshot, cx).x as u32;
                match visible_column_count {
                    Some(visible_column_count) => {
                        horizontal_position(left_column..=left_column + visible_column_count as u32)
                    }
                    None if column < left_column => Ordering::Less,
                    None => Ordering::Greater,
                }
            }
        };

        if newest_head.row() < screen_top {
            return Ordering::Less;
        }
        match self.visible_line_count() {
            Some(visible_lines)
                if newest_head.row() > DisplayRow(screen_top.0 + visible_lines as u32) =>
            {
                Ordering::Greater
            }
            Some(_) => horizontal,
            None if horizontal.is_lt() => Ordering::Less,
            None => Ordering::Greater,
        }
    }

    pub fn visible_x_range(
        &self,
        snapshot: &DisplaySnapshot,
        cx: &App,
    ) -> Option<RangeInclusive<ScrollPixelOffset>> {
        let visible_columns = self.visible_column_count()?;
        let (em_layout_width, em_advance) = self.em_widths(cx)?;
        let left_x = self.scroll_manager.scroll_position(snapshot, cx).x * em_layout_width;
        Some(left_x..=left_x + em_advance * visible_columns)
    }

    pub(crate) fn horizontal_scroll_pixels(
        &self,
        scroll_x: ScrollOffset,
        window: &Window,
        cx: &App,
    ) -> ScrollPixelOffset {
        self.em_widths(cx).map_or(0., |(em_layout_width, _)| {
            window.pixel_snap_f64(scroll_x * em_layout_width)
        })
    }

    pub fn read_scroll_position_from_db(
        &mut self,
        item_id: u64,
        workspace_id: WorkspaceId,
        window: &mut Window,
        cx: &mut Context<Editor>,
    ) {
        let scroll_position = EditorDb::global(cx).get_scroll_position(item_id, workspace_id);
        if let Ok(Some((top_row, x, y, anchor_row, anchor_column, anchor_offset))) = scroll_position
        {
            let horizontal = match (anchor_row, anchor_column, anchor_offset) {
                (Some(row), Some(column), Some(offset)) => Some((Point::new(row, column), offset)),
                _ => None,
            };
            self.restore_scroll_position(
                SerializedScrollPosition {
                    top_row,
                    offset: gpui::Point::new(x, y),
                    horizontal,
                },
                window,
                cx,
            );
        }
    }

    fn em_widths(&self, cx: &App) -> Option<(ScrollPixelOffset, ScrollPixelOffset)> {
        let (font, font_size) = self.display_map.read(cx).font_with_size(cx);
        let text_system = cx.text_system();
        let font_id = text_system.resolve_font(font);
        let em_advance = text_system.em_advance(font_id, *font_size).log_err()?;
        Some((
            ScrollPixelOffset::from(text_system.layout_width(font_id, *font_size, 'm')),
            ScrollPixelOffset::from(em_advance),
        ))
    }
}

pub(crate) fn companion_display_point(
    snapshot: &DisplaySnapshot,
    companion: &DisplaySnapshot,
    companion_point: Point,
    bias: Bias,
) -> (DisplayPoint, bool) {
    let companion_display_point = companion.point_to_display_point(companion_point, bias);
    let row = companion_display_point.row();
    if snapshot.long_row(row).is_none() || companion.long_row(row).is_none() {
        return (snapshot.clip_point(companion_display_point, bias), false);
    }
    let buffer = snapshot.buffer_snapshot();
    let buffer_row = snapshot
        .display_point_to_point(DisplayPoint::new(row, 0), Bias::Left)
        .row;
    let line_len = buffer.line_len(MultiBufferRow(buffer_row));
    let point = buffer.clip_point(
        Point::new(buffer_row, companion_point.column.min(line_len)),
        bias,
    );
    (
        snapshot.point_to_display_point(point, bias),
        companion_point.column > line_len,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{MultiBuffer, editor_tests::init_test, test::ShapingTextSystem};
    use gpui::{TestAppContext, VisualTestContext};

    #[gpui::test]
    fn test_horizontal_page_scroll_uses_layout_units() {
        let mut cx = TestAppContext::build_with_text_system(
            gpui::TestDispatcher::new(0),
            None,
            Arc::new(ShapingTextSystem(gpui::NoopTextSystem)),
        );
        init_test(&mut cx, |_| {});
        let window = cx.add_window(|window, cx| {
            let buffer = MultiBuffer::build_simple(&"x".repeat(4_000), cx);
            Editor::new(EditorMode::full(), buffer, None, window, cx)
        });
        let cx = &mut VisualTestContext::from_window(*window, &mut cx);
        let editor = window.root(cx).expect("editor");
        editor.update_in(cx, |editor, window, cx| {
            editor.scroll_manager.visible_line_count = Some(10.);
            editor.scroll_manager.visible_column_count = Some(100.5);
            for start in [10., 4_294_967_296.25] {
                editor.set_scroll_position(point(start, 0.), window, cx);
                editor.scroll_screen(&ScrollAmount::PageWidth(0.5), window, cx);
                assert_eq!(editor.scroll_position(cx), point(start + 60., 0.));
            }
        });
    }
}
