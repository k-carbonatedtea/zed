use std::{collections::VecDeque, fmt, mem, sync::Arc};

use collections::HashMap;
use language::{Bias, Point, Selection, SelectionGoal};
use multi_buffer::ToPoint as _;
use parking_lot::Mutex;
use smallvec::{SmallVec, smallvec};

use crate::{
    Anchor, DisplayPoint, DisplayRow,
    display_map::{DisplaySnapshot, ToDisplayPoint as _},
    movement::TextLayoutDetails,
    scroll::ScrollPixelOffset,
};

#[derive(Clone, Default)]
pub(crate) struct HorizontalOrigins(Arc<Mutex<RecordedOrigins>>);

#[derive(Clone, Default)]
pub(super) struct HorizontalGoals {
    origins: HorizontalOrigins,
    details: Option<Arc<TextLayoutDetails>>,
    epoch: usize,
    edit_count: usize,
    provisional: Arc<HashMap<usize, Provisional>>,
}

impl HorizontalOrigins {
    pub(crate) fn record(&self, x: ScrollPixelOffset, origin: Anchor, details: &TextLayoutDetails) {
        let mut recorded = self.0.lock();
        let origins = if let Some(selection) = recorded.selection {
            recorded.selection_origins.entry(selection).or_default()
        } else {
            if recorded.origins.len() == MAX_RECORDED_ORIGINS {
                recorded.origins.pop_front();
            }
            &mut recorded.origins
        };
        origins.push_back((x.to_bits(), origin));
        if recorded.details.is_none() {
            recorded.details = Some(details.detached());
        }
    }

    pub(crate) fn set_selection(&self, selection: Option<usize>) {
        let mut recorded = self.0.lock();
        recorded.selection = selection;
        if let Some(selection) = selection {
            recorded.selection_origins.entry(selection).or_default();
        }
    }

    fn take(&self) -> RecordedOrigins {
        mem::take(&mut *self.0.lock())
    }

    fn is_empty(&self) -> bool {
        let recorded = self.0.lock();
        recorded.origins.is_empty() && recorded.selection_origins.is_empty()
    }
}

impl fmt::Debug for HorizontalOrigins {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("HorizontalOrigins").finish_non_exhaustive()
    }
}

impl HorizontalGoals {
    pub(super) fn origins(&self) -> HorizontalOrigins {
        self.origins.clone()
    }

    pub(super) fn has_provisional(&self) -> bool {
        !self.provisional.is_empty()
    }

    pub(super) fn is_unsettled(&self, snapshot: &DisplaySnapshot) -> bool {
        self.has_provisional() && self.epoch != snapshot.long_row_epoch()
    }

    pub(super) fn has_recorded_origins(&self) -> bool {
        !self.origins.is_empty()
    }

    pub(super) fn clear_provisional(&mut self) {
        self.provisional = Arc::default();
    }

    pub(super) fn copy_goal(&mut self, from: usize, to: usize) {
        if let Some(provisional) = self.provisional.get(&from).cloned() {
            Arc::make_mut(&mut self.provisional).insert(to, provisional);
        }
        let mut recorded = self.origins.0.lock();
        let origins = recorded
            .selection_origins
            .get(&from)
            .cloned()
            .unwrap_or_default();
        recorded.selection_origins.insert(to, origins);
    }

    pub(super) fn set_details(&mut self, details: &TextLayoutDetails) {
        self.details = Some(Arc::new(details.detached()));
    }

    pub(super) fn capture<'a>(
        &mut self,
        selections: impl IntoIterator<Item = &'a Selection<Anchor>>,
        snapshot: &DisplaySnapshot,
    ) {
        let recorded = self.origins.take();
        self.capture_with(recorded, selections, snapshot, true);
    }

    pub(super) fn settle(
        &mut self,
        selections: &[Selection<Anchor>],
        snapshot: &DisplaySnapshot,
    ) -> Option<Vec<Selection<Point>>> {
        if !self.is_unsettled(snapshot) {
            return None;
        }
        self.epoch = snapshot.long_row_epoch();
        let buffer = snapshot.buffer_snapshot();
        let edited = buffer.edit_count() != self.edit_count;
        let mut remaining = HashMap::default();
        let mut changed = false;
        let settled = selections
            .iter()
            .map(|selection| {
                let resolved = selection.map(|anchor| anchor.to_point(buffer));
                let Some(provisional) = self
                    .provisional
                    .get(&selection.id)
                    .filter(|provisional| provisional.selection == *selection)
                else {
                    return resolved;
                };
                let positions = provisional
                    .positions
                    .iter()
                    .map(|position| self.settled_position(*position, snapshot))
                    .collect::<SmallVec<[GoalPosition; 2]>>();
                let mut moved = resolved;
                moved.goal = map_goal(selection.goal, &provisional.positions, &positions);
                let mut unsettled = positions.iter().any(|position| position.origin.is_some());
                let derived = if edited {
                    SmallVec::new()
                } else {
                    provisional.derived.clone()
                };
                for derived in &derived {
                    let anchor = match derived.endpoint {
                        Endpoint::Start => selection.start,
                        Endpoint::End => selection.end,
                    };
                    let row = anchor.to_display_point(snapshot).row();
                    let Some(point) = positions
                        .get(derived.position)
                        .and_then(|position| self.resolve(snapshot, row, position.x, derived.bias))
                    else {
                        continue;
                    };
                    unsettled |= snapshot
                        .long_row(point.row())
                        .is_some_and(|long_row| !long_row.is_settled_through(point.column()));
                    let point = point.to_point(snapshot);
                    match derived.endpoint {
                        Endpoint::Start => moved.start = point,
                        Endpoint::End => moved.end = point,
                    }
                }
                changed |= moved != resolved;
                if unsettled {
                    remaining.insert(
                        selection.id,
                        Provisional {
                            selection: *selection,
                            positions,
                            derived,
                        },
                    );
                }
                moved
            })
            .collect::<Vec<_>>();
        self.provisional = Arc::new(remaining);
        changed.then_some(settled)
    }

    pub(super) fn capture_settled<'a>(
        &mut self,
        selections: impl IntoIterator<Item = &'a Selection<Anchor>>,
        snapshot: &DisplaySnapshot,
    ) {
        let derive = snapshot.buffer_snapshot().edit_count() == self.edit_count;
        self.capture_with(RecordedOrigins::default(), selections, snapshot, derive);
    }

    fn capture_with<'a>(
        &mut self,
        recorded: RecordedOrigins,
        selections: impl IntoIterator<Item = &'a Selection<Anchor>>,
        snapshot: &DisplaySnapshot,
        derive: bool,
    ) {
        if let Some(details) = recorded.details {
            self.details = Some(Arc::new(details));
        }
        self.epoch = snapshot.long_row_epoch();
        self.edit_count = snapshot.buffer_snapshot().edit_count();
        if recorded.origins.is_empty()
            && recorded.selection_origins.is_empty()
            && self.provisional.is_empty()
            && !snapshot.may_have_long_rows()
        {
            return;
        }
        let previous = mem::take(&mut self.provisional);

        let buffer = snapshot.buffer_snapshot();
        let mut provisional = HashMap::default();
        for selection in selections {
            let xs = goal_positions(selection.goal);
            if xs.is_empty() {
                continue;
            }
            let own = previous.get(&selection.id);
            let recorded_origins = recorded
                .selection_origins
                .get(&selection.id)
                .unwrap_or(&recorded.origins);
            let positions = xs
                .into_iter()
                .map(|x| {
                    let bits = x.to_bits();
                    let origin = recorded_origins
                        .iter()
                        .rev()
                        .find(|(x, _)| *x == bits)
                        .map(|(_, origin)| *origin)
                        .or_else(|| {
                            own.and_then(|own| {
                                own.positions
                                    .iter()
                                    .find(|position| position.x.to_bits() == bits)
                                    .and_then(|position| position.origin)
                            })
                        })
                        .filter(|origin| buffer.can_resolve(origin));
                    GoalPosition { x, origin }
                })
                .collect::<SmallVec<[GoalPosition; 2]>>();
            let derived = if derive {
                self.derived_endpoints(selection, &positions, snapshot)
            } else {
                SmallVec::new()
            };
            if positions.iter().any(|position| position.origin.is_some()) || !derived.is_empty() {
                provisional.insert(
                    selection.id,
                    Provisional {
                        selection: *selection,
                        positions,
                        derived,
                    },
                );
            }
        }
        self.provisional = Arc::new(provisional);
    }

    fn derived_endpoints(
        &self,
        selection: &Selection<Anchor>,
        positions: &[GoalPosition],
        snapshot: &DisplaySnapshot,
    ) -> SmallVec<[Derived; 2]> {
        let mut derived = SmallVec::new();
        let is_empty = selection.start.to_point(snapshot.buffer_snapshot())
            == selection.end.to_point(snapshot.buffer_snapshot());
        for (endpoint, anchor) in [
            (Endpoint::Start, selection.start),
            (Endpoint::End, selection.end),
        ] {
            if !is_empty
                && !matches!(selection.goal, SelectionGoal::HorizontalRange { .. })
                && anchor == selection.tail()
            {
                continue;
            }
            let point = anchor.to_display_point(snapshot);
            let unsettled_row = snapshot
                .long_row(point.row())
                .is_some_and(|long_row| !long_row.is_settled_through(point.column()));
            let found = positions.iter().enumerate().find_map(|(index, position)| {
                if position.origin.is_none() && !unsettled_row {
                    return None;
                }
                [Bias::Right, Bias::Left].into_iter().find_map(|bias| {
                    (self.resolve(snapshot, point.row(), position.x, bias)? == point).then_some(
                        Derived {
                            endpoint,
                            position: index,
                            bias,
                        },
                    )
                })
            });
            derived.extend(found);
        }
        derived
    }

    fn settled_position(&self, position: GoalPosition, snapshot: &DisplaySnapshot) -> GoalPosition {
        let Some(origin) = position
            .origin
            .filter(|origin| snapshot.buffer_snapshot().can_resolve(origin))
        else {
            return GoalPosition {
                origin: None,
                ..position
            };
        };
        let point = origin.to_display_point(snapshot);
        match snapshot.long_row(point.row()) {
            Some(long_row) => GoalPosition {
                x: long_row.measured_x_for_column(point.column()),
                origin: (!long_row.is_settled_through(point.column())).then_some(origin),
            },
            None => GoalPosition {
                x: self.details.as_deref().map_or(position.x, |details| {
                    snapshot
                        .layout_row(point.row(), details)
                        .x_for_index(point.column() as usize)
                }),
                origin: None,
            },
        }
    }

    fn resolve(
        &self,
        snapshot: &DisplaySnapshot,
        row: DisplayRow,
        x: ScrollPixelOffset,
        bias: Bias,
    ) -> Option<DisplayPoint> {
        let column = match snapshot.long_row(row) {
            Some(long_row) => long_row.measured_closest_column_for_x(x),
            None => snapshot
                .layout_row(row, self.details.as_deref()?)
                .closest_index_for_x(x) as u32,
        };
        Some(snapshot.clip_point(DisplayPoint::new(row, column), bias))
    }
}

impl fmt::Debug for HorizontalGoals {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("HorizontalGoals")
            .field("epoch", &self.epoch)
            .field("provisional", &self.provisional)
            .finish_non_exhaustive()
    }
}

const MAX_RECORDED_ORIGINS: usize = 256;

#[derive(Clone, Default)]
struct RecordedOrigins {
    origins: VecDeque<(u64, Anchor)>,
    selection_origins: HashMap<usize, VecDeque<(u64, Anchor)>>,
    selection: Option<usize>,
    details: Option<TextLayoutDetails>,
}

#[derive(Clone, Debug)]
struct Provisional {
    selection: Selection<Anchor>,
    positions: SmallVec<[GoalPosition; 2]>,
    derived: SmallVec<[Derived; 2]>,
}

#[derive(Clone, Copy, Debug)]
struct GoalPosition {
    x: ScrollPixelOffset,
    origin: Option<Anchor>,
}

#[derive(Clone, Copy, Debug)]
struct Derived {
    endpoint: Endpoint,
    position: usize,
    bias: Bias,
}

#[derive(Clone, Copy, Debug)]
enum Endpoint {
    Start,
    End,
}

fn goal_positions(goal: SelectionGoal) -> SmallVec<[ScrollPixelOffset; 2]> {
    let mut positions = match goal {
        SelectionGoal::None => SmallVec::<[ScrollPixelOffset; 2]>::new(),
        SelectionGoal::HorizontalPosition(x) | SelectionGoal::WrappedHorizontalPosition((_, x)) => {
            smallvec![x]
        }
        SelectionGoal::HorizontalRange { start, end } => smallvec![start, end],
    };
    positions.retain(|x| x.is_finite());
    positions
}

fn map_goal(goal: SelectionGoal, from: &[GoalPosition], to: &[GoalPosition]) -> SelectionGoal {
    let map = |x: ScrollPixelOffset| {
        from.iter()
            .zip(to)
            .find(|(from, _)| from.x.to_bits() == x.to_bits())
            .map_or(x, |(_, to)| to.x)
    };
    match goal {
        SelectionGoal::None => SelectionGoal::None,
        SelectionGoal::HorizontalPosition(x) => SelectionGoal::HorizontalPosition(map(x)),
        SelectionGoal::WrappedHorizontalPosition((row, x)) => {
            SelectionGoal::WrappedHorizontalPosition((row, map(x)))
        }
        SelectionGoal::HorizontalRange { start, end } => SelectionGoal::HorizontalRange {
            start: map(start),
            end: map(end),
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        AddSelectionAbove, AddSelectionBelow, Editor, EditorMode, MultiBuffer, SelectionEffects,
        editor_tests::init_test, test::ShapingTextSystem,
    };
    use gpui::{TestAppContext, VisualTestContext};

    #[gpui::test]
    fn test_selection_scoped_horizontal_goals(cx: &mut TestAppContext) {
        init_test(cx, |_| {});
        let editor = cx.add_window(|window, cx| {
            let buffer = MultiBuffer::build_simple("abcd\nabcd\nabcd\nabcd", cx);
            Editor::new(EditorMode::full(), buffer, None, window, cx)
        });
        editor
            .update(cx, |editor, window, cx| {
                let snapshot = editor.display_snapshot(cx);
                let details = editor.text_layout_details(window, cx);
                let buffer = snapshot.buffer_snapshot();
                let x = snapshot.x_for_display_point(DisplayPoint::new(DisplayRow(0), 4), &details);
                let mut selections = [
                    (0, Point::new(1, 4)..Point::new(1, 4)),
                    (1, Point::new(2, 4)..Point::new(3, 4)),
                ]
                .map(|(id, range)| Selection {
                    id,
                    start: buffer.anchor_before(range.start),
                    end: buffer.anchor_before(range.end),
                    reversed: false,
                    goal: SelectionGoal::HorizontalPosition(x),
                });
                let mut goals = HorizontalGoals::default();
                for (selection, source) in
                    selections.iter().zip([Point::new(0, 2), Point::new(2, 3)])
                {
                    goals.origins.set_selection(Some(selection.id));
                    goals
                        .origins
                        .record(x, buffer.anchor_before(source), &details);
                }
                goals.origins.set_selection(None);
                goals.copy_goal(0, 2);
                selections[0].id = 2;
                goals.capture(&selections, &snapshot);
                goals.copy_goal(2, 3);
                selections[0].id = 3;
                goals.capture(&selections, &snapshot);
                goals.epoch = snapshot.long_row_epoch().wrapping_sub(1);
                let settled = goals
                    .settle(&selections, &snapshot)
                    .expect("settled selections");
                assert_eq!(
                    settled
                        .iter()
                        .map(|selection| selection.start..selection.end)
                        .collect::<Vec<_>>(),
                    [
                        Point::new(1, 2)..Point::new(1, 2),
                        Point::new(2, 4)..Point::new(3, 3)
                    ]
                );

                for above in [false, true] {
                    for (tail, expected_tail) in [(4, 3), (2, 1)] {
                        let row = if above { 2 } else { 0 };
                        editor.change_selections(
                            SelectionEffects::no_scroll(),
                            window,
                            cx,
                            |selections| {
                                selections
                                    .select_ranges([Point::new(row, 4)..Point::new(row, tail)]);
                            },
                        );
                        for seed_origins in [false, true] {
                            if seed_origins {
                                for (column, source_column) in [(2, 1), (4, 3)] {
                                    let x = snapshot.x_for_display_point(
                                        DisplayPoint::new(DisplayRow(0), column),
                                        &details,
                                    );
                                    editor.selections.horizontal_origins().record(
                                        x,
                                        buffer.anchor_before(Point::new(3, source_column)),
                                        &details,
                                    );
                                }
                                editor.change_selections(
                                    SelectionEffects::no_scroll(),
                                    window,
                                    cx,
                                    |_| {},
                                );
                            }
                            if above {
                                editor.add_selection_above(
                                    &AddSelectionAbove {
                                        skip_soft_wrap: false,
                                    },
                                    window,
                                    cx,
                                );
                            } else {
                                editor.add_selection_below(
                                    &AddSelectionBelow {
                                        skip_soft_wrap: false,
                                    },
                                    window,
                                    cx,
                                );
                            }
                        }
                        editor.selections.horizontal_goals.epoch =
                            snapshot.long_row_epoch().wrapping_sub(1);
                        editor.change_selections(SelectionEffects::no_scroll(), window, cx, |_| {});
                        assert_eq!(
                            editor.selections.ranges::<Point>(&snapshot),
                            (0..3)
                                .map(|row| Point::new(row, 3)..Point::new(row, expected_tail))
                                .collect::<Vec<_>>()
                        );
                    }
                }
            })
            .expect("editor window");
    }

    #[gpui::test]
    fn test_initial_columnar_groups_keep_distinct_horizontal_origins() {
        for (above, tail_offset, head_offset, repeated) in
            [(false, 0, 0, false), (true, 0, 4, true)]
        {
            let mut cx = TestAppContext::build_with_text_system(
                gpui::TestDispatcher::new(0),
                None,
                Arc::new(ShapingTextSystem(gpui::NoopTextSystem)),
            );
            init_test(&mut cx, |_| {});
            cx.background_executor.set_block_on_ticks(0..=0);
            let source = format!(
                "{}{}{}",
                "x".repeat(33_000),
                "w".repeat(3_000),
                "x".repeat(33_000)
            );
            let plain = "x".repeat(69_000);
            let rows_per_group = if repeated { 3 } else { 2 };
            let source_row = if above { rows_per_group - 1 } else { 0 };
            let mut rows = vec![plain.as_str(); (2 * rows_per_group) as usize];
            rows[source_row as usize] = &source;
            let window = cx.add_window(|window, cx| {
                let buffer = MultiBuffer::build_simple(&rows.join("\n"), cx);
                Editor::new(EditorMode::full(), buffer, None, window, cx)
            });
            let cx = &mut VisualTestContext::from_window(*window, &mut cx);
            let editor = window.root(cx).expect("editor");
            editor.update_in(cx, |editor, window, cx| {
                let snapshot = editor.display_snapshot(cx);
                let details = editor.text_layout_details(window, cx);
                for row in 0..2 * rows_per_group {
                    snapshot.x_for_display_point(DisplayPoint::new(DisplayRow(row), 0), &details);
                }
                let [source_x, plain_x] = [source_row, source_row + rows_per_group].map(|row| {
                    snapshot
                        .x_for_display_point(DisplayPoint::new(DisplayRow(row), 69_000), &details)
                });
                assert_eq!(source_x, plain_x);
                assert!(
                    !snapshot
                        .long_row(DisplayRow(source_row))
                        .expect("source row")
                        .is_settled_through(69_000)
                );
                editor.change_selections(SelectionEffects::no_scroll(), window, cx, |selections| {
                    selections.select_ranges([source_row, source_row + rows_per_group].map(
                        |row| {
                            Point::new(row, 69_000 - tail_offset)
                                ..Point::new(row, 69_000 - head_offset)
                        },
                    ));
                });
                for _ in 1..rows_per_group {
                    if above {
                        editor.add_selection_above(
                            &AddSelectionAbove {
                                skip_soft_wrap: false,
                            },
                            window,
                            cx,
                        );
                    } else {
                        editor.add_selection_below(
                            &AddSelectionBelow {
                                skip_soft_wrap: false,
                            },
                            window,
                            cx,
                        );
                    }
                }
                assert_eq!(editor.selections.count(), (2 * rows_per_group) as usize);
            });

            cx.background_executor.set_block_on_ticks(0..=50);
            cx.run_until_parked();
            editor.update(cx, |editor, cx| {
                let snapshot = editor.display_snapshot(cx);
                assert_eq!(
                    editor.selections.ranges::<Point>(&snapshot),
                    (0..2 * rows_per_group)
                        .map(|row| {
                            let column = if row < rows_per_group && row != source_row { 68_250 } else { 69_000 };
                            Point::new(row, column - tail_offset)
                                ..Point::new(row, column - head_offset)
                        })
                        .collect::<Vec<_>>(),
                    "above={above}, tail_offset={tail_offset}, head_offset={head_offset}, repeated={repeated}"
                );
            });
        }
    }
}
