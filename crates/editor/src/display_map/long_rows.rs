use std::{
    cmp::Ordering,
    collections::VecDeque,
    iter::Peekable,
    mem,
    ops::Range,
    sync::{
        Arc,
        atomic::{self, AtomicUsize},
    },
};

use collections::HashMap;
use futures_lite::future::yield_now;
use gpui::{
    FontId, FontRun, Hsla, LineLayout, LineWrapper, LineWrapperHandle, Pixels, ShapedGlyph,
    ShapedRun, SharedString, StrikethroughStyle, TextAlign, TextRun, UnderlineStyle, point, px,
};
use language::LanguageAwareStyling;
use parking_lot::Mutex;
use smallvec::{SmallVec, smallvec};
use sum_tree::{Bias, Cursor, SumTree};
use unicode_segmentation::GraphemeCursor;
use util::debug_panic;

use super::{
    ChunkRendererId, FoldPoint, Highlights,
    invisibles::{is_invisible, replacement},
    tab_map::{TabChunks, TabEdit, TabPoint, TabSnapshot},
};
use crate::{MAX_LINE_LEN, scroll::ScrollPixelOffset};

const MIN_CHUNK_LEN: u32 = if cfg!(test) { 16 } else { 1024 };
const MAX_CHUNK_LEN: u32 = if cfg!(test) { 64 } else { 8 * 1024 };
const FORCED_CHUNK_LEN: u32 = 2 * MAX_CHUNK_LEN;
const PHASE_PROBES: usize = 6;
const WHITESPACE_CUT_MASK: u64 = if cfg!(test) { 0b11 } else { 0x7f };
const DENSE_CUT_MASK: u64 = if cfg!(test) { 0b1111 } else { 0x7ff };
const DENSE_RUN_LEN: u32 = if cfg!(test) { 8 } else { 64 };
const MAX_CUT_SCAN_LEN: u32 = if cfg!(test) { 2 * 1024 } else { 64 * 1024 };
const SHAPING_CONTEXT_LEN: u32 = if cfg!(test) { 8 } else { 64 };
const SYNC_MEASURE_LEN: u32 = if cfg!(test) { 512 } else { 16 * 1024 };
const RESYNC_RESERVE_LEN: u32 = FORCED_CHUNK_LEN + SHAPING_CONTEXT_LEN;
const MEASURE_YIELD_LEN: u32 = if cfg!(test) { 256 } else { 8 * 1024 };
const SCAN_YIELD_STEPS: u32 = 1024;
const SYNC_SCAN_ROWS: usize = 32;
const MAX_WINDOW_LEN: u32 = 64 * 1024;
const MAX_VISIBLE_WINDOW_LEN: u32 = 1024 * 1024;
const MAX_ATOM_LEN: u32 = if cfg!(test) { 32 } else { MAX_LINE_LEN as u32 };
const CAP_SCAN_LEN: u32 = MAX_ATOM_LEN + 16;
const GRAPHEME_CONTEXT_LEN: u32 = 256;
const MAX_GRAPHEME_CONTEXT_LEN: u32 = 16 * 1024;

const MAX_RESYNC_ATTEMPTS: usize = 16;
const MAX_CACHED_LAYOUT_VARIANTS: usize = 4;
const MAX_CACHED_LAYOUT_BYTES: usize = 256 * 1024;
const MAX_CACHED_CHUNK_COLUMNS: usize = 256 * 1024;
const MAX_WINDOW_MEASURE_PASSES: usize = 4;
const ESTIMATE_RELATIVE_TOLERANCE: ScrollPixelOffset = 1e-4;

static NEXT_EPOCH: AtomicUsize = AtomicUsize::new(1);
static GEAR: [u64; 256] = gear_table();

#[derive(Clone)]
pub struct LongRow {
    snapshot: TabSnapshot,
    row: u32,
    geometry: Arc<RowGeometry>,
    stored: Option<Arc<RowGeometry>>,
    widths: Arc<CharWidths>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct RowWindow {
    segments: SmallVec<[Range<u32>; 1]>,
    elided: Option<Range<u32>>,
}

pub(crate) struct MeasureRequest {
    version: usize,
    row: u32,
    columns: SmallVec<[Range<u32>; 1]>,
    runs: SmallVec<[RowRuns; 1]>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct RowRuns {
    start: u32,
    end: u32,
    runs: Arc<[StyledRun]>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct StyledRun {
    run: FontRun,
    decoration: Option<RunDecoration>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct RunDecoration {
    color: Hsla,
    underline: Option<UnderlineStyle>,
    strikethrough: Option<StrikethroughStyle>,
}

#[derive(Clone)]
pub(crate) struct TextPiece {
    pub(crate) columns: Range<u32>,
    pub(crate) x: ScrollPixelOffset,
    pub(crate) text: SharedString,
    pub(crate) layout: Arc<LineLayout>,
    pub(crate) runs: SmallVec<[FontRun; 1]>,
    index: Option<Arc<LayoutIndex>>,
}

pub(crate) struct WindowLayout {
    pub(crate) atom_positions: Vec<(u32, ScrollPixelOffset)>,
    pub(crate) pieces: Vec<TextPiece>,
}

pub(crate) struct LongRowFrame(Option<Arc<CharWidths>>);

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct ChunkRuns {
    lead: u32,
    runs: Arc<[FontRun]>,
}

#[derive(Clone, Copy, Default)]
struct RunsView<'a>(&'a [RowRuns]);

#[derive(Clone, Default)]
struct RowStyles(Arc<[RowRuns]>);

#[derive(Default)]
struct FontRunsBuilder {
    runs: SmallVec<[StyledRun; 1]>,
    at_seam: bool,
}

impl TextPiece {
    pub(crate) fn is_shaped_with(&self, runs: impl IntoIterator<Item = FontRun>) -> bool {
        runs.into_iter()
            .filter(|run| run.len > 0)
            .eq(self.runs.iter().copied())
    }

    fn window(&self, columns: &Range<u32>, origin_x: ScrollPixelOffset) -> Option<Self> {
        let start = columns.start.max(self.columns.start);
        let end = columns.end.min(self.columns.end);
        if start >= end {
            return None;
        }
        let Some(index) = &self.index else {
            return Some(self.clone());
        };
        let start = index.boundary((start - self.columns.start) as usize, Bias::Left);
        let end = index.boundary((end - self.columns.start) as usize, Bias::Right);
        let origin_x = (origin_x - self.x).max(start.2).min(end.2);
        if start.0 == 0 && end.0 == self.layout.len && origin_x == 0. {
            return Some(self.clone());
        }
        let first_run = index
            .fonts
            .partition_point(|offset| *offset <= start.0)
            .saturating_sub(1);
        Some(Self {
            columns: self.columns.start + start.0 as u32..self.columns.start + end.0 as u32,
            x: self.x + origin_x,
            text: SharedString::new(&self.text[start.0..end.0]),
            layout: index.slice(&self.layout, start, end, origin_x),
            runs: slice_font_runs(
                &self.runs[first_run..],
                index.fonts[first_run] as u32,
                start.0 as u32..end.0 as u32,
            ),
            index: None,
        })
    }
}

struct LayoutIndex {
    partitions: Vec<(usize, usize, ScrollPixelOffset)>,
    runs: Vec<usize>,
    fonts: Vec<usize>,
    positions: Vec<f64>,
    width: f64,
}

impl LayoutIndex {
    fn new(layout: &LineLayout, fonts: &[FontRun], positions: Vec<f64>, width: f64) -> Self {
        let mut partitions = vec![(0, 0)];
        partitions.extend(glyph_partitions(layout));
        partitions.push((layout.len, positions.len()));
        partitions.dedup_by_key(|entry| entry.0);
        let partitions = partitions
            .into_iter()
            .map(|(column, offset)| {
                let x = if column == 0 {
                    0.
                } else {
                    positions.get(offset).copied().unwrap_or(width)
                };
                (column, offset, x)
            })
            .collect();
        let mut offset = 0;
        let runs = layout
            .runs
            .iter()
            .map(|run| {
                let start = offset;
                offset += run.glyphs.len();
                start
            })
            .chain([layout.runs.iter().map(|run| run.glyphs.len()).sum()])
            .collect();
        let mut offset = 0;
        let fonts = fonts
            .iter()
            .map(|run| {
                let start = offset;
                offset += run.len;
                start
            })
            .collect();
        Self {
            partitions,
            runs,
            fonts,
            positions,
            width,
        }
    }

    fn boundary(&self, column: usize, bias: Bias) -> (usize, usize, ScrollPixelOffset) {
        let index = match bias {
            Bias::Left => self
                .partitions
                .partition_point(|entry| entry.0 <= column)
                .saturating_sub(1),
            Bias::Right => self
                .partitions
                .partition_point(|entry| entry.0 < column)
                .min(self.partitions.len() - 1),
        };
        self.partitions[index]
    }

    fn slice(
        &self,
        layout: &LineLayout,
        start: (usize, usize, ScrollPixelOffset),
        end: (usize, usize, ScrollPixelOffset),
        origin_x: ScrollPixelOffset,
    ) -> Arc<LineLayout> {
        let first = self
            .runs
            .partition_point(|offset| *offset <= start.1)
            .saturating_sub(1);
        let last = self.runs.partition_point(|offset| *offset < end.1);
        let runs = layout
            .runs
            .iter()
            .enumerate()
            .skip(first)
            .take(last.saturating_sub(first))
            .map(|(index, run)| {
                let run_start = self.runs[index];
                let glyphs = run
                    .glyphs
                    .iter()
                    .zip(&self.positions[run_start..])
                    .skip(start.1.saturating_sub(run_start))
                    .take(end.1 - run_start.max(start.1))
                    .map(|(glyph, x)| ShapedGlyph {
                        index: glyph.index - start.0,
                        position: point(px((x - origin_x) as f32), glyph.position.y),
                        ..glyph.clone()
                    })
                    .collect();
                ShapedRun {
                    font_id: run.font_id,
                    glyphs,
                }
            })
            .collect();
        Arc::new(LineLayout {
            font_size: layout.font_size,
            width: px((end.2 - origin_x) as f32),
            ascent: layout.ascent,
            descent: layout.descent,
            runs,
            len: end.0 - start.0,
        })
    }
}

impl StyledRun {
    pub(crate) fn from_text_runs(
        text_runs: &[TextRun],
        font_runs: impl IntoIterator<Item = FontRun>,
    ) -> Vec<Self> {
        let mut text_runs = text_runs.iter().peekable();
        let mut text_run_start = 0;
        let mut font_run_start = 0;
        font_runs
            .into_iter()
            .map(|run| {
                while let Some(text_run) = text_runs.peek()
                    && text_run_start + text_run.len <= font_run_start
                {
                    text_run_start += text_run.len;
                    text_runs.next();
                }
                let decoration = text_runs.peek().map(|text_run| RunDecoration {
                    color: text_run.color,
                    underline: text_run.underline,
                    strikethrough: text_run.strikethrough,
                });
                font_run_start += run.len;
                Self { run, decoration }
            })
            .collect()
    }

    fn merges_with(&self, other: &Self) -> bool {
        self.run.font_id == other.run.font_id
            && (self.decoration.is_none()
                || other.decoration.is_none()
                || self.decoration == other.decoration)
    }
}

impl From<FontRun> for StyledRun {
    fn from(run: FontRun) -> Self {
        Self {
            run,
            decoration: None,
        }
    }
}

impl FontRunsBuilder {
    fn seam(&mut self) {
        self.at_seam = true;
    }

    fn push(&mut self, run: StyledRun) {
        if run.run.len == 0 {
            return;
        }
        match self.runs.last_mut() {
            Some(last) if self.at_seam && last.merges_with(&run) => {
                last.run.len += run.run.len;
                last.decoration = last.decoration.or(run.decoration);
            }
            _ => self.runs.push(run),
        }
        self.at_seam = false;
    }

    fn push_entry(&mut self, runs: impl IntoIterator<Item = StyledRun>) {
        self.seam();
        for run in runs {
            self.push(run);
        }
        self.seam();
    }

    fn font_runs(&self) -> SmallVec<[FontRun; 1]> {
        self.runs.iter().map(|styled| styled.run).collect()
    }
}

impl RowRuns {
    pub(crate) fn new(
        columns: Range<u32>,
        runs: impl IntoIterator<Item = StyledRun>,
    ) -> Option<Self> {
        let runs = runs
            .into_iter()
            .filter(|styled| styled.run.len > 0)
            .collect::<Arc<[StyledRun]>>();
        let len = runs.iter().map(|styled| styled.run.len).sum::<usize>();
        if len != columns.len() {
            debug_panic!("font runs cover {len} bytes, but were requested for columns {columns:?}");
            return None;
        }
        Some(Self {
            start: columns.start,
            end: columns.end,
            runs,
        })
    }

    fn slice(&self, columns: Range<u32>) -> Option<Self> {
        let start = columns.start.max(self.start);
        let end = columns.end.min(self.end);
        if start >= end {
            return None;
        }
        Some(Self {
            start,
            end,
            runs: slice_styled_runs(&self.runs, self.start, start..end)
                .into_iter()
                .collect(),
        })
    }

    fn shifted(mut self, delta: i64) -> Self {
        self.start = shift_column(self.start, delta);
        self.end = shift_column(self.end, delta);
        self
    }
}

impl RowStyles {
    fn view(&self) -> RunsView<'_> {
        RunsView(&self.0)
    }

    fn chunk_runs(&self, span: Range<u32>, chunk_start: u32, base: FontId) -> Option<ChunkRuns> {
        ChunkRuns::new(
            piece_runs(self.view(), span.clone(), base),
            span,
            chunk_start,
            base,
        )
    }

    fn covers_identically(&self, runs: &RowRuns) -> bool {
        let first = self.0.partition_point(|known| known.end <= runs.start);
        let mut covered = runs.start;
        let mut known_runs = FontRunsBuilder::default();
        for known in &self.0[first..] {
            if known.start > covered || covered >= runs.end {
                break;
            }
            if let Some(sliced) = known.slice(covered..runs.end) {
                known_runs.push_entry(sliced.runs.iter().copied());
                covered = sliced.end;
            }
        }
        let mut requested = FontRunsBuilder::default();
        requested.push_entry(runs.runs.iter().copied());
        covered >= runs.end && known_runs.runs == requested.runs
    }

    fn with(&self, runs: &[RowRuns]) -> Option<(Self, Vec<Range<u32>>)> {
        let changed = runs
            .iter()
            .filter(|runs| !self.covers_identically(runs))
            .collect::<Vec<_>>();
        if changed.is_empty() {
            return None;
        }
        let mut entries = self.0.to_vec();
        for runs in &changed {
            let mut next = Vec::with_capacity(entries.len() + 2);
            for known in entries {
                if known.end <= runs.start || runs.end <= known.start {
                    next.push(known);
                } else {
                    next.extend(known.slice(known.start..runs.start));
                    next.extend(known.slice(runs.end..known.end));
                }
            }
            next.push((*runs).clone());
            entries = next;
        }
        entries.sort_unstable_by_key(|known| known.start);
        let ranges = changed.iter().map(|runs| runs.start..runs.end).collect();
        Some((Self(entries.into()), ranges))
    }

    fn agreement_end(&self, other: &Self, base: FontId) -> u32 {
        if Arc::ptr_eq(&self.0, &other.0) {
            return u32::MAX;
        }
        let first_entry_difference = self
            .0
            .iter()
            .zip(other.0.iter())
            .find(|(known, other_known)| known != other_known)
            .map(|(known, other_known)| known.start.min(other_known.start))
            .or_else(|| {
                let shared = self.0.len().min(other.0.len());
                self.0
                    .get(shared)
                    .or_else(|| other.0.get(shared))
                    .map(|extra| extra.start)
            });
        let Some(first_entry_difference) = first_entry_difference else {
            return u32::MAX;
        };
        let entries_end = |styles: &Self| styles.0.last().map_or(0, |last| last.end);
        let columns = first_entry_difference.saturating_sub(1)
            ..entries_end(self).max(entries_end(other)).saturating_add(1);
        run_differences(
            &piece_runs(self.view(), columns.clone(), base),
            &piece_runs(other.view(), columns.clone(), base),
            columns.start,
        )
        .first()
        .map_or(u32::MAX, |difference| difference.start)
    }

    fn font_differences(
        &self,
        new: &Self,
        changed: &[Range<u32>],
        len: u32,
        base: FontId,
    ) -> Vec<Range<u32>> {
        let mut differences = changed
            .iter()
            .flat_map(|columns| {
                let span = columns.start.saturating_sub(1)..columns.end.saturating_add(1).min(len);
                if span.is_empty() {
                    return Vec::new();
                }
                run_differences(
                    &piece_runs(self.view(), span.clone(), base),
                    &piece_runs(new.view(), span.clone(), base),
                    span.start,
                )
            })
            .collect::<Vec<_>>();
        differences.sort_unstable_by_key(|difference| difference.start);
        let mut merged = Vec::<Range<u32>>::with_capacity(differences.len());
        for difference in differences {
            match merged.last_mut() {
                Some(last) if difference.start <= last.end => {
                    last.end = last.end.max(difference.end)
                }
                _ => merged.push(difference),
            }
        }
        merged
    }

    fn spliced(&self, edits: &[ColumnEdit]) -> Self {
        if self.0.is_empty() || edits.is_empty() {
            return self.clone();
        }
        let mut kept = Vec::with_capacity(edits.len() + 1);
        let mut previous_end = 0;
        let mut delta = 0i64;
        for edit in edits {
            kept.push((previous_end..edit.old.start, delta));
            delta += edit.delta();
            previous_end = edit.old.end;
        }
        kept.push((previous_end..u32::MAX, delta));
        let entries = self
            .0
            .iter()
            .flat_map(|known| {
                kept.iter().filter_map(|(columns, delta)| {
                    Some(known.slice(columns.clone())?.shifted(*delta))
                })
            })
            .collect::<Vec<_>>();
        Self(entries.into())
    }

    fn joined(&self, suffix: &Self, suffix_start: u32) -> Self {
        if suffix.0.is_empty() {
            return self.clone();
        }
        self.0
            .iter()
            .filter_map(|known| known.slice(0..suffix_start))
            .chain(
                suffix
                    .0
                    .iter()
                    .map(|known| known.clone().shifted(i64::from(suffix_start))),
            )
            .collect::<Vec<_>>()
            .into()
    }
}

impl From<Vec<RowRuns>> for RowStyles {
    fn from(entries: Vec<RowRuns>) -> Self {
        Self(entries.into())
    }
}

impl ChunkRuns {
    fn new(
        runs: SmallVec<[FontRun; 1]>,
        span: Range<u32>,
        chunk_start: u32,
        base: FontId,
    ) -> Option<Self> {
        (runs.len() != 1 || runs[0].font_id != base).then(|| Self {
            lead: chunk_start - span.start,
            runs: runs.into_iter().collect(),
        })
    }

    fn row_runs(&self, chunk_start: u32) -> RowRuns {
        let start = chunk_start.saturating_sub(self.lead);
        let len = self.runs.iter().map(|run| run.len as u32).sum::<u32>();
        RowRuns {
            start,
            end: start + len,
            runs: self.runs.iter().copied().map(StyledRun::from).collect(),
        }
    }
}

impl RowWindow {
    pub(crate) fn prefix(end: u32) -> Self {
        Self {
            segments: smallvec![0..end],
            elided: None,
        }
    }

    pub(crate) fn segments(&self) -> &[Range<u32>] {
        &self.segments
    }

    pub(crate) fn elided(&self) -> Option<Range<u32>> {
        self.elided.clone()
    }

    pub(crate) fn end(&self) -> u32 {
        self.segments.last().map_or(0, |segment| segment.end)
    }

    fn with_elided(mut self, elided: Option<Range<u32>>) -> Self {
        let segments_end = self.end();
        self.elided = elided
            .map(|elided| elided.start.max(segments_end)..elided.end)
            .filter(|elided| !elided.is_empty());
        self
    }
}

impl Drop for LongRowFrame {
    fn drop(&mut self) {
        if let Some(widths) = &self.0 {
            widths.chunks.lock().frame.end();
            widths.layouts.lock().frame.end();
        }
    }
}

impl FromIterator<Range<u32>> for RowWindow {
    fn from_iter<I: IntoIterator<Item = Range<u32>>>(segments: I) -> Self {
        let mut merged = SmallVec::<[Range<u32>; 1]>::new();
        for segment in segments {
            match merged.last_mut() {
                Some(previous) if segment.start <= previous.end => {
                    previous.end = previous.end.max(segment.end)
                }
                _ => merged.push(segment),
            }
        }
        Self {
            segments: merged,
            elided: None,
        }
    }
}

struct Piece {
    columns: Range<u32>,
    density: ScrollPixelOffset,
}

impl Piece {
    fn len(&self) -> u32 {
        self.columns.end - self.columns.start
    }
}

impl LongRow {
    pub(crate) fn len(&self) -> u32 {
        self.geometry.len()
    }

    pub(crate) fn width(&self) -> ScrollPixelOffset {
        self.geometry.width()
    }

    pub fn is_measured(&self) -> bool {
        self.geometry.is_measured()
    }

    pub(crate) fn trailing_whitespace_start(&self) -> Option<u32> {
        self.geometry.trailing_whitespace_start()
    }

    pub(crate) fn metrics(&self) -> RowMetrics {
        self.widths.metrics
    }

    pub(crate) fn x_for_column(&self, column: u32) -> ScrollPixelOffset {
        let atom = self.atom_range(column);
        let column = if atom.start == column {
            column
        } else {
            atom.end
        };
        self.x_for_boundary(column)
    }

    pub(crate) fn closest_column_for_x(&self, x: ScrollPixelOffset) -> u32 {
        if x <= 0. {
            return 0;
        }
        if x >= self.width() {
            return self.len();
        }
        let atom = self.atom_at_x(x);
        if x - atom.start_x <= atom.end_x - x {
            atom.start
        } else {
            atom.end
        }
    }

    pub(crate) fn column_at_x(&self, x: ScrollPixelOffset) -> u32 {
        if x <= 0. {
            return 0;
        }
        if x >= self.width() {
            return self.len();
        }
        self.atom_at_x(x).start
    }

    pub(crate) fn measured_x_for_column(&self, column: u32) -> ScrollPixelOffset {
        let mut long_row = self.refined();
        if let Some(measured) = long_row.measured_around(column) {
            measured.publish();
            long_row = measured;
        }
        long_row.x_for_column(column)
    }

    pub(crate) fn is_settled_through(&self, column: u32) -> bool {
        self.refined().geometry.is_settled_through(column)
    }

    pub(crate) fn measured_closest_column_for_x(&self, x: ScrollPixelOffset) -> u32 {
        let mut long_row = self.refined();
        let mut column = long_row.closest_column_for_x(x);
        let mut measured = false;
        for _ in 0..MAX_WINDOW_MEASURE_PASSES {
            if long_row.geometry.chunk_containing(column).chunk.measured {
                break;
            }
            let Some(remeasured) = long_row.measured_around(column) else {
                break;
            };
            long_row = remeasured;
            measured = true;
            column = long_row.closest_column_for_x(x);
        }
        if measured {
            long_row.publish();
        }
        column
    }

    pub(crate) fn visible_window(
        &self,
        scroll_x: ScrollPixelOffset,
        text_align: TextAlign,
        viewport_width: ScrollPixelOffset,
    ) -> RowWindow {
        let visible = self.visible_x(scroll_x, text_align, viewport_width);
        match self.visible_columns(visible) {
            Ok((visible_columns, _)) => RowWindow {
                segments: smallvec![visible_columns],
                elided: None,
            },
            Err(window) => window,
        }
    }

    pub(crate) fn window(
        &self,
        scroll_x: ScrollPixelOffset,
        text_align: TextAlign,
        viewport_width: ScrollPixelOffset,
    ) -> RowWindow {
        let visible = self.visible_x(scroll_x, text_align, viewport_width);
        let (visible_columns, visible_len) = match self.visible_columns(visible.clone()) {
            Ok(visible_columns) => visible_columns,
            Err(window) => return window,
        };
        let margin = visible.end - visible.start;
        let budget = (MAX_WINDOW_LEN - visible_len.min(MAX_WINDOW_LEN)) / 2;
        let start = self
            .column_at_x(visible.start - margin)
            .max(self.floor_column(visible_columns.start.saturating_sub(budget)))
            .min(visible_columns.start);
        let end = self
            .column_end_at_x(visible.end + margin)
            .min(self.floor_column(visible_columns.end.saturating_add(budget)))
            .max(visible_columns.end);
        RowWindow {
            segments: smallvec![start..end],
            elided: None,
        }
    }

    pub(crate) fn chunk_aligned_window(
        &self,
        scroll_x: ScrollPixelOffset,
        text_align: TextAlign,
        viewport_width: ScrollPixelOffset,
    ) -> RowWindow {
        self.chunk_aligned(&self.window(scroll_x, text_align, viewport_width))
    }

    pub(crate) fn window_around(&self, column: u32) -> RowWindow {
        let start = self.floor_column(column.saturating_sub(MAX_WINDOW_LEN / 2));
        let end = column.saturating_add(MAX_WINDOW_LEN / 2);
        let end = if end >= self.len() {
            self.len()
        } else {
            self.floor_column(end)
        };
        RowWindow {
            segments: smallvec![start..end.max(start)],
            elided: None,
        }
    }

    pub(crate) fn window_layout(
        &self,
        columns: Range<u32>,
        origin_x: ScrollPixelOffset,
    ) -> WindowLayout {
        let mut layout = WindowLayout {
            atom_positions: Vec::new(),
            pieces: Vec::new(),
        };
        for position in self.geometry.chunks_from(columns.start) {
            let (chunk_start, chunk_x) = (position.start, position.x);
            if chunk_start >= columns.end {
                break;
            }
            if position.chunk.measured {
                let chunk_layout = self.chunk_layout(&position);
                layout.atom_positions.extend(
                    chunk_layout
                        .atoms()
                        .skip(chunk_layout.atoms_starting_before(columns.start))
                        .take_while(|(atom_start, _)| *atom_start < columns.end)
                        .map(|(atom_start, atom_x)| (atom_start, chunk_x + atom_x)),
                );
                if !columns.is_empty() {
                    layout.pieces.extend(
                        chunk_layout
                            .pieces(columns.clone(), origin_x - chunk_x)
                            .map(|piece| TextPiece {
                                x: chunk_x + piece.x,
                                ..piece
                            }),
                    );
                }
            } else {
                let density = position.density();
                for_each_atom(
                    &self.snapshot,
                    self.row,
                    chunk_start.max(columns.start)..position.end().min(columns.end),
                    |atom| {
                        layout.atom_positions.push((
                            atom.start,
                            chunk_x + f64::from(atom.start.saturating_sub(chunk_start)) * density,
                        ));
                        true
                    },
                );
            }
        }
        layout
            .atom_positions
            .push((columns.end, self.x_for_boundary(columns.end)));
        layout
    }

    pub(crate) fn chunk_aligned(&self, window: &RowWindow) -> RowWindow {
        let geometry = &self.geometry;
        let mut aligned = window
            .segments()
            .iter()
            .map(|segment| {
                let mut aligned = segment.clone();
                if !segment.is_empty() {
                    let first = geometry.chunk_containing(segment.start);
                    if first.chunk.measured {
                        aligned.start = first.start
                            + first.chunk.layout.as_ref().map_or(0, |layout| {
                                layout.boundary(segment.start - first.start, Bias::Left)
                            });
                    }
                    let last = geometry.chunk_containing(segment.end - 1);
                    if last.chunk.measured {
                        aligned.end = last.start
                            + last.chunk.layout.as_ref().map_or(last.chunk.len, |layout| {
                                layout.boundary(segment.end - last.start, Bias::Right)
                            });
                    }
                }
                aligned
            })
            .collect::<RowWindow>();
        let mut remaining = MAX_VISIBLE_WINDOW_LEN;
        for index in 0..aligned.segments.len() {
            let segment = &aligned.segments[index];
            if segment.end - segment.start <= remaining {
                remaining -= segment.end - segment.start;
                continue;
            }
            let target = segment.start + remaining;
            let chunk = geometry.chunk_containing(target);
            let end = if chunk.chunk.measured {
                chunk.start
                    + chunk.chunk.layout.as_ref().map_or(0, |layout| {
                        layout.boundary(target - chunk.start, Bias::Left)
                    })
            } else {
                self.floor_column(target)
            }
            .max(segment.start);
            let elided_end = window
                .elided()
                .map_or(aligned.end(), |elided| elided.end.max(aligned.end()));
            let kept = index + usize::from(end > segment.start);
            aligned.segments[index].end = end;
            aligned.segments.truncate(kept);
            return aligned.with_elided(Some(end..elided_end));
        }
        aligned.with_elided(window.elided())
    }

    pub(crate) fn can_shape(&self, columns: Range<u32>) -> bool {
        columns.end - columns.start
            <= SYNC_MEASURE_LEN + RESYNC_RESERVE_LEN - 2 * (SHAPING_CONTEXT_LEN + 4)
            && [columns.start, columns.end].into_iter().all(|column| {
                grapheme_floor(&self.snapshot, self.row, self.len(), column, 0)
                    == GraphemeFloor::Boundary(column)
            })
    }

    pub(crate) fn context_bounds(&self, columns: Range<u32>) -> Range<u32> {
        let (snapshot, row) = (&self.snapshot, self.row);
        let start = shaping_span(snapshot, row, columns.start..columns.start).start;
        let end = shaping_span(snapshot, row, columns.end..columns.end).end;
        let start = if start < columns.start
            && shapes_across(snapshot, row, columns.start)
            && !starts_hard_atom(snapshot, row, columns.start)
        {
            let (context, _) = left_context_from(snapshot, row, start, start == 0, columns.start);
            columns.start - context.len() as u32
        } else {
            columns.start
        };
        let end = if columns.end < end && shapes_across(snapshot, row, columns.end) {
            columns.end
                + SeamContext::right_of(snapshot, row, columns.end, end)
                    .text
                    .len() as u32
        } else {
            columns.end
        };
        start..end
    }

    pub(crate) fn hard_breaks(&self, columns: Range<u32>) -> SmallVec<[u32; 2]> {
        let mut breaks = SmallVec::new();
        if columns.is_empty() {
            return breaks;
        }
        let geometry = &self.geometry;
        let position = geometry.chunk_containing(columns.start);
        let chunk_start = position.start;
        let scan_start = match grapheme_floor(
            &self.snapshot,
            self.row,
            geometry.len(),
            columns.start,
            chunk_start,
        ) {
            GraphemeFloor::Boundary(start) => start,
            GraphemeFloor::OversizedCluster if position.chunk.measured => chunk_start,
            GraphemeFloor::OversizedCluster => columns.start,
        };
        for_each_atom(&self.snapshot, self.row, scan_start..columns.end, |atom| {
            if atom.start < scan_start {
                return true;
            }
            let hard = atom.standalone.is_some() || matches!(atom.kind, AtomKind::Renderer { .. });
            if hard {
                for edge in [atom.start, atom.start + atom.len] {
                    if columns.start < edge && edge < columns.end && breaks.last() != Some(&edge) {
                        breaks.push(edge);
                    }
                }
            }
            true
        });
        breaks
    }

    pub(crate) fn shaping_window(&self, window: &RowWindow) -> RowWindow {
        window
            .segments()
            .iter()
            .map(|segment| shaping_span(&self.snapshot, self.row, segment.clone()))
            .collect()
    }

    pub(crate) fn measure_request(
        &self,
        window: &RowWindow,
        runs: &[RowRuns],
    ) -> Option<MeasureRequest> {
        let needs_measuring = window
            .segments()
            .iter()
            .any(|segment| self.geometry.has_unmeasured(segment));
        (needs_measuring || self.needs_restyling(runs)).then(|| MeasureRequest {
            version: self.snapshot.version,
            row: self.row,
            columns: window.segments.clone(),
            runs: runs.iter().cloned().collect(),
        })
    }

    pub(crate) fn restyled(&self, runs: &[RowRuns]) -> Option<LongRow> {
        if !self.needs_restyling(runs) {
            return None;
        }
        let mut wrapper = self.widths.wrapper.lock();
        let geometry = self.geometry.measure_columns(
            &self.snapshot,
            self.row,
            &[],
            runs,
            &mut wrapper,
            &self.widths,
            Some(&self.widths.layouts),
        )?;
        Some(LongRow {
            geometry: Arc::new(geometry),
            ..self.clone()
        })
    }

    pub(crate) fn floor_column(&self, column: u32) -> u32 {
        self.atom_range(column).start
    }

    pub(crate) fn atom_columns(&self, column: u32) -> Range<u32> {
        self.atom_range(column)
    }

    fn visible_x(
        &self,
        scroll_x: ScrollPixelOffset,
        text_align: TextAlign,
        viewport_width: ScrollPixelOffset,
    ) -> Range<ScrollPixelOffset> {
        let viewport_width = viewport_width.max(0.);
        let free_width = viewport_width - self.width();
        let alignment_offset = match text_align {
            TextAlign::Left => 0.,
            TextAlign::Center => free_width / 2.,
            TextAlign::Right => free_width,
        };
        let start = scroll_x - alignment_offset;
        start..start + viewport_width
    }

    fn atom_range(&self, column: u32) -> Range<u32> {
        let geometry = &self.geometry;
        let len = geometry.len();
        if column >= len {
            return len..len;
        }
        let chunk = geometry.chunk_containing(column);
        let chunk_start = chunk.start;
        if chunk.chunk.measured {
            let layout = self.chunk_layout(&chunk);
            let position = layout.atoms_starting_before(column.saturating_add(1));
            let start = position
                .checked_sub(1)
                .and_then(|previous| layout.atom(previous))
                .map_or(chunk_start, |(atom_start, _)| atom_start);
            let end = layout
                .atom(position)
                .map_or(chunk.end(), |(atom_start, _)| atom_start);
            return start..end;
        }
        let boundary = match grapheme_floor(&self.snapshot, self.row, len, column, chunk_start) {
            GraphemeFloor::Boundary(boundary) => boundary,
            GraphemeFloor::OversizedCluster => column,
        };
        let scan_start = clip_column(&self.snapshot, self.row, boundary, Bias::Left)
            .map_or(chunk_start, |clipped| clipped.clamp(chunk_start, boundary));
        let scan = scan_start..chunk.end().min(column.saturating_add(MAX_ATOM_LEN));
        let mut atom = scan.clone();
        for_each_atom(&self.snapshot, self.row, scan, |raw| {
            atom = raw.start..raw.start + raw.len;
            atom.end <= column
        });
        atom
    }

    fn visible_columns(
        &self,
        visible: Range<ScrollPixelOffset>,
    ) -> Result<(Range<u32>, u32), RowWindow> {
        let pieces = self.pieces(visible.clone());
        let visible_len = pieces.iter().map(Piece::len).sum::<u32>();
        if visible_len > MAX_VISIBLE_WINDOW_LEN {
            return Err(budgeted_window(&pieces, MAX_VISIBLE_WINDOW_LEN));
        }
        let visible_columns = match (pieces.first(), pieces.last()) {
            (Some(first), Some(last)) => first.columns.start..last.columns.end,
            _ => {
                let column = self.column_at_x(visible.start);
                column..column
            }
        };
        Ok((visible_columns, visible_len))
    }

    fn needs_restyling(&self, runs: &[RowRuns]) -> bool {
        runs.iter()
            .any(|runs| !self.geometry.styles.covers_identically(runs))
    }

    fn refined(&self) -> LongRow {
        let refinements = self.widths.refinements.lock();
        match refinements.get(&self.row) {
            Some(refinement) if refinement.refines(self) => LongRow {
                geometry: refinement.geometry.clone(),
                ..self.clone()
            },
            _ => self.clone(),
        }
    }

    fn publish(&self) {
        if self
            .stored
            .as_ref()
            .is_some_and(|stored| Arc::ptr_eq(stored, &self.geometry))
        {
            return;
        }
        self.widths.refinements.lock().insert(
            self.row,
            Refinement {
                version: self.snapshot.version,
                stored: self.stored.clone(),
                geometry: self.geometry.clone(),
            },
        );
    }

    fn measured_around(&self, column: u32) -> Option<LongRow> {
        let window = self.window_around(column);
        if !window
            .segments()
            .iter()
            .any(|segment| self.geometry.has_unmeasured(segment))
        {
            return None;
        }
        let mut wrapper = self.widths.wrapper.lock();
        let geometry = self.geometry.measure_columns(
            &self.snapshot,
            self.row,
            window.segments(),
            &[],
            &mut wrapper,
            &self.widths,
            Some(&self.widths.layouts),
        )?;
        Some(LongRow {
            geometry: Arc::new(geometry),
            ..self.clone()
        })
    }

    fn chunk_layout(&self, position: &ChunkPosition<'_>) -> PlacedChunk {
        let columns = position.columns();
        let start = columns.start;
        let chunk = position.chunk;
        if let Some(layout) = &chunk.layout {
            return PlacedChunk {
                start,
                layout: layout.clone(),
            };
        }
        if let Some(layout) = self.widths.chunks.lock().get(chunk.layout_id) {
            return PlacedChunk { start, layout };
        }
        let row_runs = chunk.runs.as_ref().map(|runs| runs.row_runs(start));
        let layout = shape_chunk(
            &self.snapshot,
            self.row,
            ChunkRequest {
                runs: RunsView(row_runs.as_slice()),
                columns: columns.clone(),
                cut: false,
                budget: None,
            },
            &mut self.widths.wrapper.lock(),
            Some(&self.widths.layouts),
        );
        let layout = Arc::new(layout.relative(start));
        self.widths
            .chunks
            .lock()
            .insert(chunk.layout_id, columns.end - start, layout.clone());
        PlacedChunk { start, layout }
    }

    fn x_for_boundary(&self, column: u32) -> ScrollPixelOffset {
        let geometry = &self.geometry;
        if column >= geometry.len() {
            return geometry.width();
        }
        let chunk = geometry.chunk_containing(column);
        let (chunk_start, chunk_x) = (chunk.start, chunk.x);
        if column <= chunk_start {
            return chunk_x;
        }
        if !chunk.chunk.measured {
            return chunk_x + f64::from(column - chunk_start) * chunk.density();
        }
        let layout = self.chunk_layout(&chunk);
        layout
            .atom(layout.atoms_starting_before(column))
            .map_or(chunk.x_end(), |(_, atom_x)| {
                (chunk_x + atom_x).min(chunk.x_end())
            })
    }

    fn pieces(&self, x: Range<ScrollPixelOffset>) -> Vec<Piece> {
        let start_x = x.start.max(0.);
        let end_x = x.end.min(self.width());
        let mut pieces = Vec::new();
        if start_x >= end_x {
            return pieces;
        }
        for chunk in self.geometry.chunks_from_x(start_x) {
            if chunk.x >= end_x {
                break;
            }
            let start = if chunk.x < start_x {
                self.atom_at_x(start_x).start
            } else {
                chunk.start
            };
            let end = if chunk.x_end() > end_x {
                self.atom_at_x(end_x).end
            } else {
                chunk.end()
            };
            if start < end {
                pieces.push(Piece {
                    columns: start..end,
                    density: chunk.density(),
                });
            }
        }
        pieces
    }

    fn column_end_at_x(&self, x: ScrollPixelOffset) -> u32 {
        if x >= self.width() {
            self.len()
        } else {
            self.atom_at_x(x.max(0.)).end
        }
    }

    fn atom_at_x(&self, x: ScrollPixelOffset) -> Atom {
        let chunk = self.geometry.chunk_at_x(x);
        let (chunk_start, chunk_x) = (chunk.start, chunk.x);
        if !chunk.chunk.measured {
            let density = chunk.density();
            let offset = if density > 0. {
                ((x - chunk_x) / density).floor().max(0.) as u32
            } else {
                0
            };
            let target = chunk_start
                .saturating_add(offset)
                .min(chunk.end().saturating_sub(1));
            let Range { start, end } = self.atom_range(target);
            return Atom {
                start,
                end,
                start_x: chunk_x + f64::from(start - chunk_start) * density,
                end_x: chunk_x + f64::from(end - chunk_start) * density,
            };
        }
        let layout = self.chunk_layout(&chunk);
        let position = layout
            .layout
            .atoms
            .partition_point(|(_, atom_x)| chunk_x + atom_x <= x);
        let Some((start, start_x)) = position
            .checked_sub(1)
            .and_then(|atom_index| layout.atom(atom_index))
        else {
            return Atom {
                start: chunk.end(),
                end: chunk.end(),
                start_x: chunk.x_end(),
                end_x: chunk.x_end(),
            };
        };
        let (end, end_x) = layout
            .atom(position)
            .map_or((chunk.end(), chunk.x_end()), |(end, end_x)| {
                (end, chunk_x + end_x)
            });
        Atom {
            start,
            end,
            start_x: chunk_x + start_x,
            end_x,
        }
    }
}

#[derive(Clone, Default)]
pub(crate) struct LongRows {
    widths: Option<Arc<CharWidths>>,
    rows: SumTree<RowEntry>,
    pending: Arc<Vec<PendingRows>>,
    epoch: usize,
}

#[derive(Clone, Debug)]
struct PendingRows {
    rows: Range<u32>,
    text_id: usize,
}

impl LongRows {
    pub(crate) fn new(mut wrapper: LineWrapperHandle) -> Self {
        let estimate = f64::from(wrapper.width_for_char('0')).max(1.);
        let em_width = f64::from(wrapper.width_for_char('m'));
        let base_font = wrapper.font_id();
        let base_layout = wrapper.layout_line(
            " ",
            &[FontRun {
                len: 1,
                font_id: base_font,
            }],
        );
        Self {
            widths: Some(Arc::new(CharWidths {
                wrapper: Mutex::new(wrapper),
                estimate,
                em_width,
                base_font,
                metrics: RowMetrics {
                    ascent: base_layout.ascent,
                    descent: base_layout.descent,
                },
                layouts: Mutex::default(),
                chunks: Mutex::default(),
                refinements: Mutex::default(),
            })),
            rows: SumTree::default(),
            pending: Arc::new(vec![PendingRows {
                rows: 0..u32::MAX,
                text_id: next_epoch(),
            }]),
            epoch: next_epoch(),
        }
    }

    pub(crate) fn epoch(&self) -> usize {
        self.epoch
    }

    pub(crate) fn begin_frame(&self) -> LongRowFrame {
        if let Some(widths) = &self.widths {
            widths.chunks.lock().frame.begin();
            widths.layouts.lock().frame.begin();
        }
        LongRowFrame(self.widths.clone())
    }

    pub(crate) fn max_width_in(&self, rows: Range<u32>) -> ScrollPixelOffset {
        if rows.is_empty() {
            return 0.;
        }
        let mut cursor = self.rows.cursor::<RowCount>(());
        cursor.seek(&RowCount(rows.start), Bias::Right);
        cursor
            .summary::<_, RowEntrySummary>(&RowCount(rows.end), Bias::Right)
            .max_width
    }

    pub(crate) fn absorb(&self, snapshot: &TabSnapshot, measured: LongRows) -> LongRows {
        let (Some(widths), Some(measured_widths)) = (&self.widths, &measured.widths) else {
            return self.clone();
        };
        if !Arc::ptr_eq(widths, measured_widths) {
            return self.clone();
        }
        let scanned = subtract_pending(&self.pending, &measured.pending);
        if scanned.is_empty() {
            return self.clone();
        }
        let mut rows = SumTree::default();
        let mut changed = false;
        let mut current = self.rows.cursor::<RowCount>(());
        current.seek(&RowCount(0), Bias::Right);
        let mut fresh = measured.rows.cursor::<RowCount>(());
        fresh.seek(&RowCount(0), Bias::Right);
        for scanned in scanned {
            copy_rows_before(&mut current, &mut rows, scanned.start, 0);
            fresh.seek_forward(&RowCount(scanned.start), Bias::Right);
            loop {
                let current_row = entry_row(&current).filter(|row| *row < scanned.end);
                let limit = current_row.unwrap_or(scanned.end);
                changed |= copy_rows_before(&mut fresh, &mut rows, limit, 0);
                let (Some(row), Some(entry)) = (current_row, current.item()) else {
                    break;
                };
                let current_geometry = entry.geometry.clone();
                current.next();
                let merged = take_row(&mut fresh, row)
                    .filter(|measured_geometry| {
                        !current_geometry.is_settled()
                            && current_geometry.len() == measured_geometry.len()
                    })
                    .and_then(|measured_geometry| {
                        current_geometry.merged(&measured_geometry, snapshot, row, widths)
                    });
                let geometry = match merged {
                    Some(merged) => {
                        changed = true;
                        Arc::new(merged)
                    }
                    None => current_geometry,
                };
                push_row(&mut rows, row, geometry);
            }
        }
        let rows = if changed {
            copy_rows_before(&mut current, &mut rows, u32::MAX, 0);
            rows
        } else {
            self.rows.clone()
        };
        let mut pending = intersect_pending(&measured.pending, &self.pending);
        pending.extend(uncovered_unsettled_rows(&rows, &pending));
        let epoch = if changed { next_epoch() } else { self.epoch };
        Self::with_rows(widths.clone(), rows, pending, epoch)
    }

    pub(crate) fn em_width(&self) -> Option<ScrollPixelOffset> {
        Some(self.widths.as_ref()?.em_width)
    }

    pub(crate) fn row(&self, snapshot: &TabSnapshot, row: u32) -> Option<LongRow> {
        let widths = self.widths.as_ref()?;
        let len = snapshot.line_len(row);
        if len as usize <= MAX_LINE_LEN {
            return None;
        }
        let stored = self
            .geometry_at(row)
            .filter(|geometry| geometry.len() == len);
        let geometry = stored.clone().unwrap_or_else(|| {
            RowGeometry::unmeasured(
                snapshot,
                row,
                len,
                widths.estimate,
                self.pending_text_id(row),
            )
        });
        Some(LongRow {
            snapshot: snapshot.clone(),
            row,
            geometry,
            stored,
            widths: widths.clone(),
        })
    }

    pub(crate) fn adopt_refinements(&self, snapshot: &TabSnapshot) -> Option<Self> {
        let widths = self.widths.as_ref()?;
        let mut refinements = mem::take(&mut *widths.refinements.lock())
            .into_iter()
            .filter(|(_, refinement)| refinement.version == snapshot.version)
            .collect::<Vec<_>>();
        if refinements.is_empty() {
            return None;
        }
        refinements.sort_unstable_by_key(|(row, _)| *row);
        let mut rows = SumTree::default();
        let mut old_rows = self.rows.cursor::<RowCount>(());
        old_rows.seek(&RowCount(0), Bias::Right);
        let mut changed = false;
        for (row, refinement) in refinements {
            copy_rows_before(&mut old_rows, &mut rows, row, 0);
            let existing = take_row(&mut old_rows, row);
            let len = snapshot.line_len(row);
            let stored = existing.as_ref().filter(|existing| existing.len() == len);
            let adopted = refinement.geometry.len() == len
                && match (stored, &refinement.stored) {
                    (Some(stored), Some(source)) => Arc::ptr_eq(stored, source),
                    (None, None) => self.pending_text_id(row) == refinement.geometry.text_id,
                    _ => false,
                };
            if adopted {
                changed = true;
                push_row(&mut rows, row, refinement.geometry);
            } else if let Some(existing) = existing {
                push_row(&mut rows, row, existing);
            }
        }
        copy_rows_before(&mut old_rows, &mut rows, u32::MAX, 0);
        changed.then(|| Self {
            widths: Some(widths.clone()),
            rows,
            pending: self.pending.clone(),
            epoch: next_epoch(),
        })
    }

    pub(crate) fn clear_pending(&mut self) {
        self.pending = Arc::default();
    }

    fn pending_text_id(&self, row: u32) -> usize {
        pending_at(&self.pending, row).map_or(0, |pending| pending.text_id)
    }

    pub(crate) fn has_pending(&self) -> bool {
        self.widths.is_some() && !self.pending.is_empty()
    }

    pub(crate) fn interpolate(&self, snapshot: &TabSnapshot, edits: &[TabEdit]) -> Self {
        let mut budget = SYNC_MEASURE_LEN + RESYNC_RESERVE_LEN;
        self.interpolate_within(snapshot, edits, &mut budget)
    }

    pub(crate) fn measure(
        &self,
        snapshot: &TabSnapshot,
        requests: &[MeasureRequest],
    ) -> Option<Self> {
        let widths = self.widths.as_ref()?;
        let mut requests = requests
            .iter()
            .filter(|request| request.version == snapshot.version)
            .collect::<Vec<_>>();
        requests.sort_unstable_by_key(|request| request.row);
        requests.dedup_by_key(|request| request.row);
        let mut wrapper = widths.wrapper.lock();
        let mut rows = SumTree::default();
        let mut old_rows = self.rows.cursor::<RowCount>(());
        old_rows.seek(&RowCount(0), Bias::Right);
        let mut changed = false;
        for request in requests {
            let len = snapshot.line_len(request.row);
            if len as usize <= MAX_LINE_LEN {
                continue;
            }
            copy_rows_before(&mut old_rows, &mut rows, request.row, 0);
            let geometry = take_row(&mut old_rows, request.row)
                .filter(|geometry| geometry.len() == len)
                .unwrap_or_else(|| {
                    RowGeometry::unmeasured(
                        snapshot,
                        request.row,
                        len,
                        widths.estimate,
                        self.pending_text_id(request.row),
                    )
                });
            let geometry = match geometry.measure_columns(
                snapshot,
                request.row,
                &request.columns,
                &request.runs,
                &mut wrapper,
                widths,
                None,
            ) {
                Some(measured) => {
                    changed = true;
                    Arc::new(measured)
                }
                None => geometry,
            };
            push_row(&mut rows, request.row, geometry);
        }
        copy_rows_before(&mut old_rows, &mut rows, u32::MAX, 0);
        changed.then(|| {
            let mut pending = self.pending.to_vec();
            pending.extend(uncovered_unsettled_rows(&rows, &self.pending));
            Self::with_rows(widths.clone(), rows, pending, next_epoch())
        })
    }

    pub(crate) async fn measure_pending(
        self,
        snapshot: TabSnapshot,
        mut wrapper: LineWrapperHandle,
    ) -> Self {
        let Some(widths) = self.widths.clone() else {
            return self;
        };
        let row_count = snapshot.max_point().row() + 1;
        let mut rows = SumTree::default();
        let mut old_rows = self.rows.cursor::<RowCount>(());
        old_rows.seek(&RowCount(0), Bias::Right);
        let mut steps = 0;
        for pending in self.pending.iter() {
            let range = pending.rows.start.min(row_count)..pending.rows.end.min(row_count);
            copy_rows_before(&mut old_rows, &mut rows, range.start, 0);
            let mut scan = LongRowScan::new(range.clone());
            loop {
                let step = scan.step(&snapshot);
                steps += 1;
                if steps >= SCAN_YIELD_STEPS {
                    steps = 0;
                    yield_now().await;
                }
                let (row, len) = match step {
                    ScanStep::Done => break,
                    ScanStep::Continue => continue,
                    ScanStep::Row { row, len } => (row, len),
                };
                old_rows.seek_forward(&RowCount(row), Bias::Right);
                let geometry = take_row(&mut old_rows, row)
                    .filter(|old| old.len() == len)
                    .unwrap_or_else(|| {
                        RowGeometry::unmeasured(
                            &snapshot,
                            row,
                            len,
                            widths.estimate,
                            pending.text_id,
                        )
                    });
                let geometry = if geometry.is_settled() {
                    geometry
                } else {
                    Arc::new(
                        geometry
                            .measure_remaining(&snapshot, row, &mut wrapper)
                            .await,
                    )
                };
                push_row(&mut rows, row, geometry);
            }
            old_rows.seek_forward(&RowCount(range.end), Bias::Right);
        }
        copy_rows_before(&mut old_rows, &mut rows, u32::MAX, 0);
        Self::with_rows(widths, rows, Vec::new(), next_epoch())
    }

    fn interpolate_within(
        &self,
        snapshot: &TabSnapshot,
        edits: &[TabEdit],
        budget: &mut u32,
    ) -> Self {
        let Some(widths) = &self.widths else {
            return self.clone();
        };
        if edits.is_empty() {
            return self.clone();
        }
        let mut wrapper = widths.wrapper.lock();
        let mut measurer = Measurer {
            snapshot,
            wrapper: &mut wrapper,
            estimate: widths.estimate,
            budget: *budget,
            styles: RowStyles::default(),
            cache: None,
        };
        let mut rows = SumTree::default();
        let mut pending = Vec::new();
        let mut old_rows = self.rows.cursor::<RowCount>(());
        old_rows.seek(&RowCount(0), Bias::Right);
        let mut old_pending = self.pending.iter().rev().cloned().collect::<Vec<_>>();
        let mut row_delta = 0i64;
        let mut edits = edits.iter().peekable();
        while let Some(group) = RowEditGroup::next(&mut edits) {
            copy_rows_before(&mut old_rows, &mut rows, group.old_rows.start, row_delta);
            while let Some(old) = old_pending.pop_if(|old| old.rows.start < group.old_rows.end) {
                if old.rows.start < group.old_rows.start {
                    pending.push(PendingRows {
                        rows: shift_rows(
                            old.rows.start..old.rows.end.min(group.old_rows.start),
                            row_delta,
                        ),
                        text_id: old.text_id,
                    });
                }
                if old.rows.end > group.old_rows.end {
                    old_pending.push(PendingRows {
                        rows: group.old_rows.end..old.rows.end,
                        text_id: old.text_id,
                    });
                }
            }
            old_rows.seek_forward(&RowCount(group.old_rows.end), Bias::Right);
            if group.new_rows.len() > SYNC_SCAN_ROWS {
                pending.push(PendingRows {
                    rows: group.new_rows.clone(),
                    text_id: next_epoch(),
                });
            } else {
                for new_row in group.new_rows.clone() {
                    let len = snapshot.line_len(new_row);
                    if len as usize <= MAX_LINE_LEN {
                        continue;
                    }
                    let text_id = next_epoch();
                    let geometry = self
                        .reuse_geometry(&group, new_row, len, text_id, &mut measurer)
                        .unwrap_or_else(|| measurer.measure_prefix(new_row, len, text_id));
                    if !geometry.is_settled() {
                        pending.push(PendingRows {
                            rows: new_row..new_row + 1,
                            text_id,
                        });
                    }
                    push_row(&mut rows, new_row, Arc::new(geometry));
                }
            }
            row_delta +=
                i64::from(group.new_rows.len() as u32) - i64::from(group.old_rows.len() as u32);
        }
        copy_rows_before(&mut old_rows, &mut rows, u32::MAX, row_delta);
        pending.extend(old_pending.into_iter().rev().map(|old| PendingRows {
            rows: shift_rows(old.rows, row_delta),
            text_id: old.text_id,
        }));
        *budget = measurer.budget;
        Self::with_rows(widths.clone(), rows, pending, next_epoch())
    }

    fn reuse_geometry(
        &self,
        group: &RowEditGroup,
        new_row: u32,
        new_len: u32,
        text_id: usize,
        measurer: &mut Measurer,
    ) -> Option<RowGeometry> {
        if group.is_single_row() {
            let old = self.geometry_at(group.old_rows.start)?;
            return measurer.splice(&old, &group.edits, new_row, new_len, text_id);
        }
        let first = group.edits.first()?;
        let last = group.edits.last()?;
        let prefix = (new_row == group.new_rows.start
            && first.old.start.row() == group.old_rows.start
            && first.new.start.row() == new_row)
            .then(|| self.geometry_at(group.old_rows.start))
            .flatten();
        let suffix = (new_row + 1 == group.new_rows.end
            && last.old.end.row() + 1 == group.old_rows.end
            && last.new.end.row() == new_row)
            .then(|| self.geometry_at(last.old.end.row()))
            .flatten();
        if prefix.is_none() && suffix.is_none() {
            return None;
        }
        let prefix_len = prefix.as_ref().map_or(0, |prefix| prefix.len());
        let mut chunks = prefix
            .as_ref()
            .map_or_else(SumTree::default, |prefix| prefix.chunks.clone());
        if let Some(suffix) = &suffix {
            chunks.append(suffix.chunks.clone(), ());
        }
        let canonical_until = prefix
            .as_ref()
            .filter(|prefix| prefix.canonical_until < prefix.len())
            .map(|prefix| prefix.canonical_until)
            .or_else(|| {
                suffix
                    .as_ref()
                    .map(|suffix| prefix_len.saturating_add(suffix.canonical_until))
            })
            .unwrap_or(u32::MAX);
        let edit_start = if prefix.is_some() {
            first.new.start.column()
        } else {
            0
        };
        let (old_end, new_end) = match &suffix {
            Some(_) => (prefix_len + last.old.end.column(), last.new.end.column()),
            None => (prefix_len, new_len),
        };
        let edit = TabEdit {
            old: TabPoint::new(new_row, edit_start)..TabPoint::new(new_row, old_end),
            new: TabPoint::new(new_row, edit_start)..TabPoint::new(new_row, new_end),
        };
        let prefix_styles = prefix
            .as_ref()
            .map_or_else(RowStyles::default, |prefix| prefix.styles.clone());
        let styles = match &suffix {
            Some(suffix) => prefix_styles.joined(&suffix.styles, prefix_len),
            None => prefix_styles,
        };
        measurer.splice(
            &RowGeometry::new(chunks, canonical_until, text_id, styles, measurer.estimate),
            &[&edit],
            new_row,
            new_len,
            text_id,
        )
    }

    fn geometry_at(&self, row: u32) -> Option<Arc<RowGeometry>> {
        let mut cursor = self.rows.cursor::<RowCount>(());
        cursor.seek(&RowCount(row), Bias::Right);
        take_row(&mut cursor, row)
    }

    fn with_rows(
        widths: Arc<CharWidths>,
        rows: SumTree<RowEntry>,
        mut pending: Vec<PendingRows>,
        epoch: usize,
    ) -> Self {
        pending.retain(|pending| !pending.rows.is_empty());
        pending.sort_unstable_by_key(|pending| pending.rows.start);
        let mut merged = Vec::<PendingRows>::with_capacity(pending.len());
        for mut next in pending {
            if let Some(last) = merged.last_mut() {
                next.rows.start = next.rows.start.max(last.rows.end);
                if next.rows.is_empty() {
                    continue;
                }
                if next.rows.start == last.rows.end && next.text_id == last.text_id {
                    last.rows.end = next.rows.end;
                    continue;
                }
            }
            merged.push(next);
        }
        Self {
            widths: Some(widths),
            rows,
            pending: Arc::new(merged),
            epoch,
        }
    }
}

fn next_epoch() -> usize {
    NEXT_EPOCH.fetch_add(1, atomic::Ordering::Relaxed)
}

fn entry_row(cursor: &Cursor<'_, '_, RowEntry, RowCount>) -> Option<u32> {
    cursor.item().map(|entry| cursor.start().0 + entry.gap)
}

fn take_row(cursor: &mut Cursor<'_, '_, RowEntry, RowCount>, row: u32) -> Option<Arc<RowGeometry>> {
    let geometry = cursor
        .item()
        .filter(|_| entry_row(cursor) == Some(row))
        .map(|entry| entry.geometry.clone());
    if geometry.is_some() {
        cursor.next();
    }
    geometry
}

fn pending_at(pending: &[PendingRows], row: u32) -> Option<&PendingRows> {
    let index = pending.partition_point(|pending| pending.rows.end <= row);
    pending
        .get(index)
        .filter(|pending| pending.rows.contains(&row))
}

fn uncovered_unsettled_rows(rows: &SumTree<RowEntry>, pending: &[PendingRows]) -> Vec<PendingRows> {
    let mut uncovered = Vec::new();
    let mut cursor = rows.filter::<_, RowCount>((), |summary: &RowEntrySummary| summary.unsettled);
    cursor.next();
    while let Some(entry) = cursor.item() {
        let row = cursor.start().0 + entry.gap;
        if pending_at(pending, row).is_none() {
            uncovered.push(PendingRows {
                rows: row..row + 1,
                text_id: entry.geometry.text_id,
            });
        }
        cursor.next();
    }
    uncovered
}

fn subtract_pending(current: &[PendingRows], measured: &[PendingRows]) -> Vec<Range<u32>> {
    let mut remaining = Vec::new();
    let mut measured = measured.iter().peekable();
    for current in current {
        let mut start = current.rows.start;
        while start < current.rows.end {
            while measured
                .next_if(|measured| measured.rows.end <= start)
                .is_some()
            {}
            match measured.peek() {
                Some(measured) if measured.rows.start < current.rows.end => {
                    if start < measured.rows.start {
                        remaining.push(start..measured.rows.start);
                    }
                    start = start.max(measured.rows.end);
                }
                _ => {
                    remaining.push(start..current.rows.end);
                    break;
                }
            }
        }
    }
    remaining
}

fn intersect_pending(measured: &[PendingRows], current: &[PendingRows]) -> Vec<PendingRows> {
    let mut pending = Vec::new();
    let (mut measured_index, mut current_index) = (0, 0);
    while let (Some(measured), Some(current)) =
        (measured.get(measured_index), current.get(current_index))
    {
        let start = measured.rows.start.max(current.rows.start);
        let end = measured.rows.end.min(current.rows.end);
        if start < end {
            pending.push(PendingRows {
                rows: start..end,
                text_id: current.text_id,
            });
        }
        if measured.rows.end < current.rows.end {
            measured_index += 1;
        } else {
            current_index += 1;
        }
    }
    pending
}

#[derive(Clone)]
struct RowEntry {
    gap: u32,
    geometry: Arc<RowGeometry>,
}

#[derive(Clone, Copy, Default)]
struct RowEntrySummary {
    rows: u32,
    max_width: ScrollPixelOffset,
    unsettled: bool,
}

#[derive(Clone, Copy, Default, PartialEq, Eq, PartialOrd, Ord)]
struct RowCount(u32);

impl sum_tree::Item for RowEntry {
    type Summary = RowEntrySummary;

    fn summary(&self, (): ()) -> RowEntrySummary {
        RowEntrySummary {
            rows: self.gap + 1,
            max_width: self.geometry.width(),
            unsettled: !self.geometry.is_settled(),
        }
    }
}

impl sum_tree::ContextLessSummary for RowEntrySummary {
    fn zero() -> Self {
        Self::default()
    }

    fn add_summary(&mut self, summary: &Self) {
        self.rows += summary.rows;
        self.max_width = self.max_width.max(summary.max_width);
        self.unsettled |= summary.unsettled;
    }
}

impl<'a> sum_tree::Dimension<'a, RowEntrySummary> for RowCount {
    fn zero((): ()) -> Self {
        Self(0)
    }

    fn add_summary(&mut self, summary: &'a RowEntrySummary, (): ()) {
        self.0 += summary.rows;
    }
}

fn push_row(rows: &mut SumTree<RowEntry>, row: u32, geometry: Arc<RowGeometry>) {
    let gap = row.saturating_sub(rows.summary().rows);
    rows.push(RowEntry { gap, geometry }, ());
}

fn copy_rows_before(
    old_rows: &mut Cursor<'_, '_, RowEntry, RowCount>,
    rows: &mut SumTree<RowEntry>,
    limit: u32,
    row_delta: i64,
) -> bool {
    let Some(entry) = old_rows.item() else {
        return false;
    };
    let row = old_rows.start().0 + entry.gap;
    if row >= limit {
        return false;
    }
    push_row(rows, shift_row(row, row_delta), entry.geometry.clone());
    old_rows.next();
    rows.append(old_rows.slice(&RowCount(limit), Bias::Right), ());
    true
}

struct CharWidths {
    wrapper: Mutex<LineWrapperHandle>,
    estimate: ScrollPixelOffset,
    em_width: ScrollPixelOffset,
    base_font: FontId,
    metrics: RowMetrics,
    layouts: Mutex<LayoutCache>,
    chunks: Mutex<ChunkLayoutCache>,
    refinements: Mutex<HashMap<u32, Refinement>>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum GraphemeFloor {
    Boundary(u32),
    OversizedCluster,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct RowMetrics {
    pub(crate) ascent: Pixels,
    pub(crate) descent: Pixels,
}

#[derive(Default)]
struct ChunkLayoutCache {
    layouts: HashMap<usize, Arc<ChunkLayout>>,
    order: VecDeque<(usize, u32)>,
    columns: usize,
    frame: FrameRetained<usize, Arc<ChunkLayout>>,
}

struct PlacedChunk {
    start: u32,
    layout: Arc<ChunkLayout>,
}

struct FrameRetained<K, V> {
    recording: bool,
    current: HashMap<K, V>,
    previous: HashMap<K, V>,
}

impl<K, V> Default for FrameRetained<K, V> {
    fn default() -> Self {
        Self {
            recording: false,
            current: HashMap::default(),
            previous: HashMap::default(),
        }
    }
}

impl<K: Clone + Eq + std::hash::Hash, V: Clone> FrameRetained<K, V> {
    fn begin(&mut self) {
        if !self.current.is_empty() {
            self.previous = mem::take(&mut self.current);
        }
        self.recording = true;
    }

    fn end(&mut self) {
        self.recording = false;
    }

    fn is_idle(&self) -> bool {
        !self.recording && self.current.is_empty() && self.previous.is_empty()
    }

    fn get(&mut self, key: &K) -> Option<V> {
        if let Some(value) = self.current.get(key) {
            return Some(value.clone());
        }
        if !self.recording {
            return self.previous.get(key).cloned();
        }
        let value = self.previous.remove(key)?;
        self.current.insert(key.clone(), value.clone());
        Some(value)
    }

    fn record(&mut self, key: &K, value: &V) {
        if self.recording && !self.current.contains_key(key) {
            self.current.insert(key.clone(), value.clone());
        }
    }
}

impl ChunkLayoutCache {
    fn get(&mut self, layout_id: usize) -> Option<Arc<ChunkLayout>> {
        match self.layouts.get(&layout_id) {
            Some(layout) => {
                self.frame.record(&layout_id, layout);
                Some(layout.clone())
            }
            None => self.frame.get(&layout_id),
        }
    }

    fn insert(&mut self, layout_id: usize, columns: u32, layout: Arc<ChunkLayout>) {
        self.frame.record(&layout_id, &layout);
        if columns as usize > MAX_CACHED_CHUNK_COLUMNS {
            return;
        }
        if self.layouts.insert(layout_id, layout).is_some() {
            return;
        }
        self.order.push_back((layout_id, columns));
        self.columns += columns as usize;
        while self.columns > MAX_CACHED_CHUNK_COLUMNS {
            let Some((oldest, columns)) = self.order.pop_front() else {
                break;
            };
            self.columns -= columns as usize;
            self.layouts.remove(&oldest);
        }
    }
}

impl PlacedChunk {
    fn atom(&self, index: usize) -> Option<(u32, ScrollPixelOffset)> {
        self.layout
            .atoms
            .get(index)
            .map(|(offset, x)| (self.start + offset, *x))
    }

    fn atoms(&self) -> impl Iterator<Item = (u32, ScrollPixelOffset)> + '_ {
        self.layout
            .atoms
            .iter()
            .map(|(offset, x)| (self.start + offset, *x))
    }

    fn atoms_starting_before(&self, column: u32) -> usize {
        let Some(offset) = column.checked_sub(self.start) else {
            return 0;
        };
        self.layout
            .atoms
            .partition_point(|(atom_offset, _)| *atom_offset < offset)
    }

    fn pieces(
        &self,
        columns: Range<u32>,
        origin_x: ScrollPixelOffset,
    ) -> impl Iterator<Item = TextPiece> + '_ {
        let columns =
            columns.start.saturating_sub(self.start)..columns.end.saturating_sub(self.start);
        self.layout
            .pieces
            .iter()
            .filter_map(move |piece| piece.window(&columns, origin_x))
            .map(|piece| TextPiece {
                columns: self.start + piece.columns.start..self.start + piece.columns.end,
                ..piece
            })
    }
}

struct Refinement {
    version: usize,
    stored: Option<Arc<RowGeometry>>,
    geometry: Arc<RowGeometry>,
}

impl Refinement {
    fn refines(&self, long_row: &LongRow) -> bool {
        let same_source = match (&self.stored, &long_row.stored) {
            (Some(stored), Some(current)) => Arc::ptr_eq(stored, current),
            (None, None) => true,
            _ => false,
        };
        same_source
            && self.version == long_row.snapshot.version
            && self.geometry.len() == long_row.len()
            && self.geometry.text_id == long_row.geometry.text_id
            && (Arc::ptr_eq(&self.geometry.styles.0, &long_row.geometry.styles.0)
                || self.geometry.styles.0.is_empty() && long_row.geometry.styles.0.is_empty())
    }
}

#[derive(Clone, PartialEq, Eq, Hash)]
struct LayoutVariant {
    runs: SmallVec<[FontRun; 1]>,
    left: usize,
    right: usize,
}

type LayoutVariants = SmallVec<[(LayoutVariant, Arc<LineLayout>); 1]>;

#[derive(Default)]
struct LayoutCache {
    layouts: HashMap<String, LayoutVariants>,
    order: VecDeque<String>,
    bytes: usize,
    frame: FrameRetained<(String, LayoutVariant), Arc<LineLayout>>,
}

impl LayoutCache {
    fn get(&mut self, text: &str, variant: &LayoutVariant) -> Option<Arc<LineLayout>> {
        let cached = self.layouts.get(text).and_then(|variants| {
            variants
                .iter()
                .find(|(cached, _)| cached == variant)
                .map(|(_, layout)| layout.clone())
        });
        if self.frame.is_idle() {
            return cached;
        }
        let key = (text.to_string(), variant.clone());
        match cached {
            Some(layout) => {
                self.frame.record(&key, &layout);
                Some(layout)
            }
            None => self.frame.get(&key),
        }
    }

    fn insert(&mut self, text: &str, variant: LayoutVariant, layout: Arc<LineLayout>) {
        if self.frame.recording {
            self.frame
                .record(&(text.to_string(), variant.clone()), &layout);
        }
        if text.len() > MAX_CACHED_LAYOUT_BYTES {
            return;
        }
        if let Some(variants) = self.layouts.get_mut(text) {
            if variants.len() >= MAX_CACHED_LAYOUT_VARIANTS {
                variants.remove(0);
            }
            variants.push((variant, layout));
            return;
        }
        while self.bytes + text.len() > MAX_CACHED_LAYOUT_BYTES {
            let Some(oldest) = self.order.pop_front() else {
                break;
            };
            self.bytes -= oldest.len();
            self.layouts.remove(&oldest);
        }
        self.bytes += text.len();
        self.order.push_back(text.to_string());
        self.layouts
            .insert(text.to_string(), smallvec![(variant, layout)]);
    }
}

struct LongRowScan {
    stack: Vec<Range<u32>>,
}

enum ScanStep {
    Row { row: u32, len: u32 },
    Continue,
    Done,
}

impl LongRowScan {
    fn new(rows: Range<u32>) -> Self {
        Self { stack: vec![rows] }
    }

    fn step(&mut self, snapshot: &TabSnapshot) -> ScanStep {
        let Some(rows) = self.stack.pop() else {
            return ScanStep::Done;
        };
        if rows.is_empty() || snapshot.max_line_len_bound(rows.clone()) as usize <= MAX_LINE_LEN {
            return ScanStep::Continue;
        }
        if rows.len() == 1 {
            let len = snapshot.line_len(rows.start);
            return if len as usize > MAX_LINE_LEN {
                ScanStep::Row {
                    row: rows.start,
                    len,
                }
            } else {
                ScanStep::Continue
            };
        }
        let middle = rows.start + (rows.end - rows.start) / 2;
        self.stack.push(middle..rows.end);
        self.stack.push(rows.start..middle);
        ScanStep::Continue
    }
}

struct Atom {
    start: u32,
    end: u32,
    start_x: ScrollPixelOffset,
    end_x: ScrollPixelOffset,
}

#[derive(Clone)]
struct RowGeometry {
    chunks: SumTree<GeometryChunk>,
    char_width: ScrollPixelOffset,
    canonical_until: u32,
    text_id: usize,
    styles: RowStyles,
    revision: usize,
    measured_from: usize,
}

#[derive(Clone)]
struct GeometryChunk {
    len: u32,
    chars: u32,
    width: ScrollPixelOffset,
    measured: bool,
    canonical: bool,
    content_end: Option<u32>,
    runs: Option<ChunkRuns>,
    layout_id: usize,
    layout: Option<Arc<ChunkLayout>>,
}

#[derive(Clone, Copy, Debug, Default)]
struct ChunkSummary {
    len: u32,
    measured_width: ScrollPixelOffset,
    measured_chars: u64,
    unmeasured_chars: u64,
    unmeasured: u32,
    unsettled: u32,
    trailing: TrailingContent,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
enum TrailingContent {
    #[default]
    Whitespace,
    Unmeasured,
    Content(u32),
}

#[derive(Clone, Copy)]
struct ChunkPosition<'a> {
    chunk: &'a GeometryChunk,
    start: u32,
    x: ScrollPixelOffset,
    width: ScrollPixelOffset,
}

struct ChunkIter<'a> {
    geometry: &'a RowGeometry,
    cursor: Cursor<'a, 'static, GeometryChunk, ChunkSummary>,
}

#[derive(Default)]
struct ChunkBuilder {
    tree: SumTree<GeometryChunk>,
    pending: Vec<GeometryChunk>,
    pending_len: u32,
}

struct ColumnTarget(u32);

struct XTarget {
    x: ScrollPixelOffset,
    char_width: ScrollPixelOffset,
}

struct SettledTarget {
    unsettled: u32,
    canonical_until: u32,
}

static EMPTY_CHUNK: GeometryChunk = GeometryChunk {
    len: 0,
    chars: 0,
    width: 0.,
    measured: false,
    canonical: false,
    content_end: None,
    runs: None,
    layout_id: 0,
    layout: None,
};

impl sum_tree::Item for GeometryChunk {
    type Summary = ChunkSummary;

    fn summary(&self, (): ()) -> ChunkSummary {
        let (measured_chars, unmeasured_chars) = if self.measured {
            (u64::from(self.chars), 0)
        } else {
            (0, u64::from(self.chars))
        };
        ChunkSummary {
            len: self.len,
            measured_width: if self.measured { self.width } else { 0. },
            measured_chars,
            unmeasured_chars,
            unmeasured: u32::from(!self.measured),
            unsettled: u32::from(!self.is_canonical()),
            trailing: match (self.measured, self.content_end) {
                (false, _) => TrailingContent::Unmeasured,
                (true, Some(content_end)) => TrailingContent::Content(content_end),
                (true, None) => TrailingContent::Whitespace,
            },
        }
    }
}

impl sum_tree::ContextLessSummary for ChunkSummary {
    fn zero() -> Self {
        Self::default()
    }

    fn add_summary(&mut self, summary: &Self) {
        self.trailing = match summary.trailing {
            TrailingContent::Whitespace => self.trailing,
            TrailingContent::Unmeasured => TrailingContent::Unmeasured,
            TrailingContent::Content(content_end) => {
                TrailingContent::Content(self.len + content_end)
            }
        };
        self.len += summary.len;
        self.measured_width += summary.measured_width;
        self.measured_chars += summary.measured_chars;
        self.unmeasured_chars += summary.unmeasured_chars;
        self.unmeasured += summary.unmeasured;
        self.unsettled += summary.unsettled;
    }
}

impl ChunkSummary {
    fn x(&self, char_width: ScrollPixelOffset) -> ScrollPixelOffset {
        self.measured_width + self.unmeasured_chars as f64 * char_width
    }
}

impl<'a> sum_tree::SeekTarget<'a, ChunkSummary, ChunkSummary> for ColumnTarget {
    fn cmp(&self, location: &ChunkSummary, (): ()) -> Ordering {
        Ord::cmp(&self.0, &location.len)
    }
}

impl<'a> sum_tree::SeekTarget<'a, ChunkSummary, ChunkSummary> for XTarget {
    fn cmp(&self, location: &ChunkSummary, (): ()) -> Ordering {
        self.x
            .partial_cmp(&location.x(self.char_width))
            .unwrap_or(Ordering::Greater)
    }
}

impl<'a> sum_tree::SeekTarget<'a, ChunkSummary, ChunkSummary> for SettledTarget {
    fn cmp(&self, location: &ChunkSummary, (): ()) -> Ordering {
        if location.unsettled <= self.unsettled && location.len <= self.canonical_until {
            Ordering::Greater
        } else {
            Ordering::Less
        }
    }
}

impl ChunkPosition<'_> {
    fn end(&self) -> u32 {
        self.start + self.chunk.len
    }

    fn x_end(&self) -> ScrollPixelOffset {
        self.x + self.width
    }

    fn columns(&self) -> Range<u32> {
        self.start..self.end()
    }

    fn density(&self) -> ScrollPixelOffset {
        if self.chunk.len == 0 {
            return 0.;
        }
        self.width / f64::from(self.chunk.len)
    }

    fn is_fresh(&self, styles: &RowStyles, snapshot: &TabSnapshot, row: u32, base: FontId) -> bool {
        if !self.chunk.measured {
            return false;
        }
        if styles.0.is_empty() {
            return self.chunk.runs.is_none();
        }
        let span = shaping_span(snapshot, row, self.columns());
        self.chunk.runs == styles.chunk_runs(span, self.start, base)
    }
}

impl<'a> Iterator for ChunkIter<'a> {
    type Item = ChunkPosition<'a>;

    fn next(&mut self) -> Option<Self::Item> {
        let chunk = self.cursor.item()?;
        let position = self.geometry.position(self.cursor.start(), chunk);
        self.cursor.next();
        Some(position)
    }
}

impl ChunkBuilder {
    fn len(&self) -> u32 {
        self.tree.summary().len + self.pending_len
    }

    fn push(&mut self, chunk: GeometryChunk) {
        self.pending_len += chunk.len;
        self.pending.push(chunk);
    }

    fn append(&mut self, chunks: SumTree<GeometryChunk>) {
        self.flush();
        self.tree.append(chunks, ());
    }

    fn last_is_canonical(&self, canonical_until: u32) -> bool {
        self.pending
            .last()
            .or_else(|| self.tree.last())
            .is_some_and(|last| last.is_canonical() && self.len() - last.len < canonical_until)
    }

    fn push_unmeasured(&mut self, snapshot: &TabSnapshot, row: u32, end: u32) {
        let start = self.len();
        if end <= start {
            return;
        }
        let len = end - start;
        let chars = char_count(snapshot, row, start..end);
        let extend = |last: &mut GeometryChunk| {
            last.len += len;
            last.chars += chars;
        };
        if let Some(last) = self.pending.last_mut() {
            if !last.measured {
                extend(last);
                self.pending_len += len;
                return;
            }
        } else if self.tree.last().is_some_and(|last| !last.measured) {
            self.tree.update_last(extend, ());
            return;
        }
        self.push(GeometryChunk {
            len,
            chars,
            width: 0.,
            measured: false,
            canonical: false,
            content_end: None,
            runs: None,
            layout_id: 0,
            layout: None,
        });
    }

    fn finish(mut self) -> SumTree<GeometryChunk> {
        self.flush();
        self.tree
    }

    fn flush(&mut self) {
        if !self.pending.is_empty() {
            self.tree.extend(mem::take(&mut self.pending), ());
            self.pending_len = 0;
        }
    }
}

impl RowGeometry {
    fn unmeasured(
        snapshot: &TabSnapshot,
        row: u32,
        len: u32,
        estimate: ScrollPixelOffset,
        text_id: usize,
    ) -> Arc<Self> {
        let mut builder = ChunkBuilder::default();
        builder.push_unmeasured(snapshot, row, len);
        Arc::new(Self::new(
            builder.finish(),
            u32::MAX,
            text_id,
            RowStyles::default(),
            estimate,
        ))
    }

    fn new(
        chunks: SumTree<GeometryChunk>,
        canonical_until: u32,
        text_id: usize,
        styles: RowStyles,
        estimate: ScrollPixelOffset,
    ) -> Self {
        let summary = chunks.summary();
        let measured_char_width = summary.measured_width / summary.measured_chars as f64;
        let char_width = if summary.measured_chars > 0
            && (measured_char_width - estimate).abs() > estimate * ESTIMATE_RELATIVE_TOLERANCE
        {
            measured_char_width
        } else {
            estimate
        };
        Self {
            chunks,
            char_width,
            canonical_until,
            text_id,
            styles,
            revision: next_epoch(),
            measured_from: 0,
        }
    }

    fn with_styles(&self, styles: RowStyles) -> Self {
        Self {
            chunks: self.chunks.clone(),
            char_width: self.char_width,
            canonical_until: self.canonical_until,
            text_id: self.text_id,
            styles,
            revision: next_epoch(),
            measured_from: 0,
        }
    }

    fn len(&self) -> u32 {
        self.chunks.summary().len
    }

    fn width(&self) -> ScrollPixelOffset {
        self.chunks.summary().x(self.char_width)
    }

    fn is_measured(&self) -> bool {
        self.chunks.summary().unmeasured == 0
    }

    fn is_settled(&self) -> bool {
        self.chunks.summary().unsettled == 0 && self.canonical_until >= self.len()
    }

    fn is_settled_through(&self, column: u32) -> bool {
        if self.is_settled() {
            return true;
        }
        let column = column.min(self.len());
        if column == 0 {
            return true;
        }
        let mut cursor = self.cursor();
        cursor.seek(&ColumnTarget(column), Bias::Left);
        if cursor.start().unsettled > 0 {
            return false;
        }
        cursor
            .item()
            .is_some_and(|chunk| self.is_canonical(&self.position(cursor.start(), chunk)))
    }

    fn trailing_whitespace_start(&self) -> Option<u32> {
        match self.chunks.summary().trailing {
            TrailingContent::Whitespace => Some(0),
            TrailingContent::Unmeasured => None,
            TrailingContent::Content(content_end) => Some(content_end),
        }
    }

    fn position<'a>(&self, start: &ChunkSummary, chunk: &'a GeometryChunk) -> ChunkPosition<'a> {
        ChunkPosition {
            chunk,
            start: start.len,
            x: start.x(self.char_width),
            width: if chunk.measured {
                chunk.width
            } else {
                f64::from(chunk.chars) * self.char_width
            },
        }
    }

    fn cursor(&self) -> Cursor<'_, 'static, GeometryChunk, ChunkSummary> {
        self.chunks.cursor::<ChunkSummary>(())
    }

    fn chunk_containing(&self, column: u32) -> ChunkPosition<'_> {
        let mut cursor = self.cursor();
        cursor.seek(
            &ColumnTarget(column.min(self.len().saturating_sub(1))),
            Bias::Right,
        );
        match cursor.item() {
            Some(chunk) => self.position(cursor.start(), chunk),
            None => ChunkPosition {
                chunk: &EMPTY_CHUNK,
                start: self.len(),
                x: self.width(),
                width: 0.,
            },
        }
    }

    fn chunk_at_x(&self, x: ScrollPixelOffset) -> ChunkPosition<'_> {
        self.chunks_from_x(x)
            .next()
            .unwrap_or_else(|| self.chunk_containing(self.len()))
    }

    fn chunk_starting_at(&self, column: u32) -> Option<ChunkPosition<'_>> {
        self.chunks_from(column)
            .next()
            .filter(|position| position.start == column)
    }

    fn chunks_from(&self, column: u32) -> ChunkIter<'_> {
        let mut cursor = self.cursor();
        cursor.seek(&ColumnTarget(column), Bias::Right);
        ChunkIter {
            geometry: self,
            cursor,
        }
    }

    fn chunks_from_x(&self, x: ScrollPixelOffset) -> ChunkIter<'_> {
        let mut cursor = self.cursor();
        cursor.seek(
            &XTarget {
                x: if x > 0. { x } else { 0. },
                char_width: self.char_width,
            },
            Bias::Right,
        );
        ChunkIter {
            geometry: self,
            cursor,
        }
    }

    #[cfg(test)]
    fn placed_chunks(&self) -> ChunkIter<'_> {
        self.chunks_from(0)
    }

    fn is_canonical(&self, position: &ChunkPosition<'_>) -> bool {
        position.chunk.is_canonical() && position.start < self.canonical_until
    }

    fn is_canonical_boundary(&self, column: u32) -> bool {
        if column == 0 || column >= self.len() {
            return true;
        }
        let mut cursor = self.cursor();
        cursor.seek(&ColumnTarget(column), Bias::Left);
        let Some(chunk) = cursor.item() else {
            return false;
        };
        let ending = self.position(cursor.start(), chunk);
        if ending.end() != column {
            return false;
        }
        if self.is_canonical(&ending) {
            return true;
        }
        cursor.next();
        cursor
            .item()
            .is_some_and(|next| self.is_canonical(&self.position(cursor.start(), next)))
    }

    fn has_unmeasured(&self, columns: &Range<u32>) -> bool {
        if self.is_measured() || columns.is_empty() {
            return false;
        }
        let mut cursor = self.cursor();
        cursor.seek(&ColumnTarget(columns.start), Bias::Right);
        let before = cursor.start().unmeasured;
        cursor.seek_forward(&ColumnTarget(columns.end), Bias::Left);
        let mut through = cursor.start().unmeasured;
        if let Some(chunk) = cursor.item()
            && cursor.start().len < columns.end
            && !chunk.measured
        {
            through += 1;
        }
        through > before
    }

    fn first_unsettled_from(&self, column: u32) -> u32 {
        let len = self.len();
        if column >= len || column >= self.canonical_until {
            return column.min(len);
        }
        let mut cursor = self.cursor();
        cursor.seek(&ColumnTarget(column), Bias::Right);
        let target = SettledTarget {
            unsettled: cursor.start().unsettled,
            canonical_until: self.canonical_until,
        };
        cursor.seek_forward(&target, Bias::Right);
        match cursor.item() {
            None => len,
            Some(chunk) => {
                let position = self.position(cursor.start(), chunk);
                if self.is_canonical(&position) {
                    position.end()
                } else {
                    position.start
                }
            }
        }
    }

    fn measure_columns(
        &self,
        snapshot: &TabSnapshot,
        row: u32,
        ranges: &[Range<u32>],
        runs: &[RowRuns],
        wrapper: &mut LineWrapper,
        widths: &CharWidths,
        cache: Option<&Mutex<LayoutCache>>,
    ) -> Option<Self> {
        let base = widths.base_font;
        let restyled = self.styles.with(runs).map(|(styles, changed)| {
            self.restyled(snapshot, row, styles, &changed, wrapper, widths, cache)
        });
        let geometry = restyled.as_ref().unwrap_or(self);
        if !ranges.iter().any(|range| geometry.has_unmeasured(range)) {
            return restyled;
        }
        let styles = geometry.styles.view();
        let len = geometry.len();
        let mut ranges = ranges
            .iter()
            .filter(|range| !range.is_empty())
            .cloned()
            .collect::<Vec<_>>();
        ranges.sort_unstable_by_key(|range| range.start);
        let mut builder = ChunkBuilder::default();
        let mut old_chunks = geometry.cursor();
        old_chunks.seek(&ColumnTarget(0), Bias::Right);
        let mut canonical_until = geometry.canonical_until;
        let mut changed = restyled.is_some();
        for range in &ranges {
            let target = range.start.max(old_chunks.start().len);
            builder.append(old_chunks.slice(&ColumnTarget(target), Bias::Right));
            while let Some(chunk) = old_chunks.item() {
                let position = geometry.position(old_chunks.start(), chunk);
                if position.start >= range.end {
                    break;
                }
                old_chunks.next();
                if chunk.measured {
                    builder.push(chunk.clone());
                    continue;
                }
                let chunk_end = position.end();
                let end_is_canonical = geometry.is_canonical_boundary(chunk_end);
                let mut column = position.start;
                let mut column_is_canonical = geometry.is_canonical_boundary(column);
                for range in &ranges {
                    if range.end <= column || range.start >= chunk_end {
                        continue;
                    }
                    let target = range.start.max(column);
                    let (start, canonical) =
                        if column_is_canonical && target - column <= MAX_CUT_SCAN_LEN {
                            (
                                last_cut_before(
                                    snapshot,
                                    row,
                                    column,
                                    target,
                                    chunk_end,
                                    styles,
                                    SYNC_MEASURE_LEN + RESYNC_RESERVE_LEN,
                                    wrapper,
                                ),
                                true,
                            )
                        } else {
                            let GraphemeFloor::Boundary(start) =
                                grapheme_floor(snapshot, row, len, target, column)
                            else {
                                continue;
                            };
                            (start, column_is_canonical && start == column)
                        };
                    builder.push_unmeasured(snapshot, row, start);
                    let walked = measure_chunks(
                        ChunkWalk {
                            snapshot,
                            row,
                            columns: start..chunk_end,
                            max_len: range.end.saturating_sub(start).max(1),
                            budget: Some(SYNC_MEASURE_LEN + RESYNC_RESERVE_LEN),
                            canonical,
                            end_is_canonical,
                            runs: styles,
                            base,
                            cache,
                        },
                        wrapper,
                        &mut builder,
                    );
                    if canonical && start <= canonical_until {
                        canonical_until = canonical_until.max(walked);
                    }
                    changed |= walked > start;
                    column = walked;
                    column_is_canonical = canonical;
                }
                builder.push_unmeasured(snapshot, row, chunk_end);
            }
        }
        builder.append(old_chunks.suffix());
        changed.then(|| {
            Self::new(
                builder.finish(),
                canonical_until,
                self.text_id,
                geometry.styles.clone(),
                widths.estimate,
            )
        })
    }

    fn restyled(
        &self,
        snapshot: &TabSnapshot,
        row: u32,
        styles: RowStyles,
        changed: &[Range<u32>],
        wrapper: &mut LineWrapper,
        widths: &CharWidths,
        cache: Option<&Mutex<LayoutCache>>,
    ) -> Self {
        let len = self.len();
        let differences = self
            .styles
            .font_differences(&styles, changed, len, widths.base_font);
        if differences.is_empty() {
            return self.with_styles(styles);
        }
        let mut budget = 0_u32;
        let mut groups = differences.iter().peekable();
        while let Some(first) = groups.next() {
            let mut group_end = first.end;
            while let Some(next) =
                groups.next_if(|next| next.start <= group_end.saturating_add(SYNC_MEASURE_LEN))
            {
                group_end = group_end.max(next.end);
            }
            budget = budget
                .saturating_add(group_end - first.start)
                .saturating_add(2 * RESYNC_RESERVE_LEN)
                .saturating_add(SYNC_MEASURE_LEN + RESYNC_RESERVE_LEN);
        }
        let edits = differences
            .into_iter()
            .map(ColumnEdit::restyle)
            .collect::<Vec<_>>();
        let mut measurer = Measurer {
            snapshot,
            wrapper,
            estimate: widths.estimate,
            budget,
            styles,
            cache,
        };
        measurer
            .reshape(self, &edits, SHAPING_CONTEXT_LEN, row, len, self.text_id)
            .unwrap_or_else(|| {
                let mut builder = ChunkBuilder::default();
                builder.push_unmeasured(snapshot, row, len);
                measurer.geometry(builder, u32::MAX, self.text_id)
            })
    }

    fn merged(
        &self,
        measured: &RowGeometry,
        snapshot: &TabSnapshot,
        row: u32,
        widths: &CharWidths,
    ) -> Option<Self> {
        if measured.measured_from == self.revision {
            return Some(measured.clone());
        }
        let len = self.len();
        if measured.len() != len {
            return None;
        }
        let base = widths.base_font;
        let agreement_end = measured.styles.agreement_end(&self.styles, base);
        let cut = measured.fresh_end(snapshot, row, agreement_end);
        let mut builder = ChunkBuilder::default();
        let mut changed = false;
        let mut canonical_until = u32::MAX;
        let mut column = 0;
        while column < len {
            let stale = measured.first_unsettled_from(column).min(cut);
            if stale >= len {
                changed |= self.first_unsettled_from(column) < len;
                builder.append(measured.slice(column..len));
                break;
            }
            let start = self
                .last_common_boundary(measured, stale.max(column))
                .max(column);
            changed |= self.first_unsettled_from(column) < start;
            builder.append(measured.slice(column..start));
            let merge = ChunkMerge {
                current: self,
                measured,
                cut,
                snapshot,
                row,
                base,
            };
            if stale >= cut {
                let end = self.boundary_at_or_after(cut).max(start);
                changed |= merge.merge_chunks(start..end, &mut builder);
                builder.append(self.slice(end..len));
                if self.canonical_until < len {
                    canonical_until = self.canonical_until.max(end);
                }
                break;
            }
            let end = self.next_common_boundary(measured, measured.chunk_containing(stale).end());
            changed |= merge.merge_chunks(start..end, &mut builder);
            column = end;
        }
        changed.then(|| {
            Self::new(
                builder.finish(),
                canonical_until,
                self.text_id,
                self.styles.clone(),
                widths.estimate,
            )
        })
    }

    fn fresh_end(&self, snapshot: &TabSnapshot, row: u32, agreement_end: u32) -> u32 {
        let len = self.len();
        if agreement_end >= len {
            return len;
        }
        let mut cursor = self.cursor();
        cursor.seek(&ColumnTarget(agreement_end), Bias::Right);
        let mut fresh_end = cursor.start().len;
        loop {
            cursor.prev();
            let Some(chunk) = cursor.item() else {
                break;
            };
            let position = self.position(cursor.start(), chunk);
            if shaping_span(snapshot, row, position.columns()).end <= agreement_end {
                break;
            }
            fresh_end = position.start;
        }
        fresh_end
    }

    fn slice(&self, columns: Range<u32>) -> SumTree<GeometryChunk> {
        if columns.is_empty() {
            return SumTree::default();
        }
        let mut cursor = self.cursor();
        cursor.seek(&ColumnTarget(columns.start), Bias::Right);
        cursor.slice(&ColumnTarget(columns.end), Bias::Right)
    }

    fn boundary_at_or_before(&self, column: u32) -> u32 {
        if column >= self.len() {
            return self.len();
        }
        self.chunk_containing(column).start
    }

    fn boundary_at_or_after(&self, column: u32) -> u32 {
        if column >= self.len() {
            return self.len();
        }
        let position = self.chunk_containing(column);
        if position.start == column {
            column
        } else {
            position.end()
        }
    }

    fn last_common_boundary(&self, other: &RowGeometry, mut column: u32) -> u32 {
        loop {
            let own = self.boundary_at_or_before(column);
            let other_boundary = other.boundary_at_or_before(own);
            if other_boundary == own {
                return own;
            }
            column = other_boundary;
        }
    }

    fn next_common_boundary(&self, other: &RowGeometry, mut column: u32) -> u32 {
        loop {
            let own = self.boundary_at_or_after(column);
            let other_boundary = other.boundary_at_or_after(own);
            if other_boundary == own {
                return own;
            }
            column = other_boundary;
        }
    }

    async fn measure_remaining(
        &self,
        snapshot: &TabSnapshot,
        row: u32,
        wrapper: &mut LineWrapper,
    ) -> Self {
        let len = self.len();
        let base = wrapper.font_id();
        let mut builder = ChunkBuilder::default();
        let mut old_chunks = self.cursor();
        old_chunks.seek(&ColumnTarget(0), Bias::Right);
        let mut measured_since_yield = 0;
        let cache = Mutex::new(LayoutCache::default());
        loop {
            let unsettled_start = self.first_unsettled_from(old_chunks.start().len);
            builder.append(old_chunks.slice(&ColumnTarget(unsettled_start), Bias::Right));
            if unsettled_start >= len {
                break;
            }
            let mut start = unsettled_start;
            while start < len {
                if start > unsettled_start
                    && self.chunk_starting_at(start).is_some_and(|next| {
                        self.is_canonical(&next) && next.is_fresh(&self.styles, snapshot, row, base)
                    })
                {
                    old_chunks.seek_forward(&ColumnTarget(start), Bias::Right);
                    break;
                }
                let walked = measure_chunks(
                    ChunkWalk {
                        snapshot,
                        row,
                        columns: start..len,
                        max_len: 1,
                        budget: None,
                        canonical: true,
                        end_is_canonical: true,
                        runs: self.styles.view(),
                        base,
                        cache: Some(&cache),
                    },
                    wrapper,
                    &mut builder,
                );
                measured_since_yield += walked - start;
                start = walked;
                if measured_since_yield >= MEASURE_YIELD_LEN {
                    measured_since_yield = 0;
                    yield_now().await;
                }
            }
            if start >= len {
                break;
            }
        }
        Self {
            measured_from: self.revision,
            ..Self::new(
                builder.finish(),
                u32::MAX,
                self.text_id,
                self.styles.clone(),
                self.char_width,
            )
        }
    }
}

struct ChunkMerge<'a> {
    current: &'a RowGeometry,
    measured: &'a RowGeometry,
    cut: u32,
    snapshot: &'a TabSnapshot,
    row: u32,
    base: FontId,
}

impl ChunkMerge<'_> {
    fn merge_chunks(&self, columns: Range<u32>, builder: &mut ChunkBuilder) -> bool {
        let mut changed = false;
        for position in self.current.chunks_from(columns.start) {
            if position.start >= columns.end {
                break;
            }
            if position.is_fresh(&self.current.styles, self.snapshot, self.row, self.base) {
                let mut chunk = position.chunk.clone();
                chunk.canonical = self.current.is_canonical(&position);
                builder.push(chunk);
            } else {
                changed |= position.chunk.measured;
                changed |= self.fill_gap(position.columns(), builder);
            }
        }
        changed
    }

    fn fill_gap(&self, gap: Range<u32>, builder: &mut ChunkBuilder) -> bool {
        let measured = self.measured;
        let mut filled = false;
        let mut column = measured.boundary_at_or_after(gap.start);
        while column < gap.end.min(self.cut) {
            let stale = measured.first_unsettled_from(column).min(self.cut);
            let fill_end = measured.boundary_at_or_before(stale.min(gap.end));
            if fill_end > column {
                builder.push_unmeasured(self.snapshot, self.row, column);
                builder.append(measured.slice(column..fill_end));
                filled = true;
            }
            if stale >= gap.end || stale >= self.cut {
                break;
            }
            column = measured.chunk_containing(stale).end();
        }
        builder.push_unmeasured(self.snapshot, self.row, gap.end);
        filled
    }
}

struct Measurer<'a> {
    snapshot: &'a TabSnapshot,
    wrapper: &'a mut LineWrapper,
    estimate: ScrollPixelOffset,
    budget: u32,
    styles: RowStyles,
    cache: Option<&'a Mutex<LayoutCache>>,
}

enum Resync {
    At(u32),
    Unmeasured(u32),
    RowEnd,
    Exhausted,
}

impl Measurer<'_> {
    fn measure_prefix(&mut self, row: u32, len: u32, text_id: usize) -> RowGeometry {
        self.styles = RowStyles::default();
        let mut builder = ChunkBuilder::default();
        let mut start = 0;
        while start < len {
            match self.measure_chunk(row, start, len, true, &mut builder) {
                Some(end) => start = end,
                None => {
                    builder.push_unmeasured(self.snapshot, row, len);
                    break;
                }
            }
        }
        self.geometry(builder, u32::MAX, text_id)
    }

    fn geometry(&self, builder: ChunkBuilder, canonical_until: u32, text_id: usize) -> RowGeometry {
        RowGeometry::new(
            builder.finish(),
            canonical_until,
            text_id,
            self.styles.clone(),
            self.estimate,
        )
    }

    fn measure_chunk(
        &mut self,
        row: u32,
        start: u32,
        len: u32,
        canonical: bool,
        builder: &mut ChunkBuilder,
    ) -> Option<u32> {
        if self.budget <= RESYNC_RESERVE_LEN {
            return None;
        }
        let end = self.shape(row, start..len, true, canonical, builder);
        (end > start).then_some(end)
    }

    fn walk_start(
        &mut self,
        row: u32,
        start: u32,
        target: u32,
        len: u32,
        canonical: bool,
    ) -> Option<(u32, bool)> {
        if target <= start {
            return None;
        }
        if canonical && target - start <= MAX_CUT_SCAN_LEN.min(self.budget) {
            let runs = self.styles.view();
            let cut = last_cut_before(
                self.snapshot,
                row,
                start,
                target,
                target,
                runs,
                self.budget,
                self.wrapper,
            );
            return (cut > start).then_some((cut, true));
        }
        match grapheme_floor(self.snapshot, row, len, target, start) {
            GraphemeFloor::Boundary(column) if column > start => Some((column, false)),
            _ => None,
        }
    }

    fn splice(
        &mut self,
        old: &RowGeometry,
        edits: &[&TabEdit],
        row: u32,
        new_len: u32,
        text_id: usize,
    ) -> Option<RowGeometry> {
        let edits = edits
            .iter()
            .map(|edit| ColumnEdit::from(*edit))
            .collect::<Vec<_>>();
        let len_delta = edits.iter().map(ColumnEdit::delta).sum::<i64>();
        if i64::from(old.len()) + len_delta != i64::from(new_len) {
            return None;
        }
        self.styles = old.styles.spliced(&edits);
        self.reshape(
            old,
            &edits,
            SHAPING_CONTEXT_LEN.max(CAP_SCAN_LEN),
            row,
            new_len,
            text_id,
        )
    }

    fn reshape(
        &mut self,
        old: &RowGeometry,
        edits: &[ColumnEdit],
        reach: u32,
        row: u32,
        new_len: u32,
        text_id: usize,
    ) -> Option<RowGeometry> {
        let mut builder = ChunkBuilder::default();
        let mut old_chunks = old.cursor();
        old_chunks.seek(&ColumnTarget(0), Bias::Right);
        let mut copied = 0;
        let mut column_delta = 0i64;
        let mut tail_canonical = true;
        let mut canonical_until = u32::MAX;
        let mut edits = edits.iter().peekable();
        while let Some(first) = edits.next() {
            let start_old = old
                .chunk_containing(first.old.start.saturating_sub(reach))
                .start
                .max(copied);
            copy_chunks(
                &mut builder,
                &mut old_chunks,
                old,
                copied..start_old,
                column_delta,
                &mut canonical_until,
            );
            let start_chunk = old.chunk_containing(start_old);
            let mut start = builder.len();
            let mut canonical = tail_canonical
                && (start == 0
                    || builder.last_is_canonical(canonical_until)
                    || old.is_canonical(&start_chunk));
            if !start_chunk.chunk.measured
                && let Some((walk_start, walk_canonical)) = self.walk_start(
                    row,
                    start,
                    first.new.start.saturating_sub(reach),
                    new_len,
                    canonical,
                )
            {
                builder.push_unmeasured(self.snapshot, row, walk_start);
                start = walk_start;
                canonical = walk_canonical;
            }
            let mut delta = column_delta + first.delta();
            let mut edits_end = first.new.end;
            let resync = loop {
                while let Some(next) = edits.next_if(|next| next.new.start <= start) {
                    delta += next.delta();
                    edits_end = edits_end.max(next.new.end);
                }
                if start >= new_len {
                    break Resync::RowEnd;
                }
                if let Some(old_start) = (start >= edits_end)
                    .then(|| u32::try_from(i64::from(start) - delta).ok())
                    .flatten()
                {
                    if start >= edits_end + reach
                        && let Some(next) = old
                            .chunk_starting_at(old_start)
                            .filter(|next| next.start >= start_old)
                    {
                        break Resync::At(next.start);
                    }
                    let containing = old.chunk_containing(old_start);
                    if containing.start >= start_old
                        && !containing.chunk.measured
                        && containing.start < old_start
                        && edits.peek().is_none_or(|next| {
                            next.old.start >= containing.end().saturating_add(reach)
                        })
                    {
                        break Resync::Unmeasured(containing.end());
                    }
                }
                match self.measure_chunk(row, start, new_len, canonical, &mut builder) {
                    Some(end) => start = end,
                    None => break Resync::Exhausted,
                }
            };
            let resume = match resync {
                Resync::At(old_column) => {
                    tail_canonical &= old.is_canonical_boundary(old_column);
                    old_column
                }
                Resync::Unmeasured(old_end) => {
                    builder.push_unmeasured(self.snapshot, row, shift_column(old_end, delta));
                    tail_canonical = false;
                    old_end
                }
                Resync::RowEnd => return Some(self.geometry(builder, canonical_until, text_id)),
                Resync::Exhausted => {
                    let next_edit = edits.peek().map(|next| next.old.start);
                    let Some(old_column) = (start >= edits_end)
                        .then(|| {
                            self.grapheme_aligned_chunk(old, row, start, delta, new_len, next_edit)
                        })
                        .flatten()
                    else {
                        builder.push_unmeasured(self.snapshot, row, new_len);
                        return Some(self.geometry(builder, canonical_until, text_id));
                    };
                    let gap_end = shift_column(old_column, delta);
                    if gap_end > start {
                        self.measure_gap(row, start..gap_end, &mut builder);
                    }
                    tail_canonical = false;
                    old_column
                }
            };
            column_delta = delta;
            copied = resume;
            tail_canonical &= canonical;
            if !tail_canonical {
                canonical_until = canonical_until.min(builder.len());
            }
        }
        copy_chunks(
            &mut builder,
            &mut old_chunks,
            old,
            copied..old.len(),
            column_delta,
            &mut canonical_until,
        );
        Some(self.geometry(builder, canonical_until, text_id))
    }

    fn measure_gap(&mut self, row: u32, columns: Range<u32>, builder: &mut ChunkBuilder) {
        let len = columns.end - columns.start;
        if len > self.budget || len > RESYNC_RESERVE_LEN {
            builder.push_unmeasured(self.snapshot, row, columns.end);
            return;
        }
        let end = columns.end;
        self.shape(row, columns, false, false, builder);
        builder.push_unmeasured(self.snapshot, row, end);
    }

    fn shape(
        &mut self,
        row: u32,
        columns: Range<u32>,
        cut: bool,
        canonical: bool,
        builder: &mut ChunkBuilder,
    ) -> u32 {
        let start = columns.start;
        let base = self.wrapper.font_id();
        let layout = shape_chunk(
            self.snapshot,
            row,
            ChunkRequest {
                columns,
                cut,
                runs: self.styles.view(),
                budget: Some(self.budget),
            },
            self.wrapper,
            self.cache,
        );
        let end = layout.end;
        if end == start {
            self.budget = 0;
            return start;
        }
        self.budget = self.budget.saturating_sub(end - start);
        let span = shaping_span(self.snapshot, row, start..end);
        let runs = ChunkRuns::new(
            piece_runs(self.styles.view(), span.clone(), base),
            span,
            start,
            base,
        );
        builder.push(layout.chunk(
            start,
            canonical,
            char_count(self.snapshot, row, start..end),
            runs,
        ));
        end
    }

    fn grapheme_aligned_chunk(
        &self,
        old: &RowGeometry,
        row: u32,
        start: u32,
        delta: i64,
        new_len: u32,
        next_edit: Option<u32>,
    ) -> Option<u32> {
        let old_start = u32::try_from(i64::from(start) - delta).ok()?;
        let mut cursor = old.cursor();
        cursor.seek(&ColumnTarget(old_start), Bias::Left);
        cursor.item()?;
        cursor.next();
        for _ in 0..MAX_RESYNC_ATTEMPTS {
            cursor.item()?;
            let chunk_start = cursor.start().len;
            if next_edit.is_some_and(|next| chunk_start > next) {
                return None;
            }
            let column = shift_column(chunk_start, delta);
            if column >= start + SHAPING_CONTEXT_LEN
                && grapheme_floor(self.snapshot, row, new_len, column, start)
                    == GraphemeFloor::Boundary(column)
            {
                return Some(chunk_start);
            }
            cursor.next();
        }
        None
    }
}

impl GeometryChunk {
    fn is_canonical(&self) -> bool {
        self.measured && self.canonical
    }
}

fn copy_chunks(
    builder: &mut ChunkBuilder,
    old_chunks: &mut Cursor<'_, '_, GeometryChunk, ChunkSummary>,
    old: &RowGeometry,
    columns: Range<u32>,
    delta: i64,
    canonical_until: &mut u32,
) {
    if columns.is_empty() {
        return;
    }
    if old.canonical_until < columns.end {
        *canonical_until =
            (*canonical_until).min(shift_column(old.canonical_until.max(columns.start), delta));
    }
    old_chunks.seek_forward(&ColumnTarget(columns.start), Bias::Right);
    builder.append(old_chunks.slice(&ColumnTarget(columns.end), Bias::Right));
}

struct RowEditGroup<'a> {
    old_rows: Range<u32>,
    new_rows: Range<u32>,
    edits: Vec<&'a TabEdit>,
}

impl<'a> RowEditGroup<'a> {
    fn next(edits: &mut Peekable<impl Iterator<Item = &'a TabEdit>>) -> Option<Self> {
        let first = edits.next()?;
        let mut group = Self {
            old_rows: first.old.start.row()..first.old.end.row() + 1,
            new_rows: first.new.start.row()..first.new.end.row() + 1,
            edits: vec![first],
        };
        while let Some(edit) = edits.next_if(|edit| edit.old.start.row() < group.old_rows.end) {
            group.old_rows.end = group.old_rows.end.max(edit.old.end.row() + 1);
            group.new_rows.end = group.new_rows.end.max(edit.new.end.row() + 1);
            group.edits.push(edit);
        }
        Some(group)
    }

    fn is_single_row(&self) -> bool {
        self.old_rows.len() == 1 && self.new_rows.len() == 1
    }
}

#[derive(Clone, Debug)]
struct ColumnEdit {
    old: Range<u32>,
    new: Range<u32>,
}

impl ColumnEdit {
    fn restyle(columns: Range<u32>) -> Self {
        Self {
            old: columns.clone(),
            new: columns,
        }
    }

    fn delta(&self) -> i64 {
        i64::from(self.new.end) - i64::from(self.new.start) + i64::from(self.old.start)
            - i64::from(self.old.end)
    }
}

impl From<&TabEdit> for ColumnEdit {
    fn from(edit: &TabEdit) -> Self {
        Self {
            old: edit.old.start.column()..edit.old.end.column(),
            new: edit.new.start.column()..edit.new.end.column(),
        }
    }
}

#[derive(Default)]
struct ChunkLayout {
    end: u32,
    width: ScrollPixelOffset,
    content_end: Option<u32>,
    atoms: Vec<(u32, ScrollPixelOffset)>,
    pieces: Vec<TextPiece>,
}

struct ChunkRequest<'a> {
    columns: Range<u32>,
    cut: bool,
    runs: RunsView<'a>,
    budget: Option<u32>,
}

#[derive(Default)]
struct PendingText {
    start: u32,
    text: String,
    atoms: Vec<(u32, usize)>,
}

impl ChunkLayout {
    fn chunk(
        self,
        start: u32,
        canonical: bool,
        chars: u32,
        runs: Option<ChunkRuns>,
    ) -> GeometryChunk {
        GeometryChunk {
            len: self.end - start,
            chars,
            width: self.width,
            measured: true,
            canonical,
            content_end: self
                .content_end
                .map(|content_end| content_end.saturating_sub(start)),
            runs,
            layout_id: next_epoch(),
            layout: (self.end - start > FORCED_CHUNK_LEN).then(|| Arc::new(self.relative(start))),
        }
    }

    fn relative(mut self, start: u32) -> Self {
        self.end -= start;
        self.content_end = self.content_end.map(|end| end.saturating_sub(start));
        for (column, _) in &mut self.atoms {
            *column -= start;
        }
        for piece in &mut self.pieces {
            piece.columns = piece.columns.start - start..piece.columns.end - start;
        }
        self
    }

    fn boundary(&self, column: u32, bias: Bias) -> u32 {
        if column >= self.end {
            return self.end;
        }
        let index = self
            .pieces
            .partition_point(|piece| piece.columns.end <= column);
        let Some(piece) = self
            .pieces
            .get(index)
            .filter(|piece| piece.columns.start <= column)
        else {
            let index = match bias {
                Bias::Left => self
                    .atoms
                    .partition_point(|(start, _)| *start <= column)
                    .saturating_sub(1),
                Bias::Right => self.atoms.partition_point(|(start, _)| *start < column),
            };
            return self.atoms.get(index).map_or(
                if bias == Bias::Left { 0 } else { self.end },
                |(start, _)| *start,
            );
        };
        match &piece.index {
            Some(index) => {
                piece.columns.start
                    + index
                        .boundary((column - piece.columns.start) as usize, bias)
                        .0 as u32
            }
            None if bias == Bias::Left || column == piece.columns.start => piece.columns.start,
            None => piece.columns.end,
        }
    }

    fn push_text(
        &mut self,
        pending: &mut PendingText,
        request: &ChunkRequest<'_>,
        context: (&str, &str),
        wrapper: &mut LineWrapper,
        cache: Option<&Mutex<LayoutCache>>,
    ) {
        if pending.text.is_empty() {
            return;
        }
        let (left, right) = context;
        let base = wrapper.font_id();
        let shaped_start = pending.start - left.len() as u32;
        let shaped_end = pending.start + (pending.text.len() + right.len()) as u32;
        let shaped_runs = piece_runs(request.runs, shaped_start..shaped_end, base);
        let (layout, index) = shape_text(left, &pending.text, right, shaped_runs, wrapper, cache);
        let x = self.width;
        let x_end = x + index
            .as_ref()
            .map_or(f64::from(layout.width), |index| index.width);
        for (atom_start, glyph_x) in pending
            .atoms
            .iter()
            .map(|(atom_start, _)| *atom_start)
            .zip(atom_offsets_x(&layout, &pending.atoms, index.as_deref()))
        {
            let previous_x = self.atoms.last().map_or(x, |(_, previous_x)| *previous_x);
            let atom_x = (x + glyph_x).clamp(previous_x, x_end.max(previous_x));
            self.atoms.push((atom_start, atom_x));
        }
        pending.atoms.clear();
        self.width = x_end;
        let start = pending.start;
        let text = mem::take(&mut pending.text);
        let columns = start..start + text.len() as u32;
        self.pieces.push(TextPiece {
            runs: piece_runs(request.runs, columns.clone(), base),
            columns,
            x,
            text: SharedString::from(text),
            layout,
            index,
        });
    }
}

fn atom_offsets_x(
    layout: &LineLayout,
    atoms: &[(u32, usize)],
    index: Option<&LayoutIndex>,
) -> Vec<ScrollPixelOffset> {
    let mut left_edges = vec![None::<ScrollPixelOffset>; atoms.len()];
    for (offset, glyph) in layout.runs.iter().flat_map(|run| &run.glyphs).enumerate() {
        let Some(atom) = atoms
            .partition_point(|(_, offset)| *offset <= glyph.index)
            .checked_sub(1)
        else {
            continue;
        };
        let x = index.map_or(f64::from(glyph.position.x), |index| index.positions[offset]);
        let left_edge = &mut left_edges[atom];
        *left_edge = Some(left_edge.map_or(x, |left_edge| left_edge.min(x)));
    }
    let width = index.map_or(f64::from(layout.width), |index| index.width);
    let mut next_left_edges = vec![width; atoms.len() + 1];
    for atom in (0..atoms.len()).rev() {
        next_left_edges[atom] = left_edges[atom].unwrap_or(next_left_edges[atom + 1]);
    }
    let mut positions = Vec::with_capacity(atoms.len());
    let mut atom = 0;
    while atom < atoms.len() {
        let Some(cluster_x) = left_edges[atom] else {
            positions.push(next_left_edges[atom]);
            atom += 1;
            continue;
        };
        let group_end = atom
            + 1
            + left_edges[atom + 1..]
                .iter()
                .take_while(|left_edge| left_edge.is_none())
                .count();
        let cluster_end_x = next_left_edges[group_end];
        let parts = (group_end - atom) as f64;
        for rank in 0..group_end - atom {
            positions.push(cluster_x + (cluster_end_x - cluster_x) * rank as f64 / parts);
        }
        atom = group_end;
    }
    positions
}

fn shape_text(
    left: &str,
    text: &str,
    right: &str,
    runs: SmallVec<[FontRun; 1]>,
    wrapper: &mut LineWrapper,
    cache: Option<&Mutex<LayoutCache>>,
) -> (Arc<LineLayout>, Option<Arc<LayoutIndex>>) {
    let shaped = if left.is_empty() && right.is_empty() {
        text.to_string()
    } else {
        [left, text, right].concat()
    };
    let base = wrapper.font_id();
    let runs = checked_runs(&shaped, runs, base);
    let target = left.len()..left.len() + text.len();
    if text.len() > FORCED_CHUNK_LEN as usize {
        let (layout, index, range) = shape_in_context_with(shaped.len(), target.clone(), |range| {
            let fonts = slice_font_runs(&runs, 0, range.start as u32..range.end as u32);
            let (layout, positions, width) =
                wrapper.layout_line_with_precise_positions(&shaped[range], &fonts);
            let index = LayoutIndex::new(&layout, &fonts, positions, width);
            (Arc::new(layout), index)
        });
        if range.start == 0 && range.end == layout.len {
            return (layout, Some(Arc::new(index)));
        }
        let start = index.boundary(range.start, Bias::Left);
        let end = index.boundary(range.end, Bias::Right);
        let layout = index.slice(&layout, start, end, start.2);
        let positions = index.positions[start.1..end.1]
            .iter()
            .map(|x| x - start.2)
            .collect();
        let fonts = slice_font_runs(&runs, 0, target.start as u32..target.end as u32);
        let index = LayoutIndex::new(&layout, &fonts, positions, end.2 - start.2);
        return (layout, Some(Arc::new(index)));
    }
    let shape = |wrapper: &mut LineWrapper| {
        shape_in_context(shaped.len(), left.len()..left.len() + text.len(), |range| {
            let range_runs = slice_font_runs(&runs, 0, range.start as u32..range.end as u32);
            Arc::new(wrapper.layout_line(&shaped[range], &range_runs))
        })
    };
    let Some(cache) = cache else {
        return (shape(wrapper), None);
    };
    let key = LayoutVariant {
        runs: runs.clone(),
        left: left.len(),
        right: right.len(),
    };
    if let Some(layout) = cache.lock().get(&shaped, &key) {
        return (layout, None);
    }
    let layout = shape(wrapper);
    cache.lock().insert(&shaped, key, layout.clone());
    (layout, None)
}

pub(crate) fn shape_in_context(
    context_len: usize,
    target: Range<usize>,
    mut shape: impl FnMut(Range<usize>) -> Arc<LineLayout>,
) -> Arc<LineLayout> {
    let (layout, (), range) =
        shape_in_context_with(context_len, target, |range| (shape(range), ()));
    slice_layout(layout, range)
}

fn shape_in_context_with<T>(
    context_len: usize,
    target: Range<usize>,
    mut shape: impl FnMut(Range<usize>) -> (Arc<LineLayout>, T),
) -> (Arc<LineLayout>, T, Range<usize>) {
    let mut shaped = 0..context_len;
    loop {
        if shaped == target || target.is_empty() {
            let (layout, extra) = shape(target.clone());
            return (layout, extra, 0..target.len());
        }
        let (layout, extra) = shape(shaped.clone());
        let local = target.start - shaped.start..target.end - shaped.start;
        let starts = partitions_at(&layout, local.start);
        let ends = partitions_at(&layout, local.end);
        if starts && ends {
            return (layout, extra, local);
        }
        let start = if starts { shaped.start } else { target.start };
        let end = if ends { shaped.end } else { target.end };
        shaped = start..end;
    }
}

fn partitions_at(layout: &LineLayout, index: usize) -> bool {
    if index == 0 || index >= layout.len {
        return true;
    }
    let mut first_after = None::<usize>;
    for glyph in layout.runs.iter().flat_map(|run| &run.glyphs) {
        if glyph.index < index {
            if first_after.is_some() {
                return false;
            }
        } else {
            first_after = Some(first_after.map_or(glyph.index, |first| first.min(glyph.index)));
        }
    }
    first_after == Some(index)
}

fn slice_layout(layout: Arc<LineLayout>, range: Range<usize>) -> Arc<LineLayout> {
    if range.start == 0 && range.end == layout.len {
        return layout;
    }
    let start_x = layout.x_for_index(range.start);
    let end_x = layout.x_for_index(range.end);
    Arc::new(LineLayout {
        font_size: layout.font_size,
        width: end_x - start_x,
        ascent: layout.ascent,
        descent: layout.descent,
        runs: layout
            .runs
            .iter()
            .filter_map(|run| {
                let glyphs = run
                    .glyphs
                    .iter()
                    .filter(|glyph| range.contains(&glyph.index))
                    .map(|glyph| ShapedGlyph {
                        index: glyph.index - range.start,
                        position: point(glyph.position.x - start_x, glyph.position.y),
                        ..glyph.clone()
                    })
                    .collect::<Vec<_>>();
                (!glyphs.is_empty()).then(|| ShapedRun {
                    font_id: run.font_id,
                    glyphs,
                })
            })
            .collect(),
        len: range.len(),
    })
}

fn shapes_across(snapshot: &TabSnapshot, row: u32, column: u32) -> bool {
    is_cluster_start_within(snapshot, row, column, SHAPING_CONTEXT_LEN)
}

fn seam_shapes_across(
    snapshot: &TabSnapshot,
    row: u32,
    column: u32,
    previous: Option<u8>,
    next: Option<u8>,
) -> bool {
    match (previous, next) {
        (Some(previous), Some(next)) if previous.is_ascii() && next.is_ascii() => {
            !(previous == b'\r' && next == b'\n')
        }
        _ => shapes_across(snapshot, row, column),
    }
}

fn left_context(snapshot: &TabSnapshot, row: u32, column: u32, next: Option<u8>) -> String {
    let start = shaping_span(snapshot, row, column..column).start;
    let (mut context, _) = left_context_from(snapshot, row, start, start == 0, column);
    if !context.is_empty()
        && !seam_shapes_across(snapshot, row, column, context.bytes().last(), next)
    {
        context.clear();
    }
    context
}

fn left_context_from(
    snapshot: &TabSnapshot,
    row: u32,
    start: u32,
    mut anchored: bool,
    column: u32,
) -> (String, bool) {
    let mut context = String::new();
    if start == column {
        return (context, true);
    }
    for_each_atom(snapshot, row, start..column, |atom| {
        if atom.start < start {
            return true;
        }
        match atom.kind {
            AtomKind::Text { text } if atom.standalone.is_none() => context.push_str(text),
            _ => {
                context.clear();
                anchored = true;
            }
        }
        true
    });
    (context, anchored)
}

fn right_context(snapshot: &TabSnapshot, row: u32, column: u32, previous: Option<u8>) -> String {
    let end = shaping_span(snapshot, row, column..column).end;
    let mut context = SeamContext::right_of(snapshot, row, column, end).text;
    if !context.is_empty()
        && !seam_shapes_across(snapshot, row, column, previous, context.bytes().next())
    {
        context.clear();
    }
    context
}

fn chunk_left_context<'a>(
    cache: &'a mut Option<String>,
    snapshot: &TabSnapshot,
    row: u32,
    start: u32,
    pending: &PendingText,
) -> &'a str {
    if pending.start != start || pending.text.is_empty() {
        return "";
    }
    cache.get_or_insert_with(|| left_context(snapshot, row, start, pending.text.bytes().next()))
}

fn shape_chunk(
    snapshot: &TabSnapshot,
    row: u32,
    request: ChunkRequest<'_>,
    wrapper: &mut LineWrapper,
    cache: Option<&Mutex<LayoutCache>>,
) -> ChunkLayout {
    let start = request.columns.start;
    let scan_end = request.budget.map_or(request.columns.end, |budget| {
        start
            .saturating_add(
                budget
                    .min(SYNC_MEASURE_LEN + RESYNC_RESERVE_LEN)
                    .saturating_sub(2 * (SHAPING_CONTEXT_LEN + 4)),
            )
            .min(request.columns.end)
    });
    let scan_end = clip_row_columns(snapshot, row, scan_end..scan_end)
        .start
        .max(start);
    let mut layout = ChunkLayout {
        end: request.columns.end,
        ..ChunkLayout::default()
    };
    let mut pending = PendingText::default();
    let mut cutter = ChunkCutter::new(start, scan_end);
    let mut left_context_cache = None;
    let mut standalone_widths = HashMap::default();
    for_each_atom(snapshot, row, start..scan_end, |atom| {
        if atom.start < start {
            return true;
        }
        if request.cut && !cutter.accepts(&atom, snapshot, row, request.runs, wrapper) {
            layout.end = atom.start;
            return false;
        }
        if !atom.is_whitespace {
            layout.content_end = Some(atom.start + atom.len);
        }
        let width = match atom.kind {
            AtomKind::Text { text } => match atom.standalone {
                Some(character) => standalone_width(
                    character,
                    atom.start..atom.start + atom.len,
                    request.runs,
                    wrapper,
                    &mut standalone_widths,
                ),
                None => {
                    if pending.text.is_empty() {
                        pending.start = atom.start;
                    }
                    pending.atoms.push((atom.start, pending.text.len()));
                    pending.text.push_str(text);
                    return true;
                }
            },
            AtomKind::Renderer {
                measured_width: Some(width),
                ..
            } => f64::from(width),
            AtomKind::Renderer {
                measured_width: None,
                texts,
            } => texts
                .iter()
                .flat_map(|text| text.chars())
                .map(|character| f64::from(wrapper.width_for_char(character)))
                .sum(),
        };
        let left = chunk_left_context(&mut left_context_cache, snapshot, row, start, &pending);
        layout.push_text(&mut pending, &request, (left, ""), wrapper, cache);
        layout.atoms.push((atom.start, layout.width));
        layout.width += width;
        true
    });
    if cutter.exhausted || layout.end > scan_end {
        return ChunkLayout {
            end: start,
            ..ChunkLayout::default()
        };
    }
    let right = match pending.text.bytes().last() {
        Some(previous) => right_context(snapshot, row, layout.end, Some(previous)),
        None => String::new(),
    };
    let left = chunk_left_context(&mut left_context_cache, snapshot, row, start, &pending);
    layout.push_text(&mut pending, &request, (left, &right), wrapper, cache);
    layout
}

fn standalone_width(
    character: char,
    columns: Range<u32>,
    runs: RunsView<'_>,
    wrapper: &mut LineWrapper,
    widths: &mut HashMap<(FontId, char), ScrollPixelOffset>,
) -> ScrollPixelOffset {
    let shown = replacement(character).unwrap_or(character);
    let base = wrapper.font_id();
    let font_id = piece_runs(runs, columns, base)
        .first()
        .map_or(base, |run| run.font_id);
    if font_id == base {
        return f64::from(wrapper.width_for_char(shown));
    }
    *widths.entry((font_id, shown)).or_insert_with(|| {
        let mut buffer = [0; 4];
        let text = shown.encode_utf8(&mut buffer);
        f64::from(
            wrapper
                .layout_line(
                    text,
                    &[FontRun {
                        len: text.len(),
                        font_id,
                    }],
                )
                .width,
        )
    })
}

fn single_invisible(text: &str) -> Option<char> {
    let mut chars = text.chars();
    match (chars.next(), chars.next()) {
        (Some(character), None) if is_invisible(character) => Some(character),
        _ => None,
    }
}

fn standalone_atom(
    snapshot: &TabSnapshot,
    row: u32,
    text: &str,
    start: u32,
    starts: Option<AtomBoundary>,
    ends: Option<AtomBoundary>,
) -> Option<char> {
    let character = single_invisible(text)?;
    let is_cluster_boundary = |boundary: Option<AtomBoundary>, column: u32| match boundary {
        Some(boundary) => boundary == AtomBoundary::Cluster,
        None => is_cluster_start(snapshot, row, column),
    };
    (is_cluster_boundary(starts, start) && is_cluster_boundary(ends, start + text.len() as u32))
        .then_some(character)
}

struct RawAtom<'a> {
    start: u32,
    len: u32,
    kind: AtomKind<'a>,
    is_whitespace: bool,
    standalone: Option<char>,
    starts: Option<AtomBoundary>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum AtomBoundary {
    Cluster,
    Cap,
}

enum AtomKind<'a> {
    Text {
        text: &'a str,
    },
    Renderer {
        measured_width: Option<Pixels>,
        texts: &'a [&'a str],
    },
}

struct PendingRenderer<'a> {
    id: ChunkRendererId,
    start: u32,
    len: u32,
    measured_width: Option<Pixels>,
    texts: SmallVec<[&'a str; 2]>,
}

fn atom_boundary(cluster: &str, offset: usize, character: char) -> Option<AtomBoundary> {
    if offset == 0 {
        None
    } else if is_cluster_boundary(cluster, offset) {
        Some(AtomBoundary::Cluster)
    } else if cluster.len() > MAX_ATOM_LEN as usize && !is_regional_indicator(character) {
        Some(AtomBoundary::Cap)
    } else {
        None
    }
}

fn budgeted_window(pieces: &[Piece], budget: u32) -> RowWindow {
    let mut kept = Vec::new();
    let mut elided = None::<Range<u32>>;
    let mut remaining = budget;
    for piece in pieces.iter().filter(|piece| piece.density > 0.) {
        if let Some(elided) = &mut elided {
            elided.end = piece.columns.end;
        } else if piece.len() <= remaining {
            remaining = remaining.saturating_sub(piece.len());
            kept.push(piece.columns.clone());
        } else {
            elided = Some(piece.columns.clone());
        }
    }
    kept.into_iter().collect::<RowWindow>().with_elided(elided)
}

fn for_each_atom(
    snapshot: &TabSnapshot,
    row: u32,
    columns: Range<u32>,
    mut f: impl FnMut(RawAtom<'_>) -> bool,
) {
    if columns.is_empty() {
        return;
    }
    let (start, chunks) = row_chunks(snapshot, row, columns);
    let mut cluster = PendingCluster {
        text: String::new(),
        start,
        starts: None,
        is_whitespace: true,
    };
    let mut renderer = None::<PendingRenderer<'_>>;
    let mut column = start;
    for chunk in chunks {
        if let Some(chunk_renderer) = &chunk.renderer {
            match &mut renderer {
                Some(pending) if pending.id == chunk_renderer.id => {
                    pending.len += chunk.text.len() as u32;
                    if pending.measured_width.is_none() {
                        pending.texts.push(chunk.text);
                    }
                }
                _ => {
                    if !cluster.emit(snapshot, row, Some(AtomBoundary::Cluster), &mut f)
                        || !emit_renderer(renderer.take(), &mut f)
                    {
                        return;
                    }
                    renderer = Some(PendingRenderer {
                        id: chunk_renderer.id,
                        start: column,
                        len: chunk.text.len() as u32,
                        measured_width: chunk_renderer.measured_width,
                        texts: if chunk_renderer.measured_width.is_none() {
                            smallvec![chunk.text]
                        } else {
                            SmallVec::new()
                        },
                    });
                }
            }
            column += chunk.text.len() as u32;
            cluster.start = column;
            cluster.starts = Some(AtomBoundary::Cluster);
            continue;
        }
        if !emit_renderer(renderer.take(), &mut f) {
            return;
        }
        for character in chunk.text.chars() {
            let cluster_len = cluster.text.len();
            cluster.text.push(character);
            if let Some(boundary) = atom_boundary(&cluster.text, cluster_len, character) {
                cluster.text.truncate(cluster_len);
                if !cluster.emit(snapshot, row, Some(boundary), &mut f) {
                    return;
                }
                cluster.text.push(character);
                cluster.start = column;
                cluster.starts = Some(boundary);
            }
            cluster.is_whitespace &=
                !chunk.is_inlay && character.is_whitespace() && !is_invisible(character);
            column += character.len_utf8() as u32;
        }
    }
    if emit_renderer(renderer.take(), &mut f) {
        cluster.emit(snapshot, row, None, &mut f);
    }
}

struct PendingCluster {
    text: String,
    start: u32,
    starts: Option<AtomBoundary>,
    is_whitespace: bool,
}

impl PendingCluster {
    fn emit(
        &mut self,
        snapshot: &TabSnapshot,
        row: u32,
        ends: Option<AtomBoundary>,
        f: &mut impl FnMut(RawAtom<'_>) -> bool,
    ) -> bool {
        if self.text.is_empty() {
            return true;
        }
        let is_whitespace = mem::replace(&mut self.is_whitespace, true);
        let standalone = standalone_atom(snapshot, row, &self.text, self.start, self.starts, ends);
        let keep_going = f(RawAtom {
            start: self.start,
            len: self.text.len() as u32,
            kind: AtomKind::Text { text: &self.text },
            is_whitespace,
            standalone,
            starts: self.starts,
        });
        self.text.clear();
        keep_going
    }
}

fn emit_renderer(
    renderer: Option<PendingRenderer<'_>>,
    f: &mut impl FnMut(RawAtom<'_>) -> bool,
) -> bool {
    let Some(renderer) = renderer else {
        return true;
    };
    f(RawAtom {
        start: renderer.start,
        len: renderer.len,
        kind: AtomKind::Renderer {
            measured_width: renderer.measured_width,
            texts: &renderer.texts,
        },
        is_whitespace: false,
        standalone: None,
        starts: Some(AtomBoundary::Cluster),
    })
}

fn starts_hard_atom(snapshot: &TabSnapshot, row: u32, column: u32) -> bool {
    let mut hard = false;
    for_each_atom(snapshot, row, column..column.saturating_add(1), |atom| {
        if atom.start < column {
            return true;
        }
        hard = atom.standalone.is_some() || matches!(atom.kind, AtomKind::Renderer { .. });
        false
    });
    hard
}

fn is_regional_indicator(character: char) -> bool {
    ('\u{1f1e6}'..='\u{1f1ff}').contains(&character)
}

pub(super) struct StandaloneInvisibles {
    atom: String,
    first: AtomBoundary,
    last: AtomBoundary,
}

impl Default for StandaloneInvisibles {
    fn default() -> Self {
        Self {
            atom: String::new(),
            first: AtomBoundary::Cluster,
            last: AtomBoundary::Cluster,
        }
    }
}

impl StandaloneInvisibles {
    pub(super) fn for_columns(snapshot: &TabSnapshot, row: u32, columns: &Range<u32>) -> Self {
        let boundary = |column: u32| {
            if is_cluster_start(snapshot, row, column) {
                AtomBoundary::Cluster
            } else {
                AtomBoundary::Cap
            }
        };
        Self {
            atom: String::new(),
            first: boundary(columns.start),
            last: boundary(columns.end),
        }
    }

    pub(super) fn reset(&mut self) {
        self.atom.clear();
        self.first = AtomBoundary::Cluster;
    }

    pub(super) fn find(
        &mut self,
        text: &str,
        next: Option<char>,
        is_last: bool,
    ) -> SmallVec<[Range<usize>; 1]> {
        let mut found = SmallVec::new();
        let mut candidate = None::<Range<usize>>;
        let mut confirm = |candidate: &mut Option<Range<usize>>, boundary: Option<AtomBoundary>| {
            if let Some(range) = candidate.take()
                && boundary == Some(AtomBoundary::Cluster)
            {
                found.push(range);
            }
        };
        let ascii_suffix = text.bytes().rev().take_while(u8::is_ascii).count();
        let general_end = (text.len() - ascii_suffix + 1).min(text.len());
        for (offset, character) in text[..general_end].char_indices() {
            let boundary = self.push(character);
            confirm(&mut candidate, boundary);
            if boundary == Some(AtomBoundary::Cluster) && is_invisible(character) {
                candidate = Some(offset..offset + character.len_utf8());
            }
        }
        let bytes = text.as_bytes();
        let mut last_atom_start = None;
        for offset in general_end..text.len() {
            let boundary = (!(bytes[offset - 1] == b'\r' && bytes[offset] == b'\n'))
                .then_some(AtomBoundary::Cluster);
            confirm(&mut candidate, boundary);
            if boundary.is_some() {
                last_atom_start = Some(offset);
                if is_invisible(char::from(bytes[offset])) {
                    candidate = Some(offset..offset + 1);
                }
            }
        }
        match last_atom_start {
            Some(start) => {
                self.atom.clear();
                self.atom.push_str(&text[start..]);
            }
            None => self.atom.push_str(&text[general_end..]),
        }
        if candidate.is_some() {
            let boundary = match next {
                Some(next) => self.would_start_atom(next),
                None if is_last => Some(self.last),
                None => Some(AtomBoundary::Cluster),
            };
            confirm(&mut candidate, boundary);
        }
        found
    }

    fn push(&mut self, character: char) -> Option<AtomBoundary> {
        let len = self.atom.len();
        self.atom.push(character);
        if len == 0 {
            return Some(self.first);
        }
        let boundary = atom_boundary(&self.atom, len, character);
        if boundary.is_some() {
            self.atom.clear();
            self.atom.push(character);
        }
        boundary
    }

    fn would_start_atom(&mut self, character: char) -> Option<AtomBoundary> {
        let len = self.atom.len();
        if len == 0 {
            return Some(self.first);
        }
        self.atom.push(character);
        let boundary = atom_boundary(&self.atom, len, character);
        self.atom.truncate(len);
        boundary
    }
}

fn is_cluster_start(snapshot: &TabSnapshot, row: u32, column: u32) -> bool {
    is_cluster_start_within(snapshot, row, column, GRAPHEME_CONTEXT_LEN)
}

fn is_cluster_start_within(snapshot: &TabSnapshot, row: u32, column: u32, context: u32) -> bool {
    let len = snapshot.line_len(row);
    if column == 0 || column >= len {
        return true;
    }
    if let Some(floor) = grapheme_floor_within(snapshot, row, len, column, 0, context)
        .or_else(|| regional_indicator_floor(snapshot, row, column, 0))
    {
        return floor == column;
    }
    let (start, chunks) = row_chunks(
        snapshot,
        row,
        column.saturating_sub(4)..column.saturating_add(4).min(len),
    );
    let mut text = String::new();
    let mut position = start;
    for chunk in chunks {
        let end = position + chunk.text.len() as u32;
        if chunk.renderer.is_some() && position <= column && column <= end {
            return true;
        }
        text.push_str(chunk.text);
        position = end;
    }
    column
        .checked_sub(start)
        .is_none_or(|offset| is_cluster_boundary(&text, offset as usize))
}

fn is_cluster_boundary(text: &str, offset: usize) -> bool {
    let bytes = text.as_bytes();
    if let (Some(&previous), Some(&next)) = (bytes.get(offset.wrapping_sub(1)), bytes.get(offset))
        && previous.is_ascii()
        && next.is_ascii()
    {
        return !(previous == b'\r' && next == b'\n');
    }
    GraphemeCursor::new(offset, text.len(), true)
        .is_boundary(text, 0)
        .unwrap_or(true)
}

pub(super) fn clip_row_columns(
    snapshot: &TabSnapshot,
    row: u32,
    columns: Range<u32>,
) -> Range<u32> {
    let len = snapshot.line_len(row);
    let clip = |column: u32, bias: Bias, fallback: u32| {
        let column = column.min(len);
        let (_, _, to_next_tab_stop) =
            snapshot.tab_point_to_fold_point(TabPoint::new(row, column), Bias::Left);
        if to_next_tab_stop > 0 {
            column
        } else {
            clip_column(snapshot, row, column, bias).unwrap_or(fallback)
        }
    };
    let start = clip(columns.start, Bias::Left, 0);
    let end = clip(columns.end, Bias::Right, len);
    start..end.max(start)
}

fn row_chunks(snapshot: &TabSnapshot, row: u32, columns: Range<u32>) -> (u32, TabChunks<'_>) {
    let columns = clip_row_columns(snapshot, row, columns);
    let chunks = snapshot.chunks(
        TabPoint::new(row, columns.start)..TabPoint::new(row, columns.end),
        LanguageAwareStyling {
            tree_sitter: false,
            diagnostics: false,
        },
        Highlights::default(),
    );
    (columns.start, chunks)
}

fn clip_column(snapshot: &TabSnapshot, row: u32, column: u32, bias: Bias) -> Option<u32> {
    let fold_point = snapshot
        .tab_point_to_fold_point(TabPoint::new(row, column), bias)
        .0;
    let clipped = snapshot.fold_point_to_tab_point(
        snapshot
            .fold_snapshot
            .clip_point_to_char_boundary(fold_point, bias),
    );
    (clipped.row() == row).then_some(clipped.column())
}

struct ChunkWalk<'a> {
    snapshot: &'a TabSnapshot,
    row: u32,
    columns: Range<u32>,
    max_len: u32,
    budget: Option<u32>,
    canonical: bool,
    end_is_canonical: bool,
    runs: RunsView<'a>,
    base: FontId,
    cache: Option<&'a Mutex<LayoutCache>>,
}

fn measure_chunks(
    walk: ChunkWalk<'_>,
    wrapper: &mut LineWrapper,
    builder: &mut ChunkBuilder,
) -> u32 {
    let mut chunk_start = walk.columns.start;
    while chunk_start < walk.columns.end && chunk_start - walk.columns.start < walk.max_len {
        let layout = shape_chunk(
            walk.snapshot,
            walk.row,
            ChunkRequest {
                columns: chunk_start..walk.columns.end,
                cut: true,
                runs: walk.runs,
                budget: walk.budget,
            },
            wrapper,
            walk.cache,
        );
        let end = layout.end;
        if end == chunk_start {
            break;
        }
        let canonical = walk.canonical && (end < walk.columns.end || walk.end_is_canonical);
        let chars = char_count(walk.snapshot, walk.row, chunk_start..layout.end);
        let span = shaping_span(walk.snapshot, walk.row, chunk_start..layout.end);
        let runs = ChunkRuns::new(
            piece_runs(walk.runs, span.clone(), walk.base),
            span,
            chunk_start,
            walk.base,
        );
        builder.push(layout.chunk(chunk_start, canonical, chars, runs));
        chunk_start = end;
    }
    chunk_start
}

fn char_count(snapshot: &TabSnapshot, row: u32, columns: Range<u32>) -> u32 {
    if columns.is_empty() {
        return 0;
    }
    snapshot
        .text_summary_for_range(TabPoint::new(row, columns.start)..TabPoint::new(row, columns.end))
        .first_line_chars
        .min(columns.end - columns.start)
}

struct ChunkCutter {
    start: u32,
    hash: u64,
    whitespace_end: u32,
    after_whitespace: bool,
    deferred_cut: Option<u32>,
    retry_at: u32,
    limit: u32,
    exhausted: bool,
}

enum Seam {
    At(u32),
    Retry(u32),
}

struct SeamContext {
    text: String,
    atom_starts: Vec<usize>,
    terminal: bool,
}

impl ChunkCutter {
    fn new(start: u32, limit: u32) -> Self {
        Self {
            start,
            hash: 0,
            whitespace_end: start,
            after_whitespace: false,
            deferred_cut: None,
            retry_at: start,
            limit,
            exhausted: false,
        }
    }

    fn accepts(
        &mut self,
        atom: &RawAtom<'_>,
        snapshot: &TabSnapshot,
        row: u32,
        runs: RunsView<'_>,
        wrapper: &mut LineWrapper,
    ) -> bool {
        match self.deferred_cut {
            Some(cut) if atom.start >= cut => return false,
            None if atom.starts != Some(AtomBoundary::Cap) && self.wants_cut(atom) => {
                let forced = atom.start - self.start >= FORCED_CHUNK_LEN;
                let Some(seam) = find_seam(
                    snapshot, row, self.start, atom.start, forced, self.limit, runs, wrapper,
                ) else {
                    self.exhausted = true;
                    return false;
                };
                match seam {
                    Seam::At(cut) if cut <= atom.start => return false,
                    Seam::At(cut) => self.deferred_cut = Some(cut),
                    Seam::Retry(column) => self.retry_at = column,
                }
            }
            _ => {}
        }
        let mut feed_bytes = |text: &str| {
            for byte in text.bytes() {
                self.hash = (self.hash << 1).wrapping_add(GEAR[byte as usize]);
            }
        };
        match &atom.kind {
            AtomKind::Text { text } => feed_bytes(text),
            AtomKind::Renderer { texts, .. } => texts.iter().for_each(|text| feed_bytes(text)),
        }
        if atom.is_whitespace {
            self.whitespace_end = atom.start + atom.len;
        }
        self.after_whitespace = atom.is_whitespace;
        true
    }

    fn wants_cut(&self, atom: &RawAtom<'_>) -> bool {
        if atom.start < self.retry_at {
            return false;
        }
        let offset = atom.start - self.start;
        if offset >= MAX_CHUNK_LEN {
            return true;
        }
        offset >= MIN_CHUNK_LEN
            && if self.after_whitespace {
                self.hash & WHITESPACE_CUT_MASK == 0
            } else {
                atom.start - self.whitespace_end >= DENSE_RUN_LEN && self.hash & DENSE_CUT_MASK == 0
            }
    }
}

const fn gear_table() -> [u64; 256] {
    let mut table = [0; 256];
    let mut state = 0x9e37_79b9_7f4a_7c15_u64;
    let mut index = 0;
    while index < table.len() {
        state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut value = state;
        value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        table[index] = value ^ (value >> 31);
        index += 1;
    }
    table
}

fn next_cut(
    snapshot: &TabSnapshot,
    row: u32,
    start: u32,
    limit: u32,
    scan_end: u32,
    runs: RunsView<'_>,
    wrapper: &mut LineWrapper,
) -> Option<u32> {
    let scan_end = scan_end.min(limit);
    let scan_end = clip_row_columns(snapshot, row, scan_end..scan_end)
        .start
        .max(start);
    let mut cutter = ChunkCutter::new(start, scan_end);
    let mut end = (scan_end == limit).then_some(limit);
    for_each_atom(snapshot, row, start..scan_end, |atom| {
        if atom.start < start {
            return true;
        }
        if !cutter.accepts(&atom, snapshot, row, runs, wrapper) {
            end = (!cutter.exhausted).then_some(atom.start);
            return false;
        }
        true
    });
    end
}

fn find_seam(
    snapshot: &TabSnapshot,
    row: u32,
    chunk_start: u32,
    column: u32,
    forced: bool,
    limit: u32,
    runs: RunsView<'_>,
    wrapper: &mut LineWrapper,
) -> Option<Seam> {
    let lookahead = if forced {
        (column - chunk_start)
            .saturating_sub(FORCED_CHUNK_LEN)
            .max(SHAPING_CONTEXT_LEN)
    } else {
        SHAPING_CONTEXT_LEN
    };
    let end = clip_row_columns(snapshot, row, column..column.saturating_add(lookahead)).end;
    if end > limit {
        return None;
    }
    let right = SeamContext::right_of(snapshot, row, column, end);
    let Some(next) = right.text.bytes().next() else {
        return Some(Seam::At(column));
    };
    let nearby_start = shaping_span(snapshot, row, column..column).start;
    let (nearby, nearby_anchored) =
        left_context_from(snapshot, row, nearby_start, nearby_start == 0, column);
    if nearby.is_empty() && nearby_anchored {
        return Some(Seam::At(column));
    }
    if !seam_shapes_across(snapshot, row, column, nearby.bytes().last(), Some(next)) {
        return Some(Seam::Retry(column + 1));
    }
    let (left, anchored) = if forced {
        left_context_from(snapshot, row, chunk_start, true, column)
    } else {
        (nearby, nearby_anchored)
    };
    if left.is_empty() {
        return Some(if anchored {
            Seam::At(column)
        } else {
            Seam::Retry(column + 1)
        });
    }
    let probe_count = if anchored { 1 } else { PHASE_PROBES };
    let base = wrapper.font_id();
    let left_start = column - left.len() as u32;
    let right_end = column + right.text.len() as u32;
    let mut candidates = right.atom_starts;
    if right.terminal {
        candidates.push(right.text.len());
    }
    for probe_start in (0..left.len())
        .filter(|offset| left.is_char_boundary(*offset))
        .take(probe_count)
    {
        let text = [&left[probe_start..], right.text.as_str()].concat();
        let probe_runs = piece_runs(runs, left_start + probe_start as u32..right_end, base);
        let layout = wrapper.layout_line(&text, &checked_runs(&text, probe_runs, base));
        let seam = left.len() - probe_start;
        let partitions = glyph_partitions(&layout);
        candidates.retain(|offset| {
            *offset == right.text.len()
                || partitions
                    .binary_search_by_key(&(seam + offset), |entry| entry.0)
                    .is_ok()
        });
        if candidates.is_empty() {
            break;
        }
    }
    Some(match candidates.first() {
        Some(offset) => Seam::At(column + *offset as u32),
        None => Seam::Retry(right_end),
    })
}

impl SeamContext {
    fn right_of(snapshot: &TabSnapshot, row: u32, column: u32, end: u32) -> Self {
        let end = clip_row_columns(snapshot, row, column..end).end;
        let mut context = Self {
            text: String::new(),
            atom_starts: Vec::new(),
            terminal: false,
        };
        for_each_atom(snapshot, row, column..end, |atom| {
            if atom.start < column {
                return true;
            }
            match atom.kind {
                AtomKind::Text { text } if atom.standalone.is_none() => {
                    if atom.starts != Some(AtomBoundary::Cap) {
                        context.atom_starts.push(context.text.len());
                    }
                    context.text.push_str(text);
                    true
                }
                _ => {
                    context.terminal = true;
                    false
                }
            }
        });
        context.terminal |= column + context.text.len() as u32 >= snapshot.line_len(row);
        context
    }
}

fn glyph_partitions(layout: &LineLayout) -> Vec<(usize, usize)> {
    let indices = layout
        .runs
        .iter()
        .flat_map(|run| &run.glyphs)
        .map(|glyph| glyph.index)
        .collect::<Vec<_>>();
    let mut suffix_min = vec![usize::MAX; indices.len() + 1];
    for position in (0..indices.len()).rev() {
        suffix_min[position] = suffix_min[position + 1].min(indices[position]);
    }
    let mut partitions = Vec::new();
    let mut prefix_max = None::<usize>;
    for (position, index) in indices.iter().enumerate() {
        if prefix_max.is_none_or(|max| max < suffix_min[position]) {
            partitions.push((suffix_min[position], position));
        }
        prefix_max = Some(prefix_max.map_or(*index, |max| max.max(*index)));
    }
    partitions
}

fn last_cut_before(
    snapshot: &TabSnapshot,
    row: u32,
    start: u32,
    target: u32,
    limit: u32,
    runs: RunsView<'_>,
    budget: u32,
    wrapper: &mut LineWrapper,
) -> u32 {
    let mut cut = start;
    while cut < target {
        let scan_end = cut.saturating_add(budget.min(SYNC_MEASURE_LEN + RESYNC_RESERVE_LEN));
        let Some(next) = next_cut(snapshot, row, cut, limit, scan_end, runs, wrapper) else {
            break;
        };
        if next > target || next >= limit || next <= cut {
            break;
        }
        cut = next;
    }
    cut
}

fn shaping_span(snapshot: &TabSnapshot, row: u32, columns: Range<u32>) -> Range<u32> {
    clip_row_columns(
        snapshot,
        row,
        columns.start.saturating_sub(SHAPING_CONTEXT_LEN)
            ..columns.end.saturating_add(SHAPING_CONTEXT_LEN),
    )
}

fn piece_runs(runs: RunsView<'_>, columns: Range<u32>, base: FontId) -> SmallVec<[FontRun; 1]> {
    let mut sliced = FontRunsBuilder::default();
    let mut cursor = columns.start;
    let first = runs.0.partition_point(|known| known.end <= columns.start);
    for known in &runs.0[first..] {
        if known.start >= columns.end {
            break;
        }
        let start = known.start.max(cursor);
        let end = known.end.min(columns.end);
        if start >= end {
            continue;
        }
        sliced.push_entry([StyledRun::from(FontRun {
            len: (start - cursor) as usize,
            font_id: base,
        })]);
        sliced.push_entry(slice_styled_runs(&known.runs, known.start, start..end));
        cursor = end;
    }
    sliced.push_entry([StyledRun::from(FontRun {
        len: columns.end.saturating_sub(cursor) as usize,
        font_id: base,
    })]);
    sliced.font_runs()
}

fn run_differences(old: &[FontRun], new: &[FontRun], offset: u32) -> Vec<Range<u32>> {
    let mut differences = Vec::<Range<u32>>::new();
    let mut mark = |range: Range<usize>| {
        let range = offset + range.start as u32..offset + range.end as u32;
        match differences.last_mut() {
            Some(last) if range.start <= last.end => last.end = last.end.max(range.end),
            _ => differences.push(range),
        }
    };
    let (mut old_index, mut new_index) = (0, 0);
    let mut old_end = old.first().map_or(0, |run| run.len);
    let mut new_end = new.first().map_or(0, |run| run.len);
    let mut position = 0;
    while let (Some(old_run), Some(new_run)) = (old.get(old_index), new.get(new_index)) {
        let segment_end = old_end.min(new_end);
        if old_run.font_id != new_run.font_id {
            mark(position..segment_end);
        }
        if old_end != new_end {
            mark(segment_end..segment_end);
        }
        position = segment_end;
        if old_end == segment_end {
            old_index += 1;
            old_end += old.get(old_index).map_or(0, |run| run.len);
        }
        if new_end == segment_end {
            new_index += 1;
            new_end += new.get(new_index).map_or(0, |run| run.len);
        }
    }
    differences
}

fn slice_styled_runs(
    runs: &[StyledRun],
    start: u32,
    columns: Range<u32>,
) -> SmallVec<[StyledRun; 1]> {
    let mut sliced = SmallVec::new();
    let mut run_start = start;
    for styled in runs {
        let run_end = run_start + styled.run.len as u32;
        let sliced_start = run_start.max(columns.start);
        let sliced_end = run_end.min(columns.end);
        if sliced_start < sliced_end {
            sliced.push(StyledRun {
                run: FontRun {
                    len: (sliced_end - sliced_start) as usize,
                    font_id: styled.run.font_id,
                },
                decoration: styled.decoration,
            });
        }
        run_start = run_end;
        if run_start >= columns.end {
            break;
        }
    }
    sliced
}

fn slice_font_runs(runs: &[FontRun], start: u32, columns: Range<u32>) -> SmallVec<[FontRun; 1]> {
    let mut sliced = SmallVec::new();
    let mut run_start = start;
    for run in runs {
        let run_end = run_start + run.len as u32;
        let sliced_start = run_start.max(columns.start);
        let sliced_end = run_end.min(columns.end);
        if sliced_start < sliced_end {
            sliced.push(FontRun {
                len: (sliced_end - sliced_start) as usize,
                font_id: run.font_id,
            });
        }
        run_start = run_end;
        if run_start >= columns.end {
            break;
        }
    }
    sliced
}

fn checked_runs(text: &str, runs: SmallVec<[FontRun; 1]>, base: FontId) -> SmallVec<[FontRun; 1]> {
    let mut end = 0;
    let partitions_text = runs.iter().all(|run| {
        end += run.len;
        text.is_char_boundary(end)
    }) && end == text.len();
    if partitions_text {
        return runs;
    }
    debug_panic!(
        "font runs {runs:?} do not split {} bytes of text at char boundaries",
        text.len()
    );
    smallvec![FontRun {
        len: text.len(),
        font_id: base,
    }]
}

fn grapheme_floor(
    snapshot: &TabSnapshot,
    row: u32,
    len: u32,
    column: u32,
    floor: u32,
) -> GraphemeFloor {
    grapheme_floor_within(snapshot, row, len, column, floor, GRAPHEME_CONTEXT_LEN)
        .or_else(|| regional_indicator_floor(snapshot, row, column, floor))
        .map_or(GraphemeFloor::OversizedCluster, GraphemeFloor::Boundary)
}

fn regional_indicator_floor(
    snapshot: &TabSnapshot,
    row: u32,
    column: u32,
    floor: u32,
) -> Option<u32> {
    let column = clip_column(snapshot, row, column, Bias::Left)?;
    if column <= floor {
        return Some(floor);
    }
    let (around_start, around) = plain_row_text(
        snapshot,
        row,
        column.saturating_sub(4)..column.saturating_add(4),
    );
    let local = column.checked_sub(around_start)? as usize;
    let around = around.as_bytes();
    if !(local >= 4
        && around
            .get(local - 4..local + 4)
            .is_some_and(|bytes| is_regional_indicator_pair(bytes)))
    {
        return None;
    }
    let end = snapshot
        .tab_point_to_fold_point(TabPoint::new(row, column), Bias::Left)
        .0;
    let preceding = snapshot
        .fold_snapshot
        .text_summary_for_range(FoldPoint::new(end.row(), 0)..end);
    Some(column - 4 * u32::from(preceding.regional_indicators.odd))
}

fn plain_row_text(snapshot: &TabSnapshot, row: u32, columns: Range<u32>) -> (u32, String) {
    let (mut text_start, chunks) = row_chunks(snapshot, row, columns);
    let mut text = String::new();
    let mut position = text_start;
    for chunk in chunks {
        position += chunk.text.len() as u32;
        if chunk.renderer.is_some() {
            text.clear();
            text_start = position;
        } else {
            text.push_str(chunk.text);
        }
    }
    (text_start, text)
}

fn is_regional_indicator_bytes(bytes: &[u8]) -> bool {
    matches!(bytes, [0xF0, 0x9F, 0x87, 0xA6..=0xBF])
}

fn is_regional_indicator_pair(bytes: &[u8]) -> bool {
    bytes.len() == 8
        && is_regional_indicator_bytes(&bytes[..4])
        && is_regional_indicator_bytes(&bytes[4..])
}

fn grapheme_floor_within(
    snapshot: &TabSnapshot,
    row: u32,
    len: u32,
    column: u32,
    floor: u32,
    mut context: u32,
) -> Option<u32> {
    let floor =
        clip_column(snapshot, row, floor, Bias::Left).map_or(floor, |aligned| aligned.min(floor));
    if column <= floor {
        return Some(floor);
    }
    if column >= len {
        return Some(len);
    }
    let end = clip_column(snapshot, row, column + 1, Bias::Right).map_or(len, |end| end.min(len));
    loop {
        let start = clip_column(
            snapshot,
            row,
            column.saturating_sub(context).max(floor),
            Bias::Left,
        )
        .map_or(floor, |start| start.clamp(floor, column));
        let (start, chunks) = row_chunks(snapshot, row, start..end);
        let floor = floor.min(start);
        let mut text = String::new();
        let mut text_start = start;
        let mut position = start;
        let mut renderer_run = None;
        for chunk in chunks {
            let chunk_end = position + chunk.text.len() as u32;
            if let Some(renderer) = &chunk.renderer {
                let renderer_start = match renderer_run {
                    Some((id, renderer_start)) if id == renderer.id => renderer_start,
                    _ => position,
                };
                renderer_run = Some((renderer.id, renderer_start));
                if renderer_start == column {
                    return Some(column);
                }
                if position > column {
                    break;
                }
                if column < chunk_end {
                    return Some(renderer_start.max(floor));
                }
                text.clear();
                text_start = chunk_end;
            } else {
                renderer_run = None;
                text.push_str(chunk.text);
            }
            position = chunk_end;
        }
        if text_start >= column {
            return Some(column);
        }
        let base = if text_start > start {
            text_start
        } else {
            floor
        };
        let chunk_start = (text_start - base) as usize;
        let offset = chunk_start + text.floor_char_boundary((column - text_start) as usize);
        let mut cursor = GraphemeCursor::new(offset, (len - base) as usize, true);
        let boundary = match cursor.is_boundary(&text, chunk_start) {
            Ok(true) => Ok(Some(offset)),
            Ok(false) => cursor.prev_boundary(&text, chunk_start),
            Err(error) => Err(error),
        };
        match boundary {
            Ok(boundary) => return Some(base + boundary.unwrap_or(0) as u32),
            Err(_) if start > floor && context < MAX_GRAPHEME_CONTEXT_LEN => {
                context = context.saturating_mul(2)
            }
            Err(_) => return None,
        }
    }
}

fn shift_row(row: u32, delta: i64) -> u32 {
    (i64::from(row) + delta).clamp(0, i64::from(u32::MAX)) as u32
}

fn shift_rows(rows: Range<u32>, delta: i64) -> Range<u32> {
    shift_row(rows.start, delta)..shift_row(rows.end, delta)
}

fn shift_column(column: u32, delta: i64) -> u32 {
    (i64::from(column) + delta).max(0) as u32
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        EditorStyle,
        display_map::{
            ChunkRendererId, ChunkReplacement, Crease, DisplayMap, DisplayPoint, DisplayRow,
            DisplaySnapshot, FoldPlaceholder, HighlightKey, RendererWidthsKey, fold_map::FoldMap,
            inlay_map::InlayMap, tab_map::TabMap,
        },
        inlays::Inlay,
        test::{ShapingTextSystem, test_font},
    };
    use collections::HashSet;
    use gpui::{
        AppContext as _, BorrowAppContext as _, Entity, GlyphId, HighlightStyle, SharedString,
        TestAppContext, rgba,
    };
    use language::{Buffer, Diagnostic, DiagnosticEntry, DiagnosticSet, Point};
    use lsp::LanguageServerId;
    use multi_buffer::{MultiBuffer, MultiBufferOffset};
    use project::{InlayId, project_settings::DiagnosticSeverity};
    use rand::prelude::*;
    use settings::SettingsStore;
    use std::{num::NonZeroU32, ops::RangeInclusive, ptr, slice};
    use text::PointUtf16;
    use theme::LoadThemes;
    use unicode_segmentation::UnicodeSegmentation as _;

    const FONT_SIZE: gpui::Pixels = px(14.);
    const POSITION_TOLERANCE: ScrollPixelOffset = 0.01;

    #[gpui::test(iterations = 10)]
    async fn test_random_long_row_edits(cx: &mut TestAppContext, mut rng: StdRng) {
        cx.background_executor.set_block_on_ticks(0..=50);
        let text = (0..4)
            .map(|_| random_text(&mut rng, 200..1_200))
            .collect::<Vec<_>>()
            .join("\n");
        let (buffer, map) = build_display_map(&text, 4, cx);
        let mut wrapper = cx.update(|cx| cx.text_system().line_wrapper(test_font(), FONT_SIZE));

        for _ in 0..10 {
            for _ in 0..rng.random_range(1..=3) {
                randomly_mutate(&buffer, &map, &mut rng, cx);
                let snapshot = map.update(cx, |map, cx| map.snapshot(cx));
                assert_round_trips(&snapshot);
            }
            cx.run_until_parked();
            let snapshot = map.update(cx, |map, cx| map.snapshot(cx));
            assert_matches_char_widths(&snapshot, &mut wrapper);
            assert_matches_fresh_geometry(&snapshot);
        }
    }

    #[gpui::test]
    async fn test_long_inlays_at_buffer_edges(cx: &mut TestAppContext) {
        let (buffer, map) = build_display_map("x", 4, cx);
        let mut wrapper = cx.update(|cx| cx.text_system().line_wrapper(test_font(), FONT_SIZE));
        let buffer_snapshot = buffer.read_with(cx, |buffer, cx| buffer.snapshot(cx));
        let inlays = [
            Inlay::mock_hint(
                0,
                buffer_snapshot.anchor_before(MultiBufferOffset(0)),
                "漢".repeat(200),
            ),
            Inlay::mock_hint(
                1,
                buffer_snapshot.anchor_after(MultiBufferOffset(1)),
                "é".repeat(1_000),
            ),
        ];
        for inlay in inlays {
            map.update(cx, |map, cx| map.splice_inlays(&[], vec![inlay], cx));
            cx.run_until_parked();
            let snapshot = map.update(cx, |map, cx| map.snapshot(cx));
            assert_round_trips(&snapshot);
            assert_matches_char_widths(&snapshot, &mut wrapper);
        }
    }

    #[gpui::test]
    async fn test_tab_expanded_long_rows(cx: &mut TestAppContext) {
        let text = format!("{}{}", "\t".repeat(129), "é".repeat(50));
        let (_, map) = build_display_map(&text, 8, cx);
        let mut wrapper = cx.update(|cx| cx.text_system().line_wrapper(test_font(), FONT_SIZE));
        cx.run_until_parked();
        let snapshot = map.update(cx, |map, cx| map.snapshot(cx));
        assert_eq!(snapshot.line_len(DisplayRow(0)), 129 * 8 + 50 * 2);
        assert_matches_char_widths(&snapshot, &mut wrapper);
    }

    #[gpui::test]
    async fn test_long_rows_keep_grapheme_clusters_across_splices(cx: &mut TestAppContext) {
        let joined = (0..150)
            .map(|index| {
                let letter = char::from(b'a' + (index % 26) as u8);
                format!("{letter}\u{200d}\u{1f600}b\u{1f1fa}\u{1f1f8}e\u{301}")
            })
            .collect::<String>();
        let flags =
            |pairs: usize| format!("{}{}", "\u{1f1e8}\u{1f1e6}".repeat(pairs), "ab".repeat(600));
        for (text, joins_seam) in [(joined, true), (flags(40), false), (flags(300), false)] {
            let (buffer, map) = build_display_map(&text, 4, cx);
            cx.run_until_parked();
            let snapshot = map.update(cx, |map, cx| map.snapshot(cx));
            assert_cluster_atoms(&snapshot);

            let (edited, inserted) = if joins_seam {
                let long_row = snapshot.long_row(DisplayRow(0)).expect("a long row");
                let seam = long_row
                    .geometry
                    .placed_chunks()
                    .map(|chunk| chunk.end() as usize)
                    .find(|&end| text[end..].starts_with('\u{1f600}'))
                    .expect("a chunk boundary before an emoji");
                let joiner_start = seam - "a\u{200d}".len();
                (joiner_start..joiner_start + 1, "\u{1f600}")
            } else {
                (0..0, "\u{1f1e6}")
            };
            buffer.update(cx, |buffer, cx| {
                buffer.edit(
                    [(
                        MultiBufferOffset(edited.start)..MultiBufferOffset(edited.end),
                        inserted,
                    )],
                    None,
                    cx,
                )
            });
            let snapshot = map.update(cx, |map, cx| map.snapshot(cx));
            let long_row = snapshot.long_row(DisplayRow(0)).expect("a long row");
            assert_eq!(
                long_row.len() as usize,
                text.len() - edited.len() + inserted.len()
            );
            assert_cluster_atoms(&snapshot);
            cx.run_until_parked();
            assert_cluster_atoms(&map.update(cx, |map, cx| map.snapshot(cx)));
        }
    }

    #[gpui::test]
    async fn test_long_rows_use_measured_inlay_widths(cx: &mut TestAppContext) {
        let (buffer, map) = build_display_map(&"x".repeat(2_000), 4, cx);
        let position = buffer.read_with(cx, |buffer, cx| {
            buffer.snapshot(cx).anchor_after(MultiBufferOffset(10))
        });
        let inlay = Inlay::color(0, position, rgba(0xff0000ff));
        map.update(cx, |map, cx| map.splice_inlays(&[], vec![inlay], cx));
        cx.run_until_parked();
        let inlay_width = |map: &Entity<DisplayMap>, cx: &mut TestAppContext| {
            let snapshot = map.update(cx, |map, cx| map.snapshot(cx));
            let long_row = snapshot.long_row(DisplayRow(0)).expect("a long row");
            long_row.x_for_column(10 + "\u{25fc}".len() as u32) - long_row.x_for_column(10)
        };
        let mut wrapper = cx.update(|cx| cx.text_system().line_wrapper(test_font(), FONT_SIZE));
        assert_eq!(
            inlay_width(&map, cx),
            f64::from(wrapper.width_for_char('\u{25fc}'))
        );

        let key = |font_size: f32| RendererWidthsKey {
            rem_size: px(16.),
            font: test_font(),
            font_size: px(font_size),
            buffer_font: test_font(),
        };
        let renderer_widths = [(ChunkRendererId::Inlay(InlayId::Color(0)), px(37.))];
        assert!(map.update(cx, |map, cx| {
            map.update_renderer_widths(key(14.), renderer_widths, cx)
        }));
        assert_eq!(inlay_width(&map, cx), 37.);
        assert!(!map.update(cx, |map, cx| {
            map.update_renderer_widths(key(14.), renderer_widths, cx)
        }));

        assert!(map.update(cx, |map, cx| {
            map.update_renderer_widths(key(28.), [], cx)
        }));
        assert_eq!(
            inlay_width(&map, cx),
            f64::from(wrapper.width_for_char('\u{25fc}'))
        );
    }

    #[gpui::test]
    async fn test_soft_wrap_counts_renderer_width_once(cx: &mut TestAppContext) {
        let (buffer, map) = build_display_map("a", 4, cx);
        let position = buffer.read_with(cx, |buffer, cx| {
            buffer.snapshot(cx).anchor_after(MultiBufferOffset(1))
        });
        let inlay = Inlay::mock_repl_result(0, position, &"r".repeat(300));
        map.update(cx, |map, cx| {
            map.splice_inlays(&[], vec![inlay], cx);
            map.set_wrap_width(Some(px(150.)), cx);
        });
        cx.run_until_parked();
        map.update(cx, |map, cx| {
            map.update_renderer_widths(
                RendererWidthsKey {
                    rem_size: px(16.),
                    font: test_font(),
                    font_size: FONT_SIZE,
                    buffer_font: test_font(),
                },
                [(ChunkRendererId::Inlay(InlayId::ReplResult(0)), px(100.))],
                cx,
            )
        });
        cx.run_until_parked();
        let snapshot = map.update(cx, |map, cx| map.snapshot(cx));
        assert_eq!(snapshot.max_point().row(), DisplayRow(0));
    }

    #[gpui::test]
    async fn test_long_rows_replace_only_standalone_invisibles(cx: &mut TestAppContext) {
        let text = format!(
            "a{}\u{200c}b\u{200b}{}ab\u{200c}c a\u{200b}b",
            "\u{301}".repeat(15),
            "x".repeat(1_100)
        );
        let (buffer, map) = build_display_map(&text, 4, cx);
        let capped_joiner = text.find('\u{200c}').unwrap_or_default() as u32;
        let attached_joiner = text.rfind('\u{200c}').unwrap_or_default();
        let highlighted = text.rfind("ab").unwrap_or_default()..attached_joiner;
        let buffer_snapshot = buffer.read_with(cx, |buffer, cx| buffer.snapshot(cx));
        map.update(cx, |map, cx| {
            map.highlight_text(
                HighlightKey::Editor,
                vec![
                    buffer_snapshot.anchor_before(MultiBufferOffset(highlighted.start))
                        ..buffer_snapshot.anchor_after(MultiBufferOffset(highlighted.end)),
                ],
                HighlightStyle::color(gpui::red()),
                false,
                cx,
            )
        });
        cx.run_until_parked();
        let snapshot = map.update(cx, |map, cx| map.snapshot(cx));
        let style = EditorStyle::default();
        let replaced = snapshot
            .highlighted_chunks(
                DisplayRow(0)..DisplayRow(1),
                LanguageAwareStyling {
                    tree_sitter: false,
                    diagnostics: false,
                },
                &style,
            )
            .filter(|chunk| matches!(chunk.replacement, Some(ChunkReplacement::Str(_))))
            .map(|chunk| chunk.text.to_string())
            .collect::<Vec<_>>();
        assert_eq!(replaced, ["\u{200b}", "\u{200b}"]);

        let long_row = snapshot.long_row(DisplayRow(0)).expect("a long row");
        let attached_joiner = attached_joiner as u32;
        assert_eq!(
            long_row.atom_range(attached_joiner).start,
            attached_joiner - 1
        );
        assert_eq!(
            long_row.atom_range(capped_joiner),
            capped_joiner..capped_joiner + 3
        );
        let language_aware = LanguageAwareStyling {
            tree_sitter: false,
            diagnostics: false,
        };
        for columns in [0..long_row.len(), capped_joiner..long_row.len()] {
            let replaced = snapshot
                .highlighted_chunks_in_row(DisplayRow(0), columns.clone(), language_aware, &style)
                .filter(|chunk| matches!(chunk.replacement, Some(ChunkReplacement::Str(_))))
                .map(|chunk| chunk.text.to_string())
                .collect::<Vec<_>>();
            assert_eq!(replaced, ["\u{200b}", "\u{200b}"], "{columns:?}");
            let mut standalone = Vec::new();
            for_each_atom(&long_row.snapshot, long_row.row, columns.clone(), |atom| {
                standalone.extend(atom.standalone);
                true
            });
            assert_eq!(standalone, ['\u{200b}', '\u{200b}'], "{columns:?}");
        }
    }

    #[gpui::test]
    async fn test_long_row_windows_cover_dense_visible_text(cx: &mut TestAppContext) {
        let text = format!("a{}", "\u{301}".repeat(500)).repeat(300);
        let (_, map) = build_display_map(&text, 4, cx);
        cx.run_until_parked();
        let snapshot = map.update(cx, |map, cx| map.snapshot(cx));
        let long_row = snapshot.long_row(DisplayRow(0)).expect("a long row");
        let viewport_width = long_row.x_for_column(100 * 1_001);
        let window = long_row.window(0., TextAlign::Left, viewport_width);
        assert_eq!(window.segments().len(), 1);
        let columns = window.segments()[0].clone();
        assert!(columns.end - columns.start > MAX_WINDOW_LEN);
        assert!(long_row.x_for_column(columns.end) >= viewport_width);
    }

    #[gpui::test]
    async fn test_long_rows_cap_oversized_grapheme_clusters(cx: &mut TestAppContext) {
        let grapheme = format!("a{}", "\u{301}".repeat(512));
        let text = format!("{grapheme}b{}", "x".repeat(1_024));
        let (_, map) = build_display_map(&text, 4, cx);
        cx.run_until_parked();
        let snapshot = map.update(cx, |map, cx| map.snapshot(cx));
        assert_cluster_atoms(&snapshot);

        let long_row = snapshot.long_row(DisplayRow(0)).expect("a long row");
        let grapheme_end = grapheme.len() as u32;
        let atom_lens = long_row
            .window_layout(0..grapheme_end, 0.)
            .atom_positions
            .windows(2)
            .map(|pair| pair[1].0 - pair[0].0)
            .collect::<Vec<_>>();
        assert_eq!(atom_lens.iter().sum::<u32>(), grapheme_end);
        assert_eq!(atom_lens.first(), Some(&(1 + 15 * 2)));
        assert_eq!(atom_lens.iter().max(), Some(&(16 * 2)));
        let window = long_row.window(0., TextAlign::Left, 1.);
        assert_eq!(
            window.segments().first().map(|segment| segment.start),
            Some(0)
        );
        let last_atom_start = grapheme_end - '\u{301}'.len_utf8() as u32;
        assert_eq!(
            [0, last_atom_start, grapheme_end]
                .map(|column| snapshot.grapheme_at(DisplayPoint::new(DisplayRow(0), column))),
            [
                Some(SharedString::from(&text[..1 + 15 * 2])),
                Some(SharedString::from("\u{301}")),
                Some(SharedString::from("b")),
            ]
        );
    }

    #[gpui::test]
    async fn test_capped_cluster_context_ignores_unaligned_scan_floors(cx: &mut TestAppContext) {
        let cluster = format!("a{}", "\u{301}".repeat(MAX_ATOM_LEN as usize / 2));
        let text = format!("{}x{cluster}b", "é".repeat(1_000));
        let (_, map) = build_display_map(&text, 4, cx);
        cx.run_until_parked();
        let snapshot = map.update(cx, |map, cx| map.snapshot(cx));
        let long_row = snapshot.long_row(DisplayRow(0)).expect("a long row");
        let cluster_end = text.len() as u32 - 1;
        assert_eq!(
            long_row.context_bounds(cluster_end - 2..cluster_end).end,
            cluster_end + 1
        );
    }

    #[gpui::test]
    async fn test_pending_long_rows_scan_next_to_fold_placeholders(cx: &mut TestAppContext) {
        let placeholder_start = 17_000;
        let text = format!(
            "{}hide{}",
            "x".repeat(placeholder_start),
            "x".repeat(20_000)
        );
        cx.background_executor.set_block_on_ticks(0..=0);
        let (_, map) = build_display_map(&text, 4, cx);
        let creases = vec![Crease::simple(
            MultiBufferOffset(placeholder_start)..MultiBufferOffset(placeholder_start + 4),
            FoldPlaceholder::test(),
        )];
        map.update(cx, |map, cx| map.fold(creases, cx));
        let snapshot = map.update(cx, |map, cx| map.snapshot(cx));
        let long_row = snapshot.long_row(DisplayRow(0)).expect("a long row");
        assert!(!long_row.is_measured());

        let placeholder_len = "\u{22ef}".len() as u32;
        let placeholder_start = placeholder_start as u32;
        for column in (placeholder_start - 16 * 1024)..=(placeholder_start - 16 * 1024 + 2) {
            assert_eq!(long_row.floor_column(column), column);
        }
        for column in placeholder_start + 1..placeholder_start + placeholder_len {
            assert_eq!(
                long_row.atom_range(column),
                placeholder_start..placeholder_start + placeholder_len
            );
        }
        let window = long_row.window(
            long_row.x_for_column(placeholder_start - 2),
            TextAlign::Left,
            50.,
        );
        assert!(window.segments().iter().any(|segment| {
            segment.contains(&placeholder_start)
                && segment.end > placeholder_start + placeholder_len
        }));
        let style = EditorStyle::default();
        let language_aware = LanguageAwareStyling {
            tree_sitter: false,
            diagnostics: true,
        };
        for segment in window.segments() {
            let text = snapshot
                .highlighted_chunks_in_row(DisplayRow(0), segment.clone(), language_aware, &style)
                .map(|chunk| chunk.text)
                .collect::<String>();
            assert_eq!(text.len() as u32, segment.end - segment.start);
        }
    }

    #[gpui::test]
    async fn test_pending_long_rows_measure_only_requested_windows(cx: &mut TestAppContext) {
        let text = "é".repeat(20_000);
        cx.background_executor.set_block_on_ticks(0..=0);
        let (_, map) = build_display_map(&text, 4, cx);
        let snapshot = map.update(cx, |map, cx| map.snapshot(cx));
        let long_row = snapshot.long_row(DisplayRow(0)).expect("a long row");
        assert!(!long_row.is_measured());

        let window_start = text.len() as u32 / 2;
        let window_len = 100 * "é".len() as u32;
        let window = RowWindow {
            segments: smallvec![window_start..window_start + window_len],
            elided: None,
        };
        let window_x = long_row.x_for_column(window_start);
        let request = long_row
            .measure_request(&window, &[])
            .expect("the window is not measured yet");
        assert!(map.update(cx, |map, cx| map.measure_long_row_columns(&[request], cx)));

        let snapshot = map.update(cx, |map, cx| map.snapshot(cx));
        let long_row = snapshot.long_row(DisplayRow(0)).expect("a long row");
        assert!(!long_row.is_measured());
        assert!(long_row.measure_request(&window, &[]).is_none());
        assert_eq!(long_row.x_for_column(window_start), window_x);
        let mut wrapper = cx.update(|cx| cx.text_system().line_wrapper(test_font(), FONT_SIZE));
        let window_width = 100. * f64::from(wrapper.width_for_char('é'));
        assert!(
            (long_row.x_for_column(window_start + window_len) - window_x - window_width).abs()
                < 1e-2
        );
    }

    #[gpui::test]
    async fn test_pending_long_rows_measure_windows_around_queried_positions(
        cx: &mut TestAppContext,
    ) {
        let text = "é".repeat(50_000);
        cx.background_executor.set_block_on_ticks(0..=0);
        let (_, map) = build_display_map(&text, 4, cx);
        let snapshot = map.update(cx, |map, cx| map.snapshot(cx));
        let long_row = snapshot.long_row(DisplayRow(0)).expect("a long row");
        assert!(!long_row.is_measured());
        let mut wrapper = cx.update(|cx| cx.text_system().line_wrapper(test_font(), FONT_SIZE));
        let character_width = f64::from(wrapper.width_for_char('é'));
        let characters = 1_000;
        let x = f64::from(characters) * character_width;
        let column = characters * "é".len() as u32;

        assert_eq!(long_row.measured_closest_column_for_x(x), column);
        assert!((long_row.measured_x_for_column(column) - x).abs() < 1e-2);
        assert!(!long_row.is_measured());

        let adopted = map
            .update(cx, |map, cx| map.snapshot(cx))
            .long_row(DisplayRow(0))
            .expect("a long row");
        let window = long_row.window_around(column);
        assert!(
            window
                .segments()
                .iter()
                .all(|segment| !adopted.geometry.has_unmeasured(segment))
        );
        assert!(!adopted.is_measured());
        assert!((adopted.x_for_column(column) - x).abs() < 1e-2);
    }

    #[gpui::test]
    async fn test_vertical_targets_in_unmeasured_rows_count_characters(cx: &mut TestAppContext) {
        let characters = 200_000;
        let text = format!(
            "{}\n{}\n{}",
            "x".repeat(characters),
            "é".repeat(characters),
            "𝒜".repeat(characters)
        );
        cx.background_executor.set_block_on_ticks(0..=0);
        let (_, map) = build_display_map(&text, 4, cx);
        let snapshot = map.update(cx, |map, cx| map.snapshot(cx));
        let source = snapshot.long_row(DisplayRow(0)).expect("a long row");
        let narrow = snapshot.long_row(DisplayRow(1)).expect("a long row");
        let wide = snapshot.long_row(DisplayRow(2)).expect("a long row");
        assert_eq!([narrow.is_measured(), wide.is_measured()], [false, false]);
        let character = 100_000;
        let goal_x = source.measured_x_for_column(character);
        assert_eq!(
            [
                narrow.measured_closest_column_for_x(goal_x),
                wide.measured_closest_column_for_x(goal_x)
            ],
            [2 * character, 4 * (character / 2)]
        );
    }

    #[gpui::test]
    async fn test_unmeasured_regional_indicators_keep_their_pairs(cx: &mut TestAppContext) {
        cx.background_executor.set_block_on_ticks(0..=0);
        let short_pair = 1 + 8 * 5_000;
        let long_pair = 1 + 8 * 150_000;
        for (text, columns, boundaries) in [
            (
                format!("a\u{1f1fa}{}", "\u{1f1fa}\u{1f1f8}".repeat(10_000)),
                vec![short_pair + 4, short_pair],
                vec![(short_pair, false), (short_pair, true)],
            ),
            (
                format!("a{}", "\u{1f1e6}\u{1f1f2}".repeat(200_000)),
                vec![long_pair + 4, long_pair + 8_004, long_pair - 4],
                vec![
                    (long_pair, false),
                    (long_pair + 8_000, false),
                    (long_pair - 8, false),
                ],
            ),
        ] {
            let (buffer, map) = build_display_map(&text, 4, cx);
            let long_row = map
                .update(cx, |map, cx| map.snapshot(cx))
                .long_row(DisplayRow(0))
                .expect("a long row");
            assert!(!long_row.is_measured());
            assert_eq!(
                columns
                    .iter()
                    .map(|column| (
                        long_row.floor_column(*column),
                        is_cluster_start(&long_row.snapshot, long_row.row, *column),
                    ))
                    .collect::<Vec<_>>(),
                boundaries
            );
            drop((buffer, map, long_row));
            cx.update(|_| {});
        }
    }

    #[gpui::test]
    async fn test_line_splits_and_joins_keep_measured_geometry(cx: &mut TestAppContext) {
        let (buffer, map) = build_display_map(&"é".repeat(10_000), 4, cx);
        cx.run_until_parked();
        let measured = map
            .update(cx, |map, cx| map.snapshot(cx))
            .long_row(DisplayRow(0))
            .expect("a long row");
        assert!(measured.is_measured());
        cx.background_executor.set_block_on_ticks(0..=0);
        let edit = |range: Range<usize>, new_text: &str, cx: &mut TestAppContext| {
            buffer.update(cx, |buffer, cx| {
                buffer.edit(
                    [(
                        MultiBufferOffset(range.start)..MultiBufferOffset(range.end),
                        new_text,
                    )],
                    None,
                    cx,
                )
            });
            map.update(cx, |map, cx| map.snapshot(cx))
        };
        let assert_shifted = |long_row: &LongRow, columns: &[u32], shift: u32| {
            assert!(long_row.is_measured());
            let shift_x = measured.x_for_column(shift);
            for column in columns {
                let expected = measured.x_for_column(column + shift) - shift_x;
                let actual = long_row.x_for_column(*column);
                assert!(
                    (actual - expected).abs() <= POSITION_TOLERANCE,
                    "column {column}: {actual} != {expected}"
                );
            }
        };
        let split = 6_000;

        let snapshot = edit(split..split, "\n", cx);
        let rows = [0, 1].map(|row| snapshot.long_row(DisplayRow(row)).expect("a long row"));
        assert_shifted(&rows[0], &[2_000, 5_998], 0);
        assert_shifted(&rows[1], &[2_000, 13_998], split as u32);

        let snapshot = edit(split..split + 1, "", cx);
        let joined = snapshot.long_row(DisplayRow(0)).expect("a long row");
        assert_shifted(&joined, &[2_000, 15_000, 19_998], 0);
    }

    #[gpui::test]
    async fn test_inlay_only_windows_keep_buffer_ranges(cx: &mut TestAppContext) {
        let (buffer, map) = build_display_map("ab", 4, cx);
        let position = buffer.read_with(cx, |buffer, cx| {
            buffer.snapshot(cx).anchor_after(MultiBufferOffset(1))
        });
        let hint = Inlay::mock_hint(0, position, "h".repeat(4_096));
        map.update(cx, |map, cx| map.splice_inlays(&[], vec![hint], cx));
        cx.run_until_parked();
        let snapshot = map.update(cx, |map, cx| map.snapshot(cx));
        let row = DisplayRow(0);
        let long_row = snapshot.long_row(row).expect("a long row");
        let window = RowWindow {
            segments: smallvec![2_000..2_100],
            elided: None,
        };
        assert_eq!(
            snapshot.visible_buffer_ranges(
                Point::new(0, 0)..Point::new(0, 2),
                &[(row, long_row, window)]
            ),
            [Point::new(0, 1)..Point::new(0, 1)]
        );
    }

    #[test]
    fn test_budgeted_windows_keep_a_visible_prefix_and_elide_the_rest() {
        let window = |pieces: &[Piece], budget| {
            let window = budgeted_window(pieces, budget);
            (window.segments().to_vec(), window.elided())
        };
        let pieces = [
            Piece {
                columns: 0..600,
                density: 0.,
            },
            Piece {
                columns: 600..1_200,
                density: 0.,
            },
            Piece {
                columns: 1_200..1_207,
                density: 8.,
            },
            Piece {
                columns: 1_207..1_400,
                density: 0.5,
            },
        ];
        assert_eq!(
            window(&pieces, 100),
            (vec![1_200..1_207], Some(1_207..1_400))
        );
        assert_eq!(window(&pieces, 1_000), (vec![1_200..1_400], None));

        let oversized = [
            Piece {
                columns: 0..2_000,
                density: 0.001,
            },
            Piece {
                columns: 2_000..2_010,
                density: 8.,
            },
        ];
        assert_eq!(window(&oversized, 100), (Vec::new(), Some(0..2_010)));

        let interleaved = [0..10, 10..20, 20..30, 30..40, 40..50].map(|columns| Piece {
            density: if columns.start % 20 == 0 { 1. } else { 0. },
            columns,
        });
        assert_eq!(
            window(&interleaved, 20),
            (vec![0..10, 20..30], Some(40..50))
        );
    }

    #[test]
    fn test_aligned_windows_elide_oversized_glyph_dependencies() {
        let cx = &mut shaping_context();
        for (prefix, len, kept_end) in [
            (0, 1_048_576, 1_048_576),
            (0, 1_048_577, 0),
            (17, 1_048_594, 17),
        ] {
            let text = "x".repeat(len as usize + 1);
            let mut long_row = unmeasured_long_row(&text, cx);
            let font_id = long_row.widths.base_font;
            let glyphs = [0]
                .into_iter()
                .chain((prefix > 0).then_some(prefix as usize))
                .enumerate()
                .map(|(position, index)| {
                    (font_id, GlyphId(1), index, position as f64, Pixels::ZERO)
                });
            let piece = retained_piece(
                &text[..len as usize],
                2.,
                &[FontRun {
                    len: len as usize,
                    font_id,
                }],
                glyphs,
            );
            let mut builder = ChunkBuilder::default();
            builder.push(
                ChunkLayout {
                    end: len,
                    width: 2.,
                    content_end: Some(len),
                    atoms: vec![(0, 0.)],
                    pieces: vec![piece],
                }
                .chunk(0, true, len, None),
            );
            builder.push_unmeasured(&long_row.snapshot, long_row.row, len + 1);
            long_row.geometry = Arc::new(RowGeometry::new(
                builder.finish(),
                u32::MAX,
                0,
                RowStyles::default(),
                long_row.widths.estimate,
            ));
            let window = [0..1, len - 1..len]
                .into_iter()
                .collect::<RowWindow>()
                .with_elided(Some(len..len + 1));
            let aligned = long_row.chunk_aligned(&window);
            let expected = if kept_end == 0 {
                Vec::new()
            } else {
                vec![0..kept_end]
            };
            assert_eq!(aligned.segments(), expected);
            assert_eq!(aligned.elided(), Some(kept_end..len + 1));
            assert_eq!(long_row.chunk_aligned(&aligned), aligned);
        }
        let renderer = ChunkLayout {
            end: 1_048_582,
            atoms: vec![(0, 0.), (5, 1.)],
            ..ChunkLayout::default()
        };
        assert_eq!([0, 10, 1_048_582].map(|column| [Bias::Left, Bias::Right].map(|bias| renderer.boundary(column, bias))), [[0, 0], [5, 1_048_582], [1_048_582, 1_048_582]]);
    }

    #[gpui::test]
    async fn test_long_rows_keep_flag_pairs_after_capped_clusters(cx: &mut TestAppContext) {
        let flags = "\u{1f1fa}\u{1f1f8}".repeat(4);
        for prepend_count in [339, 340, 341] {
            let text = format!("{}{flags}{}", "x".repeat(1_100), "z".repeat(200));
            let (buffer, map) = build_display_map(&text, 4, cx);
            cx.run_until_parked();

            let prepend = "\u{d4e}".repeat(prepend_count);
            buffer.update(cx, |buffer, cx| {
                buffer.edit(
                    [(MultiBufferOffset(1_100)..MultiBufferOffset(1_100), prepend)],
                    None,
                    cx,
                )
            });
            cx.run_until_parked();
            let snapshot = map.update(cx, |map, cx| map.snapshot(cx));
            let long_row = snapshot.long_row(DisplayRow(0)).expect("a long row");
            let flags_start = 1_100 + prepend_count as u32 * "\u{d4e}".len() as u32;
            let atom_starts = long_row
                .window_layout(flags_start..flags_start + flags.len() as u32, 0.)
                .atom_positions
                .into_iter()
                .map(|(start, _)| start - flags_start)
                .collect::<Vec<_>>();
            assert_eq!(atom_starts, [8, 16, 24, 32], "{prepend_count} prepends");
        }
    }

    #[gpui::test]
    async fn test_long_rows_after_whole_buffer_edits(cx: &mut TestAppContext) {
        let text = format!(
            "{}{}\n{}",
            "\ta\n".repeat(50),
            "\tx".repeat(800),
            "\tb\n".repeat(50)
        );
        let (_, map) = build_display_map(&text, 4, cx);
        let mut wrapper = cx.update(|cx| cx.text_system().line_wrapper(test_font(), FONT_SIZE));
        cx.run_until_parked();
        assert_matches_char_widths(&map.update(cx, |map, cx| map.snapshot(cx)), &mut wrapper);

        set_tab_size(8, cx);
        let snapshot = map.update(cx, |map, cx| map.snapshot(cx));
        assert_round_trips(&snapshot);
        cx.run_until_parked();
        let snapshot = map.update(cx, |map, cx| map.snapshot(cx));
        assert!(snapshot.long_row(DisplayRow(50)).is_some());
        assert_matches_char_widths(&snapshot, &mut wrapper);
    }

    #[gpui::test]
    async fn test_folded_buffers_do_not_widen_long_rows(cx: &mut TestAppContext) {
        init_settings(4, cx);
        let long_text = "x".repeat(2_000);
        let buffer = cx.update(|cx| {
            MultiBuffer::build_multi(
                [
                    (
                        long_text.as_str(),
                        vec![Point::new(0, 0)..Point::new(0, 2_000)],
                    ),
                    ("short", vec![Point::new(0, 0)..Point::new(0, 5)]),
                ],
                cx,
            )
        });
        let map = new_display_map(&buffer, cx);
        cx.run_until_parked();
        assert!(
            map.update(cx, |map, cx| map.snapshot(cx))
                .max_long_row_width()
                > 0.
        );

        let long_buffer_id = buffer.read_with(cx, |buffer, cx| {
            buffer
                .snapshot(cx)
                .excerpts()
                .next()
                .map(|excerpt| excerpt.context.start.buffer_id)
                .expect("an excerpt")
        });
        map.update(cx, |map, cx| map.fold_buffers([long_buffer_id], cx));
        assert_eq!(
            map.update(cx, |map, cx| map.snapshot(cx))
                .max_long_row_width(),
            0.
        );
    }

    #[gpui::test]
    async fn test_long_rows_track_trailing_whitespace(cx: &mut TestAppContext) {
        let (buffer, map) = build_display_map(&format!("x{}", " ".repeat(10_000)), 4, cx);
        let trailing_whitespace_start = |map: &Entity<DisplayMap>, cx: &mut TestAppContext| {
            cx.run_until_parked();
            map.update(cx, |map, cx| map.snapshot(cx))
                .long_row(DisplayRow(0))
                .and_then(|long_row| long_row.trailing_whitespace_start())
        };
        assert_eq!(trailing_whitespace_start(&map, cx), Some(1));

        buffer.update(cx, |buffer, cx| {
            buffer.edit(
                [(MultiBufferOffset(5_000)..MultiBufferOffset(5_000), "y")],
                None,
                cx,
            )
        });
        assert_eq!(trailing_whitespace_start(&map, cx), Some(5_001));

        let row_end = buffer.read_with(cx, |buffer, cx| {
            buffer.snapshot(cx).anchor_after(MultiBufferOffset(10_002))
        });
        let hint = Inlay::mock_hint(0, row_end, "hint ".to_string());
        map.update(cx, |map, cx| map.splice_inlays(&[], vec![hint], cx));
        assert_eq!(trailing_whitespace_start(&map, cx), Some(10_007));
    }

    #[gpui::test]
    async fn test_row_windows_keep_styles_from_before_their_start(cx: &mut TestAppContext) {
        init_settings(128, cx);
        let text = format!("abc\n{}\t{}", "x".repeat(79), "y".repeat(3_000));
        let buffer = cx.new(|cx| Buffer::local(text, cx));
        buffer.update(cx, |buffer, cx| {
            let diagnostics = DiagnosticSet::new(
                [DiagnosticEntry::new(
                    PointUtf16::new(0, 0)..PointUtf16::new(0, 3),
                    Diagnostic {
                        severity: lsp::DiagnosticSeverity::ERROR,
                        group_id: 1,
                        message: "error".into(),
                        ..Diagnostic::default()
                    },
                )],
                buffer,
            );
            buffer.update_diagnostics(LanguageServerId(0), diagnostics, cx)
        });
        let buffer = cx.new(|cx| MultiBuffer::singleton(buffer, cx));
        let buffer_snapshot = buffer.read_with(cx, |buffer, cx| buffer.snapshot(cx));
        let map = new_display_map(&buffer, cx);
        let hint = Inlay::mock_hint(
            0,
            buffer_snapshot.anchor_after(MultiBufferOffset(1)),
            "h".repeat(4_096),
        );
        map.update(cx, |map, cx| {
            map.splice_inlays(&[], vec![hint], cx);
            map.highlight_text(
                HighlightKey::Editor,
                vec![
                    buffer_snapshot.anchor_before(Point::new(1, 79))
                        ..buffer_snapshot.anchor_after(Point::new(1, 80)),
                ],
                HighlightStyle::color(gpui::red()),
                false,
                cx,
            );
        });
        cx.run_until_parked();

        let snapshot = map.update(cx, |map, cx| map.snapshot(cx));
        let style = EditorStyle::default();
        let language_aware = LanguageAwareStyling {
            tree_sitter: false,
            diagnostics: true,
        };
        let inlay_window = merge_chunks(
            snapshot
                .highlighted_chunks_in_row(DisplayRow(0), 100..350, language_aware, &style)
                .map(|chunk| (chunk.text, chunk.diagnostic_underline_severity)),
        );
        assert_eq!(
            inlay_window,
            [("h".repeat(250), Some(lsp::DiagnosticSeverity::ERROR))]
        );
        let tab_window = merge_chunks(
            snapshot
                .highlighted_chunks_in_row(DisplayRow(1), 90..110, language_aware, &style)
                .map(|chunk| (chunk.text, chunk.style.and_then(|style| style.color))),
        );
        assert_eq!(tab_window, [(" ".repeat(20), Some(gpui::red()))]);
    }

    #[gpui::test]
    async fn test_exhausted_resync_leaves_gaps_unmeasured(cx: &mut TestAppContext) {
        let text = "x".repeat(20_000);
        cx.background_executor.set_block_on_ticks(0..=0);
        let (buffer, map) = build_display_map(&text, 4, cx);
        let far_window = 15_000..15_100;
        let request = map
            .update(cx, |map, cx| map.snapshot(cx))
            .long_row(DisplayRow(0))
            .expect("a long row")
            .measure_request(
                &RowWindow {
                    segments: smallvec![0..100, far_window.clone()],
                    elided: None,
                },
                &[],
            )
            .expect("unmeasured windows");
        assert!(map.update(cx, |map, cx| map.measure_long_row_columns(&[request], cx)));

        buffer.update(cx, |buffer, cx| {
            buffer.edit(
                [(MultiBufferOffset(0)..MultiBufferOffset(0), "y")],
                None,
                cx,
            )
        });
        let long_row = map
            .update(cx, |map, cx| map.snapshot(cx))
            .long_row(DisplayRow(0))
            .expect("a long row");
        let geometry = &long_row.geometry;
        let longest_measured_chunk = geometry
            .placed_chunks()
            .filter(|chunk| chunk.chunk.measured)
            .map(|chunk| chunk.chunk.len)
            .max()
            .unwrap_or_default();
        assert!(
            longest_measured_chunk <= RESYNC_RESERVE_LEN.max(MAX_CHUNK_LEN + MAX_ATOM_LEN),
            "{longest_measured_chunk}"
        );
        assert!(!geometry.has_unmeasured(&(far_window.start + 1..far_window.end + 1)));
    }

    #[gpui::test]
    async fn test_long_rows_shape_across_grapheme_seams_only(cx: &mut TestAppContext) {
        let text = format!("a{}\u{323}b{}", "\u{301}".repeat(500), "x".repeat(1_100));
        let (_, map) = build_display_map(&text, 4, cx);
        cx.run_until_parked();
        let snapshot = map.update(cx, |map, cx| map.snapshot(cx));
        let long_row = snapshot.long_row(DisplayRow(0)).expect("a long row");
        let grapheme_starts = text
            .grapheme_indices(true)
            .map(|(start, _)| start)
            .collect::<Vec<_>>();
        let (inside, between) = long_row
            .geometry
            .placed_chunks()
            .map(|chunk| chunk.end() as usize)
            .filter(|end| *end < text.len())
            .partition::<Vec<_>, _>(|seam| grapheme_starts.binary_search(seam).is_err());
        assert_eq!([inside.is_empty(), between.is_empty()], [true, false]);
        let contexts = |seam: usize| {
            let (snapshot, row, column) = (&long_row.snapshot, long_row.row, seam as u32);
            (
                left_context(snapshot, row, column, text.as_bytes().get(seam).copied()),
                right_context(
                    snapshot,
                    row,
                    column,
                    text.as_bytes().get(seam - 1).copied(),
                ),
            )
        };
        assert_eq!(
            contexts(text.floor_char_boundary(MAX_ATOM_LEN as usize)),
            (String::new(), String::new())
        );
        let context_len = SHAPING_CONTEXT_LEN as usize;
        for seam in between {
            let start = text.floor_char_boundary(seam - context_len);
            let end = text.ceil_char_boundary(seam + context_len);
            assert_eq!(
                contexts(seam),
                (text[start..seam].to_string(), text[seam..end].to_string()),
                "{seam}"
            );
        }
    }

    #[test]
    fn test_long_row_positions_match_whole_line_shaping_across_chunks() {
        let cx = &mut shaping_context();
        let mut state = 7_u64;
        let random_text = (0..1_500)
            .map(|_| {
                state = state
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1);
                ['A', 'V', 'f', 'i', 'x'][(state >> 33) as usize % 5]
            })
            .collect::<String>();
        let ligature_runs = (1..30)
            .map(|run| format!("x{}", "w".repeat(run * 7 % 101 + 20)))
            .collect::<String>();
        let forced_runs =
            [0, 1, 3].map(|lead| format!("{}{}", "x".repeat(lead), "w".repeat(1_501)));
        let reordered = format!("a\u{915}{}b", ShapingTextSystem::PRE_BASE_VOWEL).repeat(150);
        for text in [random_text, ligature_runs, reordered]
            .into_iter()
            .chain(forced_runs)
        {
            let (buffer, map) = build_display_map(&text, 4, cx);
            cx.run_until_parked();
            assert_matches_whole_line_shaping(&map, None, cx);
            for (offset, inserted) in [(0, "w"), (700, "w"), (3, "\u{301}w")] {
                buffer.update(cx, |buffer, cx| {
                    let offset = buffer.snapshot(cx).text().floor_char_boundary(offset);
                    buffer.edit(
                        [(
                            MultiBufferOffset(offset)..MultiBufferOffset(offset),
                            inserted,
                        )],
                        None,
                        cx,
                    )
                });
                cx.run_until_parked();
                assert_matches_whole_line_shaping(&map, None, cx);
            }
        }
    }

    #[test]
    fn test_styled_long_rows_cut_at_styled_seams() {
        let cx = &mut shaping_context();
        let text = "w".repeat(20_001);
        let (_, map) = build_display_map(&text, 4, cx);
        cx.run_until_parked();
        let runs = [
            FontRun {
                len: 1,
                font_id: ShapingTextSystem::REGULAR,
            },
            FontRun {
                len: text.len() - 1,
                font_id: ShapingTextSystem::BOLD,
            },
        ];
        let long_row = map
            .update(cx, |map, cx| map.snapshot(cx))
            .long_row(DisplayRow(0))
            .expect("a long row");
        assert!(long_row.is_measured());
        let row_runs =
            RowRuns::new(0..long_row.len(), runs.map(StyledRun::from)).expect("runs cover the row");
        let request = long_row
            .measure_request(&RowWindow::prefix(long_row.len()), &[row_runs])
            .expect("new styles");
        assert!(map.update(cx, |map, cx| map.measure_long_row_columns(&[request], cx)));
        assert_matches_whole_line_shaping(&map, Some(&runs), cx);
    }

    #[test]
    fn test_forced_cuts_keep_ligatures_that_end_the_row() {
        let cx = &mut shaping_context();
        let text = "w".repeat(1_153);
        let (_, map) = build_display_map(&text, 4, cx);
        cx.run_until_parked();
        let runs = [
            (0..1_025, ShapingTextSystem::REGULAR),
            (1_025..text.len(), ShapingTextSystem::BOLD),
        ]
        .map(|(columns, font_id)| FontRun {
            len: columns.len(),
            font_id,
        });
        let long_row = map
            .update(cx, |map, cx| map.snapshot(cx))
            .long_row(DisplayRow(0))
            .expect("a long row");
        let row_runs =
            RowRuns::new(0..long_row.len(), runs.map(StyledRun::from)).expect("runs cover the row");
        let request = long_row
            .measure_request(&RowWindow::prefix(long_row.len()), &[row_runs])
            .expect("new styles");
        assert!(map.update(cx, |map, cx| map.measure_long_row_columns(&[request], cx)));
        cx.run_until_parked();
        assert_matches_whole_line_shaping(&map, Some(&runs), cx);
    }

    #[test]
    fn test_oversized_clusters_keep_whole_line_shaping_in_any_measurement_order() {
        let cluster = format!(
            "a{}",
            ShapingTextSystem::PRE_BASE_VOWEL.to_string().repeat(100)
        );
        for (prefix_len, measure_inside_cluster_first) in
            [(1_100, false), (1_100, true), (1_101, false), (1_133, true)]
        {
            let text = format!("{}{cluster}{}", "x".repeat(prefix_len), "x".repeat(50));
            let cx = &mut shaping_context();
            if measure_inside_cluster_first {
                cx.background_executor.set_block_on_ticks(0..=0);
            }
            let (_, map) = build_display_map(&text, 4, cx);
            if measure_inside_cluster_first {
                let inside = prefix_len as u32 + 100;
                let request = map
                    .update(cx, |map, cx| map.snapshot(cx))
                    .long_row(DisplayRow(0))
                    .expect("a long row")
                    .measure_request(
                        &RowWindow {
                            segments: smallvec![inside..inside + 64],
                            elided: None,
                        },
                        &[],
                    )
                    .expect("an unmeasured window");
                assert!(map.update(cx, |map, cx| map.measure_long_row_columns(&[request], cx)));
            }
            cx.run_until_parked();
            assert_matches_whole_line_shaping(&map, None, cx);
            let long_row = map
                .update(cx, |map, cx| map.snapshot(cx))
                .long_row(DisplayRow(0))
                .expect("a long row");
            let wrapper = cx.update(|cx| cx.text_system().line_wrapper(test_font(), FONT_SIZE));
            let expected = wrapper.layout_line(
                &text,
                &[FontRun {
                    len: text.len(),
                    font_id: wrapper.font_id(),
                }],
            );
            assert_window_matches_layout(&long_row, 0..long_row.len(), 0., &expected);
            assert!(long_row.hard_breaks(0..long_row.len()).is_empty());
        }
    }

    #[test]
    fn test_oversized_cluster_foreground_measurement_stays_within_budget() {
        let cx = &mut shaping_context();
        let text = format!(
            "a{}",
            ShapingTextSystem::PRE_BASE_VOWEL.to_string().repeat(1_365)
        );
        assert_eq!(text.len(), 4_096);
        assert_eq!(text.graphemes(true).count(), 1);
        let long_row = unmeasured_long_row(&text, cx);
        let budget = SYNC_MEASURE_LEN + RESYNC_RESERVE_LEN;
        let mut wrapper = long_row.widths.wrapper.lock();
        let mut measurer = Measurer {
            snapshot: &long_row.snapshot,
            wrapper: &mut wrapper,
            estimate: long_row.widths.estimate,
            budget,
            styles: RowStyles::default(),
            cache: None,
        };
        let geometry = measurer.measure_prefix(long_row.row, long_row.len(), 0);
        let measured = geometry
            .placed_chunks()
            .filter(|chunk| chunk.chunk.measured)
            .map(|chunk| chunk.chunk.len)
            .sum::<u32>();
        assert_eq!(measured, 0, "budget {budget}");
        assert!(!geometry.is_measured());
        for budget in [budget, 2 * long_row.len()] {
            measurer.budget = budget;
            let mut builder = ChunkBuilder::default();
            assert_eq!(
                measurer.shape(long_row.row, 0..long_row.len(), false, false, &mut builder),
                0
            );
            assert_eq!(builder.len(), 0);
            assert_eq!(measurer.budget, 0);
        }
        for columns in [0..1, 2_000..2_003] {
            assert!(
                geometry
                    .measure_columns(
                        &long_row.snapshot,
                        long_row.row,
                        &[columns],
                        &[],
                        &mut wrapper,
                        &long_row.widths,
                        None,
                    )
                    .is_none()
            );
        }
        let styles = RowRuns::new(
            0..long_row.len(),
            [StyledRun::from(FontRun {
                font_id: ShapingTextSystem::BOLD,
                len: text.len(),
            })],
        )
        .expect("row styles");
        let geometry = futures_lite::future::block_on(geometry.measure_remaining(
            &long_row.snapshot,
            long_row.row,
            &mut wrapper,
        ));
        drop(wrapper);
        let measured = LongRow {
            geometry: Arc::new(geometry),
            ..long_row
        };
        assert!(measured.is_measured());
        assert!(
            measured
                .geometry
                .chunks
                .first()
                .expect("measured chunk")
                .layout
                .is_some()
        );
        measured.publish();
        let restyled = measured
            .restyled(slice::from_ref(&styles))
            .expect("updated styles");
        assert!(!restyled.is_measured());
        assert!(!restyled.refined().is_measured());
        assert!(restyled.geometry.styles.covers_identically(&styles));
        assert!(
            restyled
                .window_layout(0..restyled.len(), 0.)
                .pieces
                .is_empty()
        );
    }

    #[test]
    fn test_pending_cluster_interiors_cannot_be_shaped() {
        let cx = &mut shaping_context();
        let text = format!("a{}b", "\u{301}".repeat(MAX_GRAPHEME_CONTEXT_LEN as usize));
        let long_row = unmeasured_long_row(&text, cx);
        let end = long_row.len();
        assert_eq!(
            [
                0..end,
                0..end - 1,
                end - 1..end,
                0..3,
                1..end,
                16_385..16_387
            ]
            .map(|columns| long_row.can_shape(columns)),
            [false, false, true, false, false, false],
        );
        let plain = unmeasured_long_row(&"x".repeat(text.len()), cx);
        assert!(plain.can_shape(16_385..16_387));
    }

    #[test]
    fn test_oversized_cluster_viewport_does_not_expand_to_the_whole_cluster() {
        let cx = &mut shaping_context();
        let cluster = format!(
            "a{}",
            ShapingTextSystem::PRE_BASE_VOWEL
                .to_string()
                .repeat(MAX_CACHED_CHUNK_COLUMNS / 3 + 1)
        );
        assert_eq!(cluster.len(), 262_147);
        assert_eq!(cluster.graphemes(true).count(), 1);
        let text = format!("{}{cluster}", "x".repeat(1_024));
        let mut long_row = unmeasured_long_row(&text, cx);
        let styled_start = 1_024 + 1 + 3 * 43_690;
        let runs = [
            FontRun {
                len: styled_start,
                font_id: ShapingTextSystem::REGULAR,
            },
            FontRun {
                len: 60,
                font_id: ShapingTextSystem::BOLD,
            },
            FontRun {
                len: text.len() - styled_start - 60,
                font_id: ShapingTextSystem::REGULAR,
            },
        ];
        let styles =
            RowRuns::new(0..long_row.len(), runs.map(StyledRun::from)).expect("row styles");
        long_row.geometry = Arc::new(futures_lite::future::block_on(
            long_row
                .geometry
                .with_styles(vec![styles].into())
                .measure_remaining(
                    &long_row.snapshot,
                    long_row.row,
                    &mut long_row.widths.wrapper.lock(),
                ),
        ));
        let chunk = long_row.geometry.chunk_containing(styled_start as u32);
        assert!(chunk.start > 0);
        assert!(chunk.chunk.layout.is_some());
        let first = long_row.chunk_layout(&chunk);
        *long_row.widths.chunks.lock() = ChunkLayoutCache::default();
        *long_row.widths.layouts.lock() = LayoutCache::default();
        let second = long_row.chunk_layout(&chunk);
        assert!(Arc::ptr_eq(&first.layout, &second.layout));
        let prefix = long_row.geometry.chunk_containing(0);
        assert!(
            long_row
                .chunk_layout(&prefix)
                .layout
                .pieces
                .iter()
                .all(|piece| piece.index.is_none())
        );
        let scroll_x = long_row.width() / 2.;
        let viewport_width = 14.;
        let visible = long_row.visible_window(scroll_x, TextAlign::Left, viewport_width);
        let aligned = long_row.chunk_aligned_window(scroll_x, TextAlign::Left, viewport_width);
        let columns = aligned
            .segments()
            .iter()
            .map(|range| range.len())
            .sum::<usize>();
        let budget = 4 * MAX_ATOM_LEN;
        assert!(
            columns <= budget as usize,
            "viewport={viewport_width}, visible={visible:?}, aligned={aligned:?}, columns={columns}, budget={budget}"
        );
        let window = long_row.window_layout(aligned.segments()[0].clone(), scroll_x);
        assert!(window.atom_positions.first().expect("first visible atom").1 <= scroll_x);
        assert!(
            window.atom_positions.last().expect("last visible atom").1 >= scroll_x + viewport_width
        );
        let expected = long_row.widths.wrapper.lock().layout_line(&text, &runs);
        let _wrapper = long_row.widths.wrapper.lock();
        for columns in aligned.segments().iter().cloned().chain([
            styled_start as u32 - 3..styled_start as u32 + 63,
            styled_start as u32 + 63..styled_start as u32 + 66,
        ]) {
            let aligned = long_row.chunk_aligned(&[columns].into_iter().collect());
            for columns in aligned.segments() {
                assert_window_matches_layout(&long_row, columns.clone(), scroll_x, &expected);
            }
        }
    }

    #[test]
    fn test_retained_slice_preserves_precise_positions_and_glyph_dependencies() {
        for origin in [16_777_216., 1_099_511_627_776.] {
            let positions = [
                0.125,
                origin + 0.125,
                origin + 0.5,
                origin - 0.125,
                origin + 0.875,
                origin + 1.375,
            ];
            let width = origin + 1.625;

            let fonts =
                [(FontId(0), 3), (FontId(1), 7)].map(|(font_id, len)| FontRun { font_id, len });
            let piece_for_indices = |indices: [usize; 6]| {
                retained_piece(
                    "éabçdefg",
                    width,
                    &fonts,
                    indices
                        .into_iter()
                        .zip(positions)
                        .enumerate()
                        .map(|(id, (index, x))| {
                            (FontId(id / 3), GlyphId(id as u32), index, x, px(id as f32))
                        }),
                )
            };
            let mut piece = piece_for_indices([0, 4, 2, 4, 6, 8]);
            piece.columns = 4_096..4_106;
            piece.x = 123.5;
            let index = piece.index.as_deref().expect("retained index");
            assert_eq!(
                index
                    .partitions
                    .iter()
                    .map(|entry| entry.0)
                    .collect::<Vec<_>>(),
                [0, 2, 6, 8, 10]
            );
            assert_eq!(
                atom_offsets_x(
                    &piece.layout,
                    &[(0, 0), (2, 2), (6, 6), (7, 7), (8, 8), (9, 9)],
                    Some(index)
                ),
                [
                    0.125,
                    origin - 0.125,
                    origin + 0.875,
                    origin + 1.125,
                    origin + 1.375,
                    origin + 1.5,
                ]
            );

            let sliced = piece
                .window(&(4_099..4_101), 123.5 + origin + 0.5)
                .expect("visible dependency");
            assert_eq!(sliced.columns, 4_098..4_102);
            assert_eq!(sliced.text.as_str(), "abç");
            assert_eq!(sliced.x, 123.5 + origin + 0.5);
            assert_eq!(sliced.layout.width, px(0.375));
            assert_eq!(sliced.layout.len, 4);
            assert!(sliced.is_shaped_with(
                [(FontId(0), 1), (FontId(1), 3)].map(|(font_id, len)| FontRun { font_id, len })
            ));
            assert_eq!(
                positioned_glyphs(&sliced.layout, 0, 0.),
                [
                    (FontId(0), GlyphId(1), 2, -0.375, px(1.)),
                    (FontId(0), GlyphId(2), 0, 0., px(2.)),
                    (FontId(1), GlyphId(3), 2, -0.625, px(3.)),
                ]
            );
            let end = piece.window(&(4_104..4_105), 0.).expect("last glyph");
            assert_eq!(
                (end.x, end.layout.width),
                (123.5 + origin + 1.375, px(0.25))
            );
            assert_eq!(end.columns, 4_104..4_106);
            assert!(piece.window(&(4_106..4_107), 0.).is_none());
            let piece = piece_for_indices([0; 6]);
            let sliced = piece.window(&(3..5), 0.).expect("indivisible dependency");
            assert_eq!(sliced.columns, 0..10);
            assert!(Arc::ptr_eq(&piece.layout, &sliced.layout));
            let rebased = piece
                .window(&(3..5), origin + 0.125)
                .expect("visible glyphs");
            assert_eq!(rebased.columns, 0..10);
            assert_eq!(rebased.x, origin + 0.125);
            assert_eq!(rebased.layout.width, px(1.5));
            assert_eq!(
                positioned_glyphs(&rebased.layout, 0, 0.)
                    .into_iter()
                    .skip(1)
                    .map(|glyph| (glyph.2, glyph.3))
                    .collect::<Vec<_>>(),
                [(0, 0.), (0, 0.375), (0, -0.25), (0, 0.75), (0, 1.25)]
            );
            let end = piece
                .window(&(3..5), width + 10.)
                .expect("offscreen dependency");
            assert_eq!((end.x, end.layout.width), (width, Pixels::ZERO));
        }
    }

    #[test]
    fn test_background_merges_share_chunks_outside_foreground_changes() {
        let cx = &mut shaping_context();
        cx.background_executor.set_block_on_ticks(0..=0);
        let (_, map) = build_display_map(&"x".repeat(80_000), 4, cx);
        let source = map
            .update(cx, |map, cx| map.snapshot(cx))
            .long_row(DisplayRow(0))
            .expect("a long row");
        assert!(!source.is_measured());
        let (snapshot, row) = (&source.snapshot, source.row);
        let widths = source.widths.clone();
        let mut wrapper = widths.wrapper.lock();
        let measured = futures_lite::future::block_on(source.geometry.measure_remaining(
            snapshot,
            row,
            &mut wrapper,
        ));
        let current = source
            .geometry
            .measure_columns(
                snapshot,
                row,
                &[10_000..11_000],
                &[bold_runs(75_000..75_100)],
                &mut wrapper,
                &widths,
                None,
            )
            .expect("a foreground change");
        let merged = current
            .merged(&measured, snapshot, row, &widths)
            .expect("a merged geometry");
        let fresh = futures_lite::future::block_on(
            RowGeometry::unmeasured(snapshot, row, source.len(), widths.estimate, 0)
                .with_styles(current.styles.clone())
                .measure_remaining(snapshot, row, &mut wrapper),
        );
        drop(wrapper);
        let fresh_chunks = fresh
            .placed_chunks()
            .map(|chunk| (chunk.start, (chunk.end(), chunk.width)))
            .collect::<HashMap<_, _>>();
        let mismatches = merged
            .placed_chunks()
            .filter(|chunk| merged.is_canonical(chunk))
            .filter(|chunk| fresh_chunks.get(&chunk.start) != Some(&(chunk.end(), chunk.width)))
            .map(|chunk| chunk.start)
            .collect::<Vec<_>>();
        assert_eq!(mismatches, Vec::<u32>::new());
        assert!(
            merged
                .placed_chunks()
                .filter(|chunk| merged.is_canonical(chunk))
                .count()
                > 1_000
        );
        let shared = measured
            .chunks
            .iter()
            .chain(current.chunks.iter())
            .map(ptr::from_ref)
            .collect::<HashSet<_>>();
        let chunks = merged.chunks.iter().count();
        let copied = merged
            .chunks
            .iter()
            .filter(|chunk| !shared.contains(&ptr::from_ref(*chunk)))
            .count();
        let foreground_measured = current
            .placed_chunks()
            .filter(|chunk| chunk.chunk.measured)
            .count();
        assert!(
            chunks > 1_000 && copied <= foreground_measured + 64,
            "{copied} of {chunks} chunks were copied, {foreground_measured} measured in the foreground"
        );
    }

    #[test]
    fn test_background_merges_keep_full_merge_results() {
        let cx = &mut shaping_context();
        cx.background_executor.set_block_on_ticks(0..=0);
        let (_, map) = build_display_map(&"x".repeat(80_000), 4, cx);
        let source = map
            .update(cx, |map, cx| map.snapshot(cx))
            .long_row(DisplayRow(0))
            .expect("a long row");
        let (snapshot, row, len) = (&source.snapshot, source.row, source.len());
        let widths = source.widths.clone();
        let mut wrapper = widths.wrapper.lock();
        let complete = futures_lite::future::block_on(source.geometry.measure_remaining(
            snapshot,
            row,
            &mut wrapper,
        ));
        let partial = Measurer {
            snapshot,
            wrapper: &mut wrapper,
            estimate: widths.estimate,
            budget: 20_000,
            styles: RowStyles::default(),
            cache: None,
        }
        .measure_prefix(row, len, source.geometry.text_id);
        let scenarios = [
            (
                &complete,
                vec![10_000..11_000],
                vec![bold_runs(40_000..40_100)],
            ),
            (
                &complete,
                vec![39_500..40_500],
                vec![bold_runs(40_000..40_100)],
            ),
            (&complete, vec![60_000..61_000], vec![bold_runs(100..200)]),
            (&partial, vec![30_000..31_000], Vec::new()),
            (
                &partial,
                vec![5_000..6_000],
                vec![bold_runs(50_000..50_100)],
            ),
        ];
        for (index, (measured, ranges, runs)) in scenarios.into_iter().enumerate() {
            let current = source
                .geometry
                .measure_columns(snapshot, row, &ranges, &runs, &mut wrapper, &widths, None)
                .expect("a foreground change");
            assert_eq!(
                current
                    .merged(measured, snapshot, row, &widths)
                    .map(|merged| geometry_shape(&merged)),
                full_merge(&current, measured, snapshot, row, &widths)
                    .map(|merged| geometry_shape(&merged)),
                "scenario {index}"
            );
        }
    }

    #[test]
    fn test_width_changes_share_untouched_chunks() {
        let cx = &mut shaping_context();
        let (_, map) = build_display_map(&"x".repeat(80_000), 4, cx);
        cx.run_until_parked();
        let font_runs = |columns: Range<u32>, font_id| {
            RowRuns::new(
                columns.clone(),
                [StyledRun::from(FontRun {
                    len: columns.len(),
                    font_id,
                })],
            )
            .expect("runs cover the columns")
        };
        let long_row = map
            .update(cx, |map, cx| map.snapshot(cx))
            .long_row(DisplayRow(0))
            .expect("a long row");
        let request = long_row
            .measure_request(
                &RowWindow::prefix(long_row.len()),
                &[font_runs(0..long_row.len(), ShapingTextSystem::BOLD)],
            )
            .expect("new styles");
        assert!(map.update(cx, |map, cx| map.measure_long_row_columns(&[request], cx)));
        cx.run_until_parked();
        let bold = map
            .update(cx, |map, cx| map.snapshot(cx))
            .long_row(DisplayRow(0))
            .expect("a long row");
        assert!(bold.is_measured());
        let restyled = bold
            .restyled(&[font_runs(40_000..40_100, ShapingTextSystem::REGULAR)])
            .expect("a width change");
        assert_eq!(bold.width() - restyled.width(), 100.);
        let chunks = restyled.geometry.chunks.iter().count();
        let copied = restyled
            .geometry
            .chunks
            .iter()
            .filter(|chunk| {
                chunk
                    .runs
                    .as_ref()
                    .is_some_and(|runs| Arc::strong_count(&runs.runs) > 1)
            })
            .count();
        assert!(
            chunks > 1_000 && copied <= 64,
            "{copied} of {chunks} chunks were copied"
        );
    }

    #[test]
    fn test_style_coverage_updates_share_measured_chunks() {
        let cx = &mut shaping_context();
        let (_, map) = build_display_map(&"x".repeat(20_000), 4, cx);
        cx.run_until_parked();
        let long_row = map
            .update(cx, |map, cx| map.snapshot(cx))
            .long_row(DisplayRow(0))
            .expect("a long row");
        assert!(long_row.is_measured());
        let row_runs = RowRuns::new(
            5_000..5_100,
            [StyledRun::from(FontRun {
                len: 100,
                font_id: ShapingTextSystem::REGULAR,
            })],
        )
        .expect("runs cover the columns");
        let restyled = long_row
            .restyled(slice::from_ref(&row_runs))
            .expect("new style coverage");
        let request = long_row
            .measure_request(&RowWindow::prefix(0), slice::from_ref(&row_runs))
            .expect("new style coverage");
        assert!(map.update(cx, |map, cx| map.measure_long_row_columns(&[request], cx)));
        let measured = map
            .update(cx, |map, cx| map.snapshot(cx))
            .long_row(DisplayRow(0))
            .expect("a long row");
        let chunk_addresses = |long_row: &LongRow| {
            [
                long_row.geometry.chunks.first(),
                long_row.geometry.chunks.last(),
            ]
            .map(|chunk| chunk.map(ptr::from_ref))
        };
        assert_eq!(
            [chunk_addresses(&restyled), chunk_addresses(&measured)],
            [chunk_addresses(&long_row); 2]
        );
    }

    #[test]
    fn test_local_restyles_keep_whole_line_shaping() {
        let text = "w".repeat(20_000);
        for measured_before_restyle in [true, false] {
            let cx = &mut shaping_context();
            if !measured_before_restyle {
                cx.background_executor.set_block_on_ticks(0..=0);
            }
            let (_, map) = build_display_map(&text, 4, cx);
            if measured_before_restyle {
                cx.run_until_parked();
            }
            let long_row = map
                .update(cx, |map, cx| map.snapshot(cx))
                .long_row(DisplayRow(0))
                .expect("a long row");
            assert_eq!(long_row.is_measured(), measured_before_restyle);
            let row_runs = RowRuns::new(0..200, local_bold_runs(200).map(StyledRun::from))
                .expect("runs cover the window");
            let request = long_row
                .measure_request(&RowWindow::prefix(200), &[row_runs])
                .expect("new styles");
            assert!(map.update(cx, |map, cx| map.measure_long_row_columns(&[request], cx)));
            cx.run_until_parked();
            assert_matches_whole_line_shaping(&map, Some(&local_bold_runs(text.len() as u32)), cx);
        }
    }

    #[test]
    fn test_geometry_measured_with_older_styles_is_not_merged_past_style_changes() {
        let cx = &mut shaping_context();
        let text = "w".repeat(20_000);
        cx.background_executor.set_block_on_ticks(0..=0);
        let (_, map) = build_display_map(&text, 4, cx);
        let long_row = map
            .update(cx, |map, cx| map.snapshot(cx))
            .long_row(DisplayRow(0))
            .expect("a long row");
        assert!(!long_row.is_measured());
        let row_runs = RowRuns::new(0..200, local_bold_runs(200).map(StyledRun::from))
            .expect("runs cover the window");
        let request = long_row
            .measure_request(&RowWindow::prefix(200), &[row_runs])
            .expect("new styles");
        assert!(map.update(cx, |map, cx| map.measure_long_row_columns(&[request], cx)));
        let restyled = map
            .update(cx, |map, cx| map.snapshot(cx))
            .long_row(DisplayRow(0))
            .expect("a long row");
        let (snapshot, row, len) = (&restyled.snapshot, restyled.row, restyled.len());
        let widths = restyled.widths.clone();
        let mut wrapper = widths.wrapper.lock();
        let older = Measurer {
            snapshot,
            wrapper: &mut wrapper,
            estimate: widths.estimate,
            budget: u32::MAX,
            styles: RowStyles::default(),
            cache: None,
        }
        .measure_prefix(row, len, restyled.geometry.text_id);
        let merged = restyled
            .geometry
            .merged(&older, snapshot, row, &widths)
            .unwrap_or_else(|| (*restyled.geometry).clone());
        let measured = merged
            .measure_columns(snapshot, row, &[0..len], &[], &mut wrapper, &widths, None)
            .unwrap_or(merged);
        drop(wrapper);
        assert_long_row_matches_whole_line_shaping(
            &LongRow {
                geometry: Arc::new(measured),
                stored: None,
                ..restyled.clone()
            },
            &text,
            Some(&local_bold_runs(len)),
            cx,
        );
    }

    #[test]
    fn test_long_rows_keep_same_font_run_boundaries() {
        let padding = "x".repeat(10_000);
        let text = format!("{padding}fi{padding}");
        let split = padding.len() + 1;
        let colored = |columns: Range<usize>, color| {
            let text_run = TextRun {
                len: columns.len(),
                font: test_font(),
                color,
                background_color: None,
                underline: None,
                strikethrough: None,
            };
            let font_run = FontRun {
                len: columns.len(),
                font_id: ShapingTextSystem::REGULAR,
            };
            RowRuns::new(
                columns.start as u32..columns.end as u32,
                StyledRun::from_text_runs(&[text_run], [font_run]),
            )
            .expect("runs cover the columns")
        };
        let in_one_entry = RowRuns::new(
            0..text.len() as u32,
            [0..split, split..text.len()].map(|columns| {
                StyledRun::from(FontRun {
                    len: columns.len(),
                    font_id: ShapingTextSystem::REGULAR,
                })
            }),
        )
        .expect("runs cover the row");
        let across_entries = [
            colored(0..split, gpui::red()),
            colored(split..text.len(), gpui::blue()),
        ];
        for row_runs in [vec![in_one_entry], across_entries.to_vec()] {
            let cx = &mut shaping_context();
            let (_, map) = build_display_map(&text, 4, cx);
            cx.run_until_parked();
            let long_row = map
                .update(cx, |map, cx| map.snapshot(cx))
                .long_row(DisplayRow(0))
                .expect("a long row");
            let request = long_row
                .measure_request(&RowWindow::prefix(long_row.len()), &row_runs)
                .expect("new styles");
            assert!(map.update(cx, |map, cx| map.measure_long_row_columns(&[request], cx)));
            assert_matches_whole_line_shaping(
                &map,
                Some(&[0..split, split..text.len()].map(|columns| FontRun {
                    len: columns.len(),
                    font_id: ShapingTextSystem::REGULAR,
                })),
                cx,
            );
        }
    }

    #[test]
    fn test_background_measurement_keeps_newer_styles() {
        let cx = &mut shaping_context();
        let text = "x".repeat(20_000);
        cx.background_executor.set_block_on_ticks(0..=0);
        let (_, map) = build_display_map(&text, 4, cx);
        let long_row = map
            .update(cx, |map, cx| map.snapshot(cx))
            .long_row(DisplayRow(0))
            .expect("a long row");
        assert!(!long_row.is_measured());
        let bold = 10_000..10_100;
        let window = RowWindow {
            segments: smallvec![bold.clone()],
            elided: None,
        };
        let shaped = long_row.shaping_window(&window).segments()[0].clone();
        let font_run = |columns: Range<u32>, font_id| FontRun {
            len: columns.len(),
            font_id,
        };
        let row_runs = RowRuns::new(
            shaped.clone(),
            [
                font_run(shaped.start..bold.start, ShapingTextSystem::REGULAR),
                font_run(bold.clone(), ShapingTextSystem::BOLD),
                font_run(bold.end..shaped.end, ShapingTextSystem::REGULAR),
            ]
            .map(StyledRun::from),
        )
        .expect("runs cover the window");
        let request = long_row
            .measure_request(&window, &[row_runs])
            .expect("an unmeasured window");
        assert!(map.update(cx, |map, cx| map.measure_long_row_columns(&[request], cx)));
        cx.run_until_parked();
        assert_matches_whole_line_shaping(
            &map,
            Some(&[
                font_run(0..bold.start, ShapingTextSystem::REGULAR),
                font_run(bold.clone(), ShapingTextSystem::BOLD),
                font_run(bold.end..text.len() as u32, ShapingTextSystem::REGULAR),
            ]),
            cx,
        );
    }

    #[test]
    fn test_chunk_layouts_used_by_a_frame_outlive_the_cache_capacity() {
        let keys = (1..=2 * MAX_CACHED_CHUNK_COLUMNS / MAX_CHUNK_LEN as usize).collect::<Vec<_>>();
        let layout = Arc::new(ChunkLayout {
            end: 0,
            width: 0.,
            content_end: None,
            atoms: Vec::new(),
            pieces: Vec::new(),
        });
        let frame = |cache: &mut ChunkLayoutCache, keys: &[usize]| {
            cache.frame.begin();
            let hits = keys
                .iter()
                .filter(|key| {
                    let hit = cache.get(**key).is_some();
                    if !hit {
                        cache.insert(**key, MAX_CHUNK_LEN, layout.clone());
                    }
                    hit
                })
                .count();
            cache.frame.end();
            hits
        };
        let mut cache = ChunkLayoutCache::default();
        assert_eq!(
            [
                frame(&mut cache, &keys),
                frame(&mut cache, &keys),
                frame(&mut cache, &[]),
                frame(&mut cache, &keys[1..]),
                frame(&mut cache, &[]),
                frame(&mut cache, &keys[..1]),
            ],
            [0, keys.len(), 0, keys.len() - 1, 0, 0]
        );
    }

    #[gpui::test]
    async fn test_chunk_layouts_survive_edits_outside_their_chunks(cx: &mut TestAppContext) {
        let line = (0..4_096_usize)
            .map(|index| char::from(b'a' + ((index * 7 + index / 13) % 26) as u8))
            .collect::<String>();
        let (buffer, map) = build_display_map(&format!("{line}\n{line}"), 4, cx);
        cx.run_until_parked();
        let last_chunk = |cx: &mut TestAppContext| {
            let long_row = map
                .update(cx, |map, cx| map.snapshot(cx))
                .long_row(DisplayRow(0))
                .expect("a long row");
            assert!(long_row.is_measured());
            long_row.chunk_layout(&long_row.geometry.chunk_containing(long_row.len() - 1))
        };
        let before = last_chunk(cx);

        let inserted = "yy";
        buffer.update(cx, |buffer, cx| {
            buffer.edit(
                [(MultiBufferOffset(0)..MultiBufferOffset(0), inserted)],
                None,
                cx,
            )
        });
        let after = last_chunk(cx);
        assert_eq!(Arc::as_ptr(&after.layout), Arc::as_ptr(&before.layout));
        assert_eq!(
            after.atoms().collect::<Vec<_>>(),
            before
                .atoms()
                .map(|(column, x)| (column + inserted.len() as u32, x))
                .collect::<Vec<_>>()
        );
    }

    fn bold_runs(columns: Range<u32>) -> RowRuns {
        RowRuns::new(
            columns.clone(),
            [StyledRun::from(FontRun {
                len: columns.len(),
                font_id: ShapingTextSystem::BOLD,
            })],
        )
        .expect("runs cover the columns")
    }

    fn geometry_shape(geometry: &RowGeometry) -> Vec<(u32, u32, bool, bool, ScrollPixelOffset)> {
        geometry
            .placed_chunks()
            .map(|chunk| {
                (
                    chunk.start,
                    chunk.end(),
                    chunk.chunk.measured,
                    geometry.is_canonical(&chunk),
                    chunk.width,
                )
            })
            .collect()
    }

    fn full_merge(
        current: &RowGeometry,
        measured: &RowGeometry,
        snapshot: &TabSnapshot,
        row: u32,
        widths: &CharWidths,
    ) -> Option<RowGeometry> {
        let base = widths.base_font;
        let styles_agree_until = measured.styles.agreement_end(&current.styles, base);
        let current_chunks = current.placed_chunks().collect::<Vec<_>>();
        let measured_chunks = measured.placed_chunks().collect::<Vec<_>>();
        let current_fresh =
            |index: usize| current_chunks[index].is_fresh(&current.styles, snapshot, row, base);
        let measured_fresh = |index: usize| {
            let position = &measured_chunks[index];
            measured.is_canonical(position)
                && shaping_span(snapshot, row, position.columns()).end <= styles_agree_until
                && position.is_fresh(&current.styles, snapshot, row, base)
        };
        let push = |builder: &mut ChunkBuilder, source: &RowGeometry, position: &ChunkPosition| {
            let mut chunk = position.chunk.clone();
            chunk.canonical = source.is_canonical(position);
            builder.push(chunk);
        };
        let mut builder = ChunkBuilder::default();
        let mut changed = false;
        let (mut current_index, mut fresh) = (0, 0);
        while current_index < current_chunks.len() {
            let (current_start, fresh_start) = (current_index, fresh);
            loop {
                let (current_end, fresh_end) = (
                    current_chunks.get(current_index)?.end(),
                    measured_chunks.get(fresh)?.end(),
                );
                if current_end <= fresh_end {
                    current_index += 1;
                }
                if fresh_end <= current_end {
                    fresh += 1;
                }
                if current_end == fresh_end {
                    break;
                }
            }
            if (fresh_start..fresh).all(measured_fresh) {
                changed |= current_index - current_start != fresh - fresh_start
                    || (current_start..current_index).any(|index| {
                        !current.is_canonical(&current_chunks[index]) || !current_fresh(index)
                    });
                for position in &measured_chunks[fresh_start..fresh] {
                    push(&mut builder, measured, position);
                }
                continue;
            }
            let mut filler = fresh_start;
            for index in current_start..current_index {
                let position = &current_chunks[index];
                if current_fresh(index) {
                    push(&mut builder, current, position);
                    continue;
                }
                let gap = position.columns();
                changed |= position.chunk.measured;
                while filler < fresh && measured_chunks[filler].start < gap.start {
                    filler += 1;
                }
                while filler < fresh && measured_chunks[filler].end() <= gap.end {
                    if measured_fresh(filler) {
                        builder.push_unmeasured(snapshot, row, measured_chunks[filler].start);
                        push(&mut builder, measured, &measured_chunks[filler]);
                        changed = true;
                    }
                    filler += 1;
                }
                builder.push_unmeasured(snapshot, row, gap.end);
            }
        }
        if fresh != measured_chunks.len() {
            return None;
        }
        changed.then(|| {
            RowGeometry::new(
                builder.finish(),
                u32::MAX,
                current.text_id,
                current.styles.clone(),
                widths.estimate,
            )
        })
    }

    fn merge_chunks<'a, T: PartialEq>(
        chunks: impl Iterator<Item = (&'a str, T)>,
    ) -> Vec<(String, T)> {
        let mut merged = Vec::<(String, T)>::new();
        for (text, value) in chunks {
            match merged.last_mut() {
                Some((last_text, last_value)) if *last_value == value => last_text.push_str(text),
                _ => merged.push((text.to_string(), value)),
            }
        }
        merged
    }

    fn shaping_context() -> TestAppContext {
        TestAppContext::build_with_text_system(
            gpui::TestDispatcher::new(0),
            None,
            Arc::new(ShapingTextSystem(gpui::NoopTextSystem)),
        )
    }

    fn assert_matches_whole_line_shaping(
        map: &Entity<DisplayMap>,
        runs: Option<&[FontRun]>,
        cx: &mut TestAppContext,
    ) {
        let snapshot = map.update(cx, |map, cx| map.snapshot(cx));
        let long_row = snapshot.long_row(DisplayRow(0)).expect("a long row");
        let text = snapshot.buffer_snapshot().text();
        assert_long_row_matches_whole_line_shaping(&long_row, &text, runs, cx);
    }

    fn assert_long_row_matches_whole_line_shaping(
        long_row: &LongRow,
        text: &str,
        runs: Option<&[FontRun]>,
        cx: &mut TestAppContext,
    ) {
        assert!(long_row.is_measured());
        let wrapper = cx.update(|cx| cx.text_system().line_wrapper(test_font(), FONT_SIZE));
        let base_runs = [FontRun {
            len: text.len(),
            font_id: wrapper.font_id(),
        }];
        let runs = runs.unwrap_or(&base_runs);
        let layout = wrapper.layout_line(text, runs);
        let columns = text
            .grapheme_indices(true)
            .map(|(start, _)| start)
            .chain([text.len()])
            .collect::<Vec<_>>();
        for (column, expected) in columns
            .iter()
            .copied()
            .zip(cluster_interpolated_x(&layout, &columns))
        {
            let actual = long_row.x_for_column(column as u32);
            assert!(
                (actual - expected).abs() < 0.01,
                "column {column} of {} bytes: {actual} != {expected}",
                text.len()
            );
        }
    }

    fn local_bold_runs(end: u32) -> [FontRun; 3] {
        [
            (0..1, ShapingTextSystem::REGULAR),
            (1..101, ShapingTextSystem::BOLD),
            (101..end, ShapingTextSystem::REGULAR),
        ]
        .map(|(columns, font_id)| FontRun {
            len: columns.len(),
            font_id,
        })
    }

    fn cluster_interpolated_x(layout: &LineLayout, columns: &[usize]) -> Vec<ScrollPixelOffset> {
        let mut glyph_starts = std::collections::BTreeMap::<usize, f64>::new();
        for glyph in layout.runs.iter().flat_map(|run| &run.glyphs) {
            let x = f64::from(glyph.position.x);
            glyph_starts
                .entry(glyph.index)
                .and_modify(|start| *start = start.min(x))
                .or_insert(x);
        }
        let width = f64::from(layout.width);
        columns
            .iter()
            .enumerate()
            .map(|(index, &column)| {
                if column >= layout.len {
                    return width;
                }
                let next_column = columns.get(index + 1).copied().unwrap_or(layout.len);
                if glyph_starts.range(column..next_column).next().is_some() {
                    return f64::from(layout.x_for_index(column));
                }
                let Some((&cluster_start, &cluster_x)) = glyph_starts.range(..column).next_back()
                else {
                    return glyph_starts.values().next().copied().unwrap_or(width);
                };
                let (cluster_end, cluster_end_x) = glyph_starts
                    .range(column..)
                    .next()
                    .map_or((layout.len, width), |(index, x)| (*index, *x));
                let interior = columns
                    .iter()
                    .filter(|other| cluster_start < **other && **other < cluster_end)
                    .collect::<Vec<_>>();
                let rank = interior
                    .iter()
                    .position(|other| **other == column)
                    .unwrap_or(0)
                    + 1;
                cluster_x + (cluster_end_x - cluster_x) * rank as f64 / (interior.len() + 1) as f64
            })
            .collect()
    }

    fn assert_window_matches_layout(
        long_row: &LongRow,
        columns: Range<u32>,
        origin_x: ScrollPixelOffset,
        expected: &LineLayout,
    ) {
        let window = long_row.window_layout(columns.clone(), origin_x);
        assert_eq!(
            window.atom_positions.last().map(|entry| entry.0),
            Some(columns.end)
        );
        let actual = window
            .pieces
            .iter()
            .flat_map(|piece| {
                assert!(columns.start <= piece.columns.start && piece.columns.end <= columns.end);
                positioned_glyphs(&piece.layout, piece.columns.start as usize, piece.x)
            })
            .collect::<Vec<_>>();
        let expected = positioned_glyphs(expected, 0, 0.)
            .into_iter()
            .filter(|glyph| columns.contains(&(glyph.2 as u32)))
            .collect::<Vec<_>>();
        assert_eq!(actual, expected, "{columns:?}");
    }

    fn retained_piece(
        text: &str,
        width: f64,
        fonts: &[FontRun],
        glyphs: impl IntoIterator<Item = (FontId, GlyphId, usize, f64, Pixels)>,
    ) -> TextPiece {
        let mut runs = Vec::<ShapedRun>::new();
        let mut positions = Vec::new();
        for (font_id, id, index, x, y) in glyphs {
            if runs.last().is_none_or(|run| run.font_id != font_id) {
                runs.push(ShapedRun {
                    font_id,
                    glyphs: Vec::new(),
                });
            }
            let run = runs.last_mut().expect("glyph run");
            run.glyphs.push(ShapedGlyph {
                id,
                position: point(px(x as f32), y),
                index,
                is_emoji: false,
            });
            positions.push(x);
        }
        let layout = Arc::new(LineLayout {
            len: text.len(),
            width: px(width as f32),
            runs,
            ..LineLayout::default()
        });
        TextPiece {
            columns: 0..text.len() as u32,
            x: 0.,
            text: SharedString::new(text),
            index: Some(Arc::new(LayoutIndex::new(&layout, fonts, positions, width))),
            layout,
            runs: SmallVec::from_slice(fonts),
        }
    }

    fn positioned_glyphs(
        layout: &LineLayout,
        column: usize,
        x: ScrollPixelOffset,
    ) -> Vec<(FontId, GlyphId, usize, ScrollPixelOffset, Pixels)> {
        layout
            .runs
            .iter()
            .flat_map(|run| {
                run.glyphs.iter().map(move |glyph| {
                    (
                        run.font_id,
                        glyph.id,
                        column + glyph.index,
                        x + f64::from(glyph.position.x),
                        glyph.position.y,
                    )
                })
            })
            .collect()
    }

    fn unmeasured_long_row(text: &str, cx: &mut TestAppContext) -> LongRow {
        init_settings(4, cx);
        cx.update(|cx| {
            let buffer = MultiBuffer::build_simple(text, cx);
            let (_, inlay_snapshot) = InlayMap::new(buffer.read(cx).snapshot(cx));
            let (_, fold_snapshot) = FoldMap::new(inlay_snapshot);
            let (_, snapshot) = TabMap::new(fold_snapshot, NonZeroU32::new(4).expect("tab size"));
            LongRows::new(cx.text_system().line_wrapper(test_font(), FONT_SIZE))
                .row(&snapshot, 0)
                .expect("a long row")
        })
    }

    fn build_display_map(
        text: &str,
        tab_size: u32,
        cx: &mut TestAppContext,
    ) -> (Entity<MultiBuffer>, Entity<DisplayMap>) {
        init_settings(tab_size, cx);
        let buffer = cx.update(|cx| MultiBuffer::build_simple(text, cx));
        let map = new_display_map(&buffer, cx);
        (buffer, map)
    }

    fn init_settings(tab_size: u32, cx: &mut TestAppContext) {
        cx.update(|cx| {
            let settings = SettingsStore::test(cx);
            cx.set_global(settings);
            crate::init(cx);
            theme_settings::init(LoadThemes::JustBase, cx);
        });
        set_tab_size(tab_size, cx);
    }

    fn set_tab_size(tab_size: u32, cx: &mut TestAppContext) {
        cx.update(|cx| {
            cx.update_global::<SettingsStore, _>(|store, cx| {
                store.update_user_settings(cx, |settings| {
                    settings.project.all_languages.defaults.tab_size = NonZeroU32::new(tab_size);
                });
            });
        });
    }

    fn new_display_map(
        buffer: &Entity<MultiBuffer>,
        cx: &mut TestAppContext,
    ) -> Entity<DisplayMap> {
        cx.new(|cx| {
            DisplayMap::new(
                buffer.clone(),
                test_font(),
                FONT_SIZE,
                None,
                1,
                1,
                FoldPlaceholder::test(),
                DiagnosticSeverity::Warning,
                cx,
            )
        })
    }

    fn randomly_mutate(
        buffer: &Entity<MultiBuffer>,
        map: &Entity<DisplayMap>,
        rng: &mut StdRng,
        cx: &mut TestAppContext,
    ) {
        let text = buffer.read_with(cx, |buffer, cx| buffer.snapshot(cx).text());
        let offset = |rng: &mut StdRng, range: RangeInclusive<usize>| {
            let mut offset = rng.random_range(range);
            while !text.is_char_boundary(offset) {
                offset -= 1;
            }
            offset
        };
        let start = offset(rng, 0..=text.len());
        let end = offset(rng, start..=(start + 60).min(text.len()));
        match rng.random_range(0..8) {
            0 if start < end => {
                let creases = vec![Crease::simple(
                    MultiBufferOffset(start)..MultiBufferOffset(end),
                    FoldPlaceholder::test(),
                )];
                map.update(cx, |map, cx| map.fold(creases, cx));
            }
            1 => {
                let position = buffer.read_with(cx, |buffer, cx| {
                    buffer.snapshot(cx).anchor_after(MultiBufferOffset(start))
                });
                let inlay = Inlay::mock_hint(
                    rng.random_range(0..usize::MAX),
                    position,
                    random_text(rng, 1..200).as_str(),
                );
                map.update(cx, |map, cx| map.splice_inlays(&[], vec![inlay], cx));
            }
            _ => {
                let mut edits = Vec::new();
                let mut edit_start = start;
                for _ in 0..rng.random_range(1..=3) {
                    if edit_start >= text.len() {
                        break;
                    }
                    let edit_end = offset(rng, edit_start..=(edit_start + 40).min(text.len()));
                    let inserted = if rng.random_ratio(1, 6) {
                        "\n".to_string()
                    } else {
                        random_text(rng, 0..200)
                    };
                    edits.push((
                        MultiBufferOffset(edit_start)..MultiBufferOffset(edit_end),
                        inserted,
                    ));
                    edit_start = offset(rng, edit_end..=(edit_end + 80).min(text.len()));
                }
                buffer.update(cx, |buffer, cx| buffer.edit(edits, None, cx));
            }
        }
    }

    fn random_text(rng: &mut StdRng, chars: Range<usize>) -> String {
        const ALPHABET: &[char] = &['a', 'W', ' ', '\t', 'é', '漢', '😀', '\u{0}', '\u{7f}'];
        let len = rng.random_range(chars);
        (0..len)
            .map(|_| ALPHABET[rng.random_range(0..ALPHABET.len())])
            .collect()
    }

    fn assert_round_trips(snapshot: &DisplaySnapshot) {
        for row in 0..=snapshot.max_point().row().0 {
            let Some(long_row) = snapshot.long_row(DisplayRow(row)) else {
                continue;
            };
            let line = snapshot.line(DisplayRow(row));
            assert_eq!(long_row.len() as usize, line.len());
            let mut previous_x = -1.;
            for (column, _) in line.char_indices().step_by(7) {
                let x = long_row.x_for_column(column as u32);
                assert!(
                    x > previous_x,
                    "row {row} column {column}: {x} <= {previous_x}"
                );
                assert_eq!(long_row.closest_column_for_x(x) as usize, column);
                previous_x = x;
            }
        }
    }

    fn assert_cluster_atoms(snapshot: &DisplaySnapshot) {
        for row in 0..=snapshot.max_point().row().0 {
            let Some(long_row) = snapshot.long_row(DisplayRow(row)) else {
                continue;
            };

            let line = snapshot.line(DisplayRow(row));
            let expected_atoms = line
                .grapheme_indices(true)
                .flat_map(|(start, grapheme)| capped_atoms(start, grapheme))
                .collect::<Vec<_>>();
            let atom_starts = long_row
                .window_layout(0..long_row.len(), 0.)
                .atom_positions
                .into_iter()
                .map(|(start, _)| start as usize)
                .filter(|start| *start < line.len())
                .collect::<Vec<_>>();
            assert_eq!(
                atom_starts,
                expected_atoms
                    .iter()
                    .map(|atom| atom.start)
                    .collect::<Vec<_>>(),
                "row {row}"
            );
            for atom in expected_atoms {
                let x = long_row.x_for_column(atom.start as u32);
                let inner =
                    atom.start + line[atom.clone()].chars().next().map_or(0, char::len_utf8);
                if inner < atom.end {
                    assert_eq!(
                        long_row.x_for_column(inner as u32),
                        long_row.x_for_column(atom.end as u32),
                        "row {row}"
                    );
                }
                assert_eq!(
                    long_row.closest_column_for_x(x) as usize,
                    atom.start,
                    "row {row}"
                );
            }
        }
    }

    fn capped_atoms(start: usize, grapheme: &str) -> Vec<Range<usize>> {
        let mut atoms = Vec::new();
        let mut atom_start = start;
        for (offset, character) in grapheme.char_indices() {
            let end = start + offset + character.len_utf8();
            if end - atom_start > MAX_ATOM_LEN as usize && !is_regional_indicator(character) {
                atoms.push(atom_start..start + offset);
                atom_start = start + offset;
            }
        }
        atoms.push(atom_start..start + grapheme.len());
        atoms
    }

    fn assert_matches_fresh_geometry(snapshot: &DisplaySnapshot) {
        for row in 0..=snapshot.max_point().row().0 {
            let Some(long_row) = snapshot.long_row(DisplayRow(row)) else {
                continue;
            };
            assert!(long_row.geometry.is_settled(), "row {row}");
            let mut wrapper = long_row.widths.wrapper.lock();
            let fresh = Measurer {
                snapshot: &long_row.snapshot,
                wrapper: &mut wrapper,
                estimate: long_row.widths.estimate,
                budget: u32::MAX,
                styles: RowStyles::default(),
                cache: None,
            }
            .measure_prefix(long_row.row, long_row.len(), 0);
            let ends = |geometry: &RowGeometry| {
                geometry
                    .placed_chunks()
                    .map(|chunk| chunk.end())
                    .collect::<Vec<_>>()
            };
            assert_eq!(ends(&long_row.geometry), ends(&fresh), "row {row}");
            assert!(
                fresh.placed_chunks().count() as u32 <= long_row.len() / MIN_CHUNK_LEN + 1,
                "row {row}"
            );
            for (stored, fresh) in long_row.geometry.placed_chunks().zip(fresh.placed_chunks()) {
                assert!(
                    (stored.x_end() - fresh.x_end()).abs() < 1e-6,
                    "row {row} chunk ending at {}: {} != {}",
                    stored.end(),
                    stored.x_end(),
                    fresh.x_end()
                );
            }
        }
    }

    fn assert_matches_char_widths(snapshot: &DisplaySnapshot, wrapper: &mut LineWrapper) {
        for row in 0..=snapshot.max_point().row().0 {
            let line = snapshot.line(DisplayRow(row));
            let Some(long_row) = snapshot.long_row(DisplayRow(row)) else {
                assert!(
                    line.len() <= MAX_LINE_LEN,
                    "row {row} has no long row geometry"
                );
                continue;
            };
            let standalone_invisibles = line
                .grapheme_indices(true)
                .filter(|(_, grapheme)| single_invisible(grapheme).is_some())
                .map(|(start, grapheme)| start..start + grapheme.len())
                .collect::<Vec<_>>();
            let mut expected_x = 0.;
            for (index, (column, character)) in line.char_indices().enumerate() {
                if index % 5 == 0 {
                    let x = long_row.x_for_column(column as u32);
                    assert!(
                        (x - expected_x).abs() < 1e-2,
                        "row {row} column {column}: {x} != {expected_x}"
                    );
                }
                let measured = if standalone_invisibles
                    .iter()
                    .any(|range| range.start == column)
                {
                    replacement(character).unwrap_or(character)
                } else {
                    character
                };
                expected_x += f64::from(wrapper.width_for_char(measured));
            }
            assert!((long_row.width() - expected_x).abs() < 1e-2);
        }
    }
}
