use std::{
    cell::{Cell, RefCell},
    rc::Rc,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
    time::Duration,
};

use editor::{
    Editor, EditorMode, MultiBuffer,
    display_map::{DisplayPoint, DisplayRow, DisplaySnapshot},
};
use gpui::{
    AnyWindowHandle, AppContext as _, BenchAppContext, Context, Entity, Focusable as _,
    IntoElement, ParentElement as _, Render, Styled as _, TextRun, Window, div, point,
};
use language::{Buffer, LineEnding, Rope, language_settings::SoftWrap};

#[gpui::bench(
    inputs = long_row_characters(),
    group = "Scroll an unwrapped long row",
    input_name = "character",
    sample_size = 10
)]
fn editor_scroll_long_row(character: &char, cx: &mut BenchAppContext) {
    init_context(cx);

    let text = Rope::from(character.to_string().repeat(LONG_ROW_CHARACTERS).as_str());
    let oracle_text = character.to_string().repeat(256);
    let mut session_window = None::<AnyWindowHandle>;
    cx.bench_renderer_session(
        LONG_ROW_SESSION_TIMEOUT,
        move |cx| {
            let window =
                *session_window.get_or_insert_with(|| cx.add_empty_window().window_handle());
            let text = text.clone();
            let editor_slot = Rc::new(RefCell::new(None));
            let renders = Rc::new(Cell::new(0));
            let columns_per_character = cx
                .update_window(window, |_, window, cx| {
                    let editor = cx.new(|cx| {
                        let buffer =
                            cx.new(|cx| Buffer::local_normalized(text, LineEnding::Unix, cx));
                        let buffer = cx.new(|cx| MultiBuffer::singleton(buffer, cx));
                        let mut editor = Editor::new(EditorMode::full(), buffer, None, window, cx);
                        editor.set_soft_wrap_mode(SoftWrap::None, cx);
                        editor
                    });
                    editor_slot.replace(Some(editor.clone()));
                    window.replace_root(cx, |_, _| LongRowRoot {
                        editor: editor_slot.clone(),
                        renders: renders.clone(),
                    });
                    window.focus(&editor.focus_handle(cx), cx);
                    editor.update(cx, |editor, cx| {
                        let style = editor.style(cx);
                        let font_size = style.text.font_size.to_pixels(window.rem_size());
                        let layout = |text: &str| {
                            window.text_system().layout_line(
                                text,
                                font_size,
                                &[TextRun {
                                    len: text.len(),
                                    font: style.text.font(),
                                    ..TextRun::default()
                                }],
                                None,
                            )
                        };
                        let advance = f64::from(layout(&oracle_text).width) / 256.;
                        let em_width = f64::from(layout("m").width);
                        let columns_per_character = advance / em_width;
                        assert!(columns_per_character.is_finite() && columns_per_character > 0.);
                        columns_per_character
                    })
                })
                .expect("the benchmark window must remain open");
            let stopped = Arc::new(AtomicBool::new(false));
            let session = LongRowSession {
                editor: editor_slot,
                renders,
                stopped: stopped.clone(),
                bytes_per_character: character.len_utf8() as u32,
                columns_per_character,
                scroll_x: LONG_ROW_SCROLL_START,
                requested_scroll: None,
                visible_start_character: 0,
            };
            (session, window, stopped)
        },
        |session, turn, window, cx| {
            let Some(editor) = session.editor.borrow().clone() else {
                return;
            };
            editor.update(cx, |editor, cx| {
                let display_snapshot = editor.display_snapshot(cx);
                assert_eq!(
                    display_snapshot.max_point(),
                    DisplayPoint::new(
                        DisplayRow(0),
                        LONG_ROW_CHARACTERS as u32 * session.bytes_per_character,
                    ),
                    "the complete line must remain unwrapped at turn {turn}"
                );
                if let Some(requested_scroll) = session.requested_scroll.take() {
                    session.validate_scroll(requested_scroll, turn, editor, &display_snapshot, cx);
                }
                if turn == LONG_ROW_SCROLL_TURNS {
                    session.stopped.store(true, Ordering::Release);
                    return;
                }
                session.scroll_x += LONG_ROW_SCROLL_STEP;
                let position = point(session.scroll_x, 0.);
                editor.set_scroll_position(position, window, cx);
                assert_eq!(
                    editor.scroll_position(cx),
                    position,
                    "scroll turn {turn} did not accept the scroll request"
                );
                session.requested_scroll = Some(RequestedScroll {
                    expected_start_character: position.x / session.columns_per_character,
                    renders: session.renders.get(),
                });
            });
        },
    );
}

struct LongRowSession {
    editor: Rc<RefCell<Option<Entity<Editor>>>>,
    renders: Rc<Cell<u64>>,
    stopped: Arc<AtomicBool>,
    bytes_per_character: u32,
    columns_per_character: f64,
    scroll_x: f64,
    requested_scroll: Option<RequestedScroll>,
    visible_start_character: u32,
}

impl LongRowSession {
    fn validate_scroll(
        &mut self,
        requested_scroll: RequestedScroll,
        turn: u64,
        editor: &Editor,
        display_snapshot: &DisplaySnapshot,
        cx: &mut Context<Editor>,
    ) {
        assert!(
            self.renders.get() > requested_scroll.renders,
            "no render followed the scroll request preceding turn {turn}"
        );
        let visible_ranges = editor.visible_text_ranges(display_snapshot, cx);
        let [visible_range] = visible_ranges.as_slice() else {
            panic!("expected one visible range at turn {turn}, got {visible_ranges:?}");
        };
        assert!(
            visible_range.start.row == 0
                && visible_range.end.row == 0
                && visible_range.start.column < visible_range.end.column
                && visible_range.start.column % self.bytes_per_character == 0
                && visible_range.end.column % self.bytes_per_character == 0,
            "invalid visible range {visible_range:?} at turn {turn}"
        );
        let visible_start_character = visible_range.start.column / self.bytes_per_character;
        let expected = requested_scroll.expected_start_character;
        assert!(
            visible_start_character > self.visible_start_character
                && (f64::from(visible_start_character) - expected).abs()
                    <= LONG_ROW_CHARACTER_TOLERANCE,
            "visible range at turn {turn} starts at character {visible_start_character}, \
             expected after character {} and within {LONG_ROW_CHARACTER_TOLERANCE} of {expected}",
            self.visible_start_character
        );
        self.visible_start_character = visible_start_character;
    }
}

struct RequestedScroll {
    expected_start_character: f64,
    renders: u64,
}

impl Drop for LongRowSession {
    fn drop(&mut self) {
        self.editor.borrow_mut().take();
    }
}

struct LongRowRoot {
    editor: Rc<RefCell<Option<Entity<Editor>>>>,
    renders: Rc<Cell<u64>>,
}

impl Render for LongRowRoot {
    fn render(&mut self, _: &mut Window, _: &mut Context<Self>) -> impl IntoElement {
        self.renders.set(self.renders.get() + 1);
        div().size_full().children(self.editor.borrow().clone())
    }
}

const LONG_ROW_CHARACTERS: usize = 20_000_000;
const LONG_ROW_SCROLL_TURNS: u64 = 8;
const LONG_ROW_SCROLL_START: f64 = 1_000_000.;
const LONG_ROW_SCROLL_STEP: f64 = 200.;
const LONG_ROW_CHARACTER_TOLERANCE: f64 = 1.;
const LONG_ROW_SESSION_TIMEOUT: Duration = Duration::from_secs(30);

fn long_row_characters() -> Vec<char> {
    vec!['x', 'é', '漢']
}

fn init_context(cx: &mut BenchAppContext) {
    cx.update(|cx| {
        settings::init(cx);
        assets::Assets.load_test_fonts(cx);
        theme_settings::init(theme::LoadThemes::JustBase, cx);
        editor::init(cx);
    });
}

gpui::bench_group!(benches, editor_scroll_long_row);
gpui::bench_main!(benches);
