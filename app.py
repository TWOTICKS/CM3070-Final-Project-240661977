#!/usr/bin/env python
import html
import os
import sys
import tempfile

import gradio as gr

sys.path.insert(0, os.path.dirname(__file__))
from src.chunking import build_chunks  # noqa: E402
from src.index_store import build_index, save_index  # noqa: E402
from src.pipeline import RAGPipeline  # noqa: E402

INDEX_DIR = os.path.join(os.path.dirname(__file__), "data", "index")
SAMPLE_LABEL = "Sample lecture (Introduction to Machine Learning)"

EXAMPLES = [
    "What is dropout and how does it help prevent overfitting?",
    "How does backpropagation work and why is it important?",
    "What problem do LSTMs and GRUs solve compared to basic RNNs?",
    "What did the lecturer say about quantum computing and its impact on machine learning?",
]

INK = "oklch(24% 0.006 250)"
INK_MUTED = "oklch(42% 0.012 250)"
PAPER = "oklch(98.2% 0.003 250)"
BORDER = "oklch(87% 0.007 250)"

GREEN = "oklch(34% 0.085 152)"
GREEN_WASH = "oklch(95.5% 0.025 152)"
BRICK = "oklch(40% 0.12 35)"
BRICK_WASH = "oklch(95.5% 0.03 35)"
AMBER = "oklch(46% 0.11 70)"
AMBER_WASH = "oklch(96% 0.03 70)"
NEUTRAL_WASH = "oklch(97% 0.003 250)"

SERIF = "'Lora', Georgia, 'Times New Roman', serif"
SANS = "'Inter', -apple-system, 'Segoe UI', sans-serif"

_pipeline = None
_active_transcript_label = SAMPLE_LABEL


def get_pipeline() -> RAGPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = RAGPipeline(INDEX_DIR)
    return _pipeline


def generate_examples(chunks, max_examples: int = 4) -> list[str]:
    topics = []
    for c in chunks:
        if c.topic not in topics:
            topics.append(c.topic)
    if not topics:
        return []
    if topics == ["Lecture"]:
        return ["What is this lecture about?"]
    return [f"What does the lecture say about {t.rstrip('.')}?" for t in topics[:max_examples]]


def load_transcript(file_obj):
    global _active_transcript_label
    if file_obj is None:
        return _status_html("Choose a .txt file first.", ok=False), gr.skip()

    transcript_path = file_obj if isinstance(file_obj, str) else file_obj.name
    display_name = os.path.basename(transcript_path)

    chunks = build_chunks(transcript_path)
    if not chunks:
        return _status_html(
            f"Couldn't find any text to index in '{display_name}'. "
            "Make sure the file isn't empty.", ok=False,
        ), gr.skip()

    work_dir = tempfile.mkdtemp(prefix="rag_ui_")
    index, _ = build_index(chunks)
    save_index(index, chunks, work_dir)

    get_pipeline().load_index(work_dir)
    _active_transcript_label = display_name

    topics = len({c.topic for c in chunks})
    topic_note = f"{topics} topic{'s' if topics != 1 else ''}" if topics > 1 else \
        "no section headings found, indexed as one section"
    status = _status_html(
        f"Loaded '{display_name}' &mdash; {len(chunks)} chunks, {topic_note}. "
        "Ask a question below, or try one of the new examples.", ok=True,
    )
    return status, [[q] for q in generate_examples(chunks)]


def reset_transcript():
    global _active_transcript_label
    get_pipeline().load_index(INDEX_DIR)
    _active_transcript_label = SAMPLE_LABEL
    status = _status_html(f"Back to the {SAMPLE_LABEL}.", ok=True)
    return status, [[q] for q in EXAMPLES]


def _status_html(message: str, ok: bool) -> str:
    colour = GREEN if ok else BRICK
    wash = GREEN_WASH if ok else BRICK_WASH
    return (
        f'<div style="border:1px solid {BORDER};border-radius:8px;'
        f'padding:10px 16px;background:{wash};color:{colour};'
        f'font-family:{SANS};font-weight:600;font-size:0.88rem;">{message}</div>'
    )


def _label_html() -> str:
    return (
        f'<div style="font-family:{SANS};font-size:0.85rem;color:{INK_MUTED};'
        f'margin:-4px 0 8px;">Currently loaded: '
        f'<strong>{html.escape(_active_transcript_label)}</strong></div>'
    )


def _score_colour(score: float) -> str:
    if score is None:
        return INK_MUTED
    if score >= 0.5:
        return GREEN
    if score >= 0.25:
        return AMBER
    return INK_MUTED


def _badge(label: str, value: str, colour: str) -> str:
    return (
        f'<span style="display:inline-block;padding:4px 12px;margin:2px 8px 2px 0;'
        f'border-radius:6px;background:{colour};color:{PAPER};'
        f'font-family:{SANS};font-weight:600;font-size:0.82rem;letter-spacing:0.01em;">'
        f'{html.escape(label)}: {html.escape(value)}</span>'
    )


def _card(heading: str, body: str, heading_colour: str, bg: str) -> str:
    return (
        f'<div style="border:1px solid {BORDER};border-radius:8px;'
        f'padding:18px 20px;margin:14px 0;background:{bg};line-height:1.6;">'
        f'<div style="font-family:{SANS};font-weight:700;font-size:0.78rem;'
        f'letter-spacing:0.04em;text-transform:uppercase;color:{heading_colour};'
        f'margin-bottom:8px;">{heading}</div>'
        f'{body}</div>'
    )


def render_result(result) -> str:
    parts = [f'<div style="max-width:820px;font-family:{SANS};color:{INK};">']

    if result.abstained:
        parts.append(_card(
            "Out of scope",
            f'<div style="color:{INK};font-size:1.02rem;">'
            f'{html.escape(result.rag_answer)}</div>',
            heading_colour=BRICK, bg=BRICK_WASH,
        ))
    else:
        fp = result.faithfulness_per_chunk
        fscore = fp.get("faithfulness_score")
        crate = fp.get("contradiction_rate")
        badges = _badge(
            "Faithfulness",
            f"{fscore:.0%}" if fscore is not None else "n/a",
            _score_colour(fscore),
        )
        if crate:
            badges += _badge("Contradiction", f"{crate:.0%}", BRICK)

        if result.low_confidence:
            heading = "Answer — low confidence, verify against the source"
            heading_colour, bg = AMBER, AMBER_WASH
            warning = (
                f'<div style="color:{AMBER};font-weight:600;font-size:0.88rem;'
                f'margin-bottom:10px;">This answer’s faithfulness score is very '
                f'low or it contains a likely contradiction. Check it against the '
                f'retrieved passages below before trusting it.</div>'
            )
        else:
            heading = "Answer — grounded in the lecture"
            heading_colour, bg = GREEN, GREEN_WASH
            warning = ""

        parts.append(_card(
            heading,
            f'{warning}'
            f'<div style="font-family:{SERIF};font-size:1.08rem;color:{INK};'
            f'margin-bottom:14px;">{html.escape(result.rag_answer)}</div>'
            f'<div>{badges}</div>',
            heading_colour=heading_colour, bg=bg,
        ))

    if result.retrieved:
        rows = []
        for r in result.retrieved:
            score = r.rerank_score if r.rerank_score is not None else r.embed_score
            snippet = html.escape(r.chunk.text[:280])
            rows.append(
                f'<div style="padding:12px 0;border-bottom:1px solid {BORDER};">'
                f'<div style="font-weight:600;color:{INK};font-size:0.92rem;">'
                f'{html.escape(r.chunk.topic)} '
                f'<span style="color:{INK_MUTED};font-weight:400;">'
                f'(chunk {r.chunk.chunk_id}, score {score:.2f})</span></div>'
                f'<div style="color:{INK_MUTED};font-size:0.9rem;margin-top:4px;">'
                f'{snippet}&hellip;</div></div>'
            )
        parts.append(
            f'<details style="margin:14px 0;">'
            f'<summary style="cursor:pointer;font-weight:600;color:{GREEN};'
            f'font-size:0.92rem;">'
            f'Show the {len(result.retrieved)} retrieved lecture passages</summary>'
            f'<div style="margin-top:10px;">{"".join(rows)}</div></details>'
        )

    parts.append(_card(
        "Same model without lecture access (baseline)",
        f'<div style="color:{INK_MUTED};font-family:{SERIF};font-size:1.0rem;">'
        f'{html.escape(result.baseline_answer)}</div>',
        heading_colour=INK_MUTED, bg=NEUTRAL_WASH,
    ))

    parts.append('</div>')
    return "".join(parts)


def answer_question(question: str) -> str:
    question = (question or "").strip()
    if not question:
        return _card(
            "Waiting for a question",
            f'<span style="color:{INK_MUTED};">Type a question about the '
            f'lecture above, or choose one of the examples.</span>',
            heading_colour=INK_MUTED, bg=NEUTRAL_WASH,
        )
    return render_result(get_pipeline().answer(question))


_GREEN_RAMP = gr.themes.Color(
    name="scholar_green",
    c50="#f0f7f2", c100="#dcece1", c200="#b9d9c5", c300="#8fc0a3",
    c400="#5fa17d", c500="#3d7f5c", c600="#2f6b4b", c700="#24573c",
    c800="#1c452f", c900="#163524", c950="#0f2519",
)
_NEUTRAL_RAMP = gr.themes.Color(
    name="scholar_neutral",
    c50="#fafafb", c100="#f2f3f5", c200="#e6e8eb", c300="#d3d6db",
    c400="#a8adb6", c500="#7c828d", c600="#5c6270", c700="#454b57",
    c800="#2e333d", c900="#1c2027", c950="#101318",
)

THEME = gr.themes.Base(
    primary_hue=_GREEN_RAMP,
    secondary_hue=_GREEN_RAMP,
    neutral_hue=_NEUTRAL_RAMP,
    font=[gr.themes.GoogleFont("Inter"), "-apple-system", "Segoe UI", "sans-serif"],
    font_mono=[gr.themes.GoogleFont("IBM Plex Mono"), "ui-monospace", "monospace"],
).set(
    body_background_fill="#fdfdfe",
    body_background_fill_dark="#fdfdfe",
    body_text_color="#1c2027",
    body_text_color_dark="#1c2027",
    block_background_fill="#ffffff",
    block_background_fill_dark="#ffffff",
    block_border_color="#e6e8eb",
    block_border_color_dark="#e6e8eb",
    block_label_text_color="#454b57",
    block_label_text_color_dark="#454b57",
    block_title_text_color="#1c2027",
    block_title_text_color_dark="#1c2027",
    border_color_primary="#d3d6db",
    border_color_primary_dark="#d3d6db",
    button_primary_background_fill="#2f6b4b",
    button_primary_background_fill_dark="#2f6b4b",
    button_primary_background_fill_hover="#24573c",
    button_primary_background_fill_hover_dark="#24573c",
    button_primary_text_color="#ffffff",
    button_primary_text_color_dark="#ffffff",
    button_primary_border_color="#2f6b4b",
    button_primary_border_color_dark="#2f6b4b",
    input_background_fill="#ffffff",
    input_background_fill_dark="#ffffff",
    input_border_color="#d3d6db",
    input_border_color_dark="#d3d6db",
    input_border_color_focus="#3d7f5c",
    input_border_color_focus_dark="#3d7f5c",
    checkbox_label_background_fill_selected="#dcece1",
    checkbox_label_background_fill_selected_dark="#dcece1",
    checkbox_label_text_color_selected="#163524",
    checkbox_label_text_color_selected_dark="#163524",
    checkbox_border_color_selected="#3d7f5c",
    checkbox_border_color_selected_dark="#3d7f5c",
    link_text_color="#2f6b4b",
    link_text_color_dark="#2f6b4b",
    link_text_color_hover="#24573c",
    link_text_color_hover_dark="#24573c",
    shadow_drop="none",
    shadow_drop_lg="none",
)

CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Lora:ital,wght@0,500;0,600;0,700;1,500&family=Inter:wght@400;500;600;700&display=swap');

:root, .dark, .gradio-container {
    color-scheme: light !important;
}
.gradio-container {
    font-family: 'Inter', -apple-system, 'Segoe UI', sans-serif !important;
    background: #fdfdfe !important;
}

#app-title h1 {
    font-family: 'Lora', Georgia, serif !important;
    font-weight: 600 !important;
    letter-spacing: -0.01em;
    color: #163524;
    text-wrap: balance;
}
#app-title p {
    color: #454b57;
    max-width: 68ch;
    line-height: 1.6;
}

#question-box textarea, #question-box input {
    font-size: 1rem !important;
    border-color: #d3d6db !important;
}
#question-box textarea:focus, #question-box input:focus {
    border-color: #3d7f5c !important;
    box-shadow: 0 0 0 1px #3d7f5c33 !important;
}

#example-chips .label {
    color: #454b57 !important;
    font-weight: 600 !important;
    font-size: 0.85rem !important;
}
#example-chips .gallery-item {
    border-color: #d3d6db !important;
    color: #1c2027 !important;
    transition: background-color 150ms ease-out, border-color 150ms ease-out;
}
#example-chips .gallery-item:hover {
    background: #f0f7f2 !important;
    border-color: #8fc0a3 !important;
}
#example-chips .gallery-item:focus-visible {
    outline: 2px solid #3d7f5c !important;
    outline-offset: 1px;
}

#transcript-panel {
    border-color: #e6e8eb !important;
}
#transcript-upload {
    border-color: #d3d6db !important;
}
#transcript-upload:hover {
    border-color: #8fc0a3 !important;
}

footer, .built-with { color: #a8adb6 !important; }

@media (prefers-reduced-motion: reduce) {
    * { transition-duration: 0.001ms !important; animation-duration: 0.001ms !important; }
}
"""


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="AI Education Assistant — Lecture Q&A") as demo:
        with gr.Column(elem_id="app-title"):
            gr.Markdown(
                "# AI Education Assistant\n"
                "Ask a question about a lecture transcript. Answers are drawn "
                "**only** from the transcript and shown with a faithfulness "
                "score; a no-retrieval baseline is shown for comparison."
            )

        with gr.Accordion("Change lecture transcript", open=False,
                          elem_id="transcript-panel"):
            gr.Markdown(
                "Upload a different `.txt` transcript to ask questions about "
                "it instead. Sections marked with `## Topic: <name>` are used "
                "for topic labelling; a plain transcript with no headings "
                "works too, it's just indexed as one section."
            )
            with gr.Row():
                transcript_file = gr.File(
                    label="Transcript (.txt)", file_types=[".txt"],
                    elem_id="transcript-upload", scale=3,
                )
                with gr.Column(scale=1):
                    load_btn = gr.Button("Load transcript")
                    reset_btn = gr.Button("Reset to sample lecture")
            transcript_status = gr.HTML()

        current_transcript = gr.HTML(_label_html())

        with gr.Row():
            question = gr.Textbox(
                label="Your question",
                placeholder="e.g. What is dropout and how does it help prevent overfitting?",
                scale=5, autofocus=True, elem_id="question-box",
            )
            ask_btn = gr.Button("Ask", variant="primary", scale=1)

        with gr.Column(elem_id="example-chips"):
            gr.Markdown("Try an example", elem_classes=["label"])
            example_ds = gr.Dataset(
                components=[question],
                samples=[[q] for q in EXAMPLES],
                type="values",
            )

        output = gr.HTML()

        def _load_and_label(file_obj):
            status, examples = load_transcript(file_obj)
            ds_update = gr.Dataset(samples=examples) if isinstance(examples, list) else gr.skip()
            return status, _label_html(), ds_update

        def _reset_and_label():
            status, examples = reset_transcript()
            return status, _label_html(), gr.Dataset(samples=examples)

        load_btn.click(fn=_load_and_label, inputs=transcript_file,
                       outputs=[transcript_status, current_transcript, example_ds])
        reset_btn.click(fn=_reset_and_label, inputs=None,
                        outputs=[transcript_status, current_transcript, example_ds])

        example_ds.click(fn=lambda sample: sample[0], inputs=example_ds, outputs=question)

        ask_btn.click(fn=answer_question, inputs=question, outputs=output)
        question.submit(fn=answer_question, inputs=question, outputs=output)

    return demo


if __name__ == "__main__":
    print("Loading UI (models load on the first question)...")
    build_ui().launch(theme=THEME, css=CUSTOM_CSS, server_name="127.0.0.1", server_port=7860)
