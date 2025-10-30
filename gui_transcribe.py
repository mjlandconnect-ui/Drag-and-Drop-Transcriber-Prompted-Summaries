"""Gradio application for drag-and-drop transcription and optional summarization."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple

import gradio as gr
from openai import OpenAI

OUT_DIR = Path("out")
PROMPTS_PATH = Path("prompts.json")
TRANSCRIPTION_MODEL = "whisper-1"
SUMMARY_MODEL = "gpt-4o-mini"

DEFAULT_PROMPTS: Dict[str, str] = {
    "General Summary": (
        "You are an executive assistant. Provide 5-10 concise bullet points summarizing the conversation. "
        "Focus on decisions, action items, deadlines, and unresolved questions. Include owners when possible.\n{transcript}"
    ),
    "LB Update (one line)": (
        "Produce a single-line status update no longer than 300 characters covering current status, blockers, "
        "and the next planned step. Do not add bullet points or labels.\n{transcript}"
    ),
    "Radiology Downtime (Ops)": (
        "Summarize the incident for hospital operations leadership. Highlight impact, timeline, workarounds, "
        "communication points, and next actions. Keep it concise and actionable.\n{transcript}"
    ),
    "Land Listing Summary": (
        "Imagine you are briefing a buyer's agent about a new property listing. Provide the buyer persona, top "
        "reasons to care, risks, next actions, and 3-5 attention-grabbing headlines (each 34 characters or fewer). "
        "{transcript}"
    ),
}


def ensure_prompt_library() -> Dict[str, str]:
    """Ensure the prompt library exists on disk and return its contents."""
    if not PROMPTS_PATH.exists():
        PROMPTS_PATH.write_text(json.dumps(DEFAULT_PROMPTS, indent=2), encoding="utf-8")
    with PROMPTS_PATH.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    return {str(k): str(v) for k, v in data.items()}


def save_prompt(prompt_name: str, prompt_text: str) -> None:
    prompts = ensure_prompt_library()
    prompts[prompt_name] = prompt_text
    PROMPTS_PATH.write_text(json.dumps(prompts, indent=2), encoding="utf-8")


def load_prompt_text(prompt_name: str) -> str:
    prompts = ensure_prompt_library()
    return prompts.get(prompt_name, "")


def require_api_key() -> str:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY environment variable is not set.")
    return api_key


def timestamped_basename(source_path: Path) -> str:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    base = source_path.stem
    return f"{base}-{timestamp}"


def seconds_to_srt_timestamp(seconds: float) -> str:
    millis = int(round(seconds * 1000))
    hours, remainder = divmod(millis, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    secs, millis = divmod(remainder, 1000)
    return f"{hours:02}:{minutes:02}:{secs:02},{millis:03}"


@dataclass
class TranscriptionOutputs:
    transcript_text: str
    summary_text: str
    txt_path: Optional[Path]
    srt_path: Optional[Path]
    json_path: Optional[Path]
    summary_path: Optional[Path]
    status_markdown: str


def write_transcription_outputs(transcription: Dict, base_name: str) -> Tuple[str, Path, Path, Path]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    transcript_text = transcription.get("text", "").strip()

    txt_path = OUT_DIR / f"{base_name}.txt"
    txt_path.write_text(transcript_text + "\n", encoding="utf-8")

    json_path = OUT_DIR / f"{base_name}.json"
    json_path.write_text(json.dumps(transcription, indent=2), encoding="utf-8")

    srt_lines = []
    segments = transcription.get("segments") or []
    for idx, segment in enumerate(segments, start=1):
        start_ts = seconds_to_srt_timestamp(float(segment.get("start", 0)))
        end_ts = seconds_to_srt_timestamp(float(segment.get("end", 0)))
        text = str(segment.get("text", "")).strip()
        if not text:
            continue
        srt_lines.extend([str(idx), f"{start_ts} --> {end_ts}", text, ""])

    srt_path = OUT_DIR / f"{base_name}.srt"
    srt_body = "\n".join(srt_lines).strip()
    srt_path.write_text((srt_body + "\n") if srt_body else "", encoding="utf-8")

    return transcript_text, txt_path, srt_path, json_path


def summarize_transcript(client: OpenAI, prompt_template: str, transcript_text: str) -> str:
    prompt_template = prompt_template or ""
    cleaned_transcript = transcript_text.strip()
    if "{transcript}" in prompt_template:
        prompt = prompt_template.replace("{transcript}", cleaned_transcript)
    else:
        prompt = f"{prompt_template.strip()}\n\nTranscript:\n{cleaned_transcript}"

    response = client.responses.create(
        model=SUMMARY_MODEL,
        input=prompt,
    )
    return response.output_text.strip()


def transcribe_file(
    file_path: Path,
    summarize: bool,
    prompt_template: str,
) -> TranscriptionOutputs:
    if not file_path:
        raise RuntimeError("Please upload an audio file to transcribe.")

    api_key = require_api_key()
    client = OpenAI(api_key=api_key)

    if not file_path.exists():
        raise RuntimeError("Uploaded file is unavailable. Please try again.")

    base_name = timestamped_basename(file_path)

    with file_path.open("rb") as audio_file:
        transcription = client.audio.transcriptions.create(
            model=TRANSCRIPTION_MODEL,
            file=audio_file,
            response_format="verbose_json",
        )

    transcription_dict = transcription.model_dump()
    transcript_text, txt_path, srt_path, json_path = write_transcription_outputs(
        transcription_dict, base_name
    )

    summary_text = ""
    summary_path: Optional[Path] = None
    if summarize:
        if not prompt_template.strip():
            raise RuntimeError("Selected prompt is empty. Please provide prompt text or disable summarization.")
        summary_text = summarize_transcript(client, prompt_template, transcript_text)
        summary_path = OUT_DIR / f"{base_name}-summary.txt"
        summary_path.write_text(summary_text + "\n", encoding="utf-8")

    status = (
        "✅ Transcription complete." if not summarize else "✅ Transcription and summary complete."
    )
    return TranscriptionOutputs(
        transcript_text=transcript_text,
        summary_text=summary_text,
        txt_path=txt_path,
        srt_path=srt_path,
        json_path=json_path,
        summary_path=summary_path,
        status_markdown=status,
    )


def transcribe_from_ui(
    audio_file,
    summarize: bool,
    prompt_name: str,
    prompt_text: str,
):
    if audio_file is None:
        return "⚠️ Please upload an audio file.", "", "", None, None, None, None

    try:
        result = transcribe_file(Path(audio_file.name), summarize, prompt_text)
    except Exception as exc:
        message = f"❌ {exc}"
        return message, "", "", None, None, None, None

    return (
        result.status_markdown,
        result.transcript_text,
        result.summary_text,
        str(result.txt_path) if result.txt_path else None,
        str(result.srt_path) if result.srt_path else None,
        str(result.json_path) if result.json_path else None,
        str(result.summary_path) if result.summary_path else None,
    )


def on_prompt_selected(prompt_name: str) -> Tuple[str, str]:
    text = load_prompt_text(prompt_name) if prompt_name else ""
    helper = ""
    if text:
        helper = f"Loaded prompt: **{prompt_name}**"
    return text, helper


def on_prompt_save(prompt_name: str, prompt_text: str) -> str:
    if not prompt_name:
        return "❌ Please provide a prompt name."
    if not prompt_text.strip():
        return "❌ Prompt text cannot be empty."
    save_prompt(prompt_name, prompt_text)
    return f"✅ Saved prompt '{prompt_name}'."


def build_interface() -> gr.Blocks:
    prompts = ensure_prompt_library()
    prompt_names = list(prompts.keys())
    initial_prompt_name = prompt_names[0] if prompt_names else ""
    initial_prompt_text = prompts.get(initial_prompt_name, "")

    with gr.Blocks(title="Drag-and-Drop Transcriber + Summaries") as demo:
        gr.Markdown(
            "# Drag-and-Drop Transcriber\n"
            "Drop an audio file, then click **Transcribe** to generate text, SRT, and JSON outputs."
        )

        with gr.Row():
            audio_input = gr.File(
                label="Audio file (mp3/wav/m4a)",
                file_types=[".mp3", ".wav", ".m4a", ".mp4"],
            )
            summarize_checkbox = gr.Checkbox(
                label="Also summarize with selected prompt",
                value=False,
            )

        with gr.Row():
            prompt_dropdown = gr.Dropdown(
                label="Saved prompts",
                choices=prompt_names,
                value=initial_prompt_name,
                allow_custom_value=True,
            )
            prompt_status = gr.Markdown("", elem_classes=["prompt-status"])

        prompt_editor = gr.TextArea(
            label="Prompt editor",
            value=initial_prompt_text,
            placeholder="Write or edit the prompt text here. Use {transcript} to insert the transcript.",
            lines=10,
        )
        save_prompt_button = gr.Button("Save Prompt")

        status = gr.Markdown("Ready.")

        transcribe_button = gr.Button("Transcribe", variant="primary")

        transcript_preview = gr.Textbox(
            label="Transcript preview",
            lines=12,
        )
        summary_preview = gr.Textbox(
            label="Summary preview",
            lines=10,
        )

        with gr.Row():
            txt_download = gr.File(label="Download transcript (.txt)")
            srt_download = gr.File(label="Download captions (.srt)")
            json_download = gr.File(label="Download verbose JSON")
            summary_download = gr.File(label="Download summary (.txt)")

        prompt_dropdown.change(
            on_prompt_selected,
            inputs=prompt_dropdown,
            outputs=[prompt_editor, prompt_status],
        )
        save_prompt_button.click(
            on_prompt_save,
            inputs=[prompt_dropdown, prompt_editor],
            outputs=prompt_status,
        )
        transcribe_button.click(
            fn=transcribe_from_ui,
            inputs=[audio_input, summarize_checkbox, prompt_dropdown, prompt_editor],
            outputs=[
                status,
                transcript_preview,
                summary_preview,
                txt_download,
                srt_download,
                json_download,
                summary_download,
            ],
        )

    return demo


def main() -> None:
    demo = build_interface()
    demo.launch()


if __name__ == "__main__":
    main()
