# Drag-and-Drop Transcriber + Prompted Summaries

Drop a single audio file into a local Gradio app to generate a transcript (TXT, SRT, JSON) and an optional summary using reusable prompts.

![App screenshot placeholder](docs/screenshot-placeholder.png)

## Features

- 🔄 **Drag-and-drop transcription** using OpenAI Whisper (`whisper-1`).
- 📝 **Multiple output formats**: plain text, SRT captions, verbose JSON.
- 🧠 **Reusable prompts** stored in `prompts.json` with inline editor.
- ✍️ **Optional summaries** powered by GPT (`gpt-4o-mini`).
- 💾 **Deterministic storage** under `./out/NAME-timestamp.*` with download buttons.

## Prerequisites

- Python 3.12+
- An OpenAI API key with access to `whisper-1` and `gpt-4o-mini`.
- `ffmpeg` installed and accessible in your `PATH` (required by Whisper for some formats).

## Installation (Windows)

```powershell
# Clone this repository and navigate into it first
python -m venv .venv
.\.venv\Scripts\activate
pip install -U pip
pip install -r requirements.txt
setx OPENAI_API_KEY "your_api_key_here"
```

> ℹ️ Restart your terminal after setting the environment variable so Windows picks it up.

## Running the Gradio App

```powershell
.\.venv\Scripts\activate
python gui_transcribe.py
```

Open the printed URL (default `http://127.0.0.1:7860`) in your browser.

1. Drag a single audio file (`.mp3`, `.wav`, `.m4a`, `.mp4`) into the drop zone.
2. (Optional) Tick **Also summarize with selected prompt**.
3. Pick a saved prompt, tweak it in the editor, and click **Save Prompt** to persist.
4. Click **Transcribe**.
5. Download the generated files from the download boxes or copy previews.

Outputs are saved under `./out/` as:

- `NAME-timestamp.txt`
- `NAME-timestamp.srt`
- `NAME-timestamp.json`
- `NAME-timestamp-summary.txt` (if summarization is enabled)

## Command-Line Runner

For scripted workflows:

```powershell
python transcribe_cloud.py path\to\audio.mp3 --summarize --prompt "General Summary"
```

Use `--no-summarize` to skip the summary or `--prompt-text` to pass ad-hoc prompt text.

## Prompt Library

Prompts live in `prompts.json` as a simple dictionary of `{ "Prompt Name": "Prompt text with optional {transcript}" }`.

Default prompts ship with:

- General Summary – 5–10 bullets with decisions and actions.
- LB Update (one line) – Single line, ≤300 characters.
- Radiology Downtime (Ops) – Operational recap and next steps.
- Land Listing Summary – Buyer persona, highlights, risks, next steps, and short headlines.

If `{transcript}` appears in a prompt it will be replaced with the raw transcript. Otherwise the transcript is appended under a `Transcript:` section automatically.

## Error Handling

- The app checks that `OPENAI_API_KEY` is set before calling the API.
- Friendly error banners surface API or validation issues (missing files, empty prompts, etc.).
- Each run writes files with timestamps, preventing accidental overwrites.

## Roadmap Ideas

- Batch uploads with ZIP export
- Project-based prompt presets
- Text post-processing options
- Privacy toggles for redaction
- GitHub Actions smoke tests

## License

MIT
