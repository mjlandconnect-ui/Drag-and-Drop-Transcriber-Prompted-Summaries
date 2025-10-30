"""CLI helper to transcribe a single audio file using the shared workflow."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import click

from gui_transcribe import ensure_prompt_library, load_prompt_text, transcribe_file


@click.command()
@click.argument("audio_path", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option(
    "--summarize/--no-summarize",
    default=True,
    show_default=True,
    help="Also run the summary prompt.",
)
@click.option(
    "--prompt",
    "prompt_name",
    default="General Summary",
    show_default=True,
    help="Name of the saved prompt to use.",
)
@click.option(
    "--prompt-text",
    default=None,
    help="Override prompt text instead of using the saved prompt.",
)
def main(audio_path: Path, summarize: bool, prompt_name: str, prompt_text: Optional[str]) -> None:
    """Transcribe AUDIO_PATH and save outputs under ./out/."""
    ensure_prompt_library()
    template = prompt_text if prompt_text is not None else load_prompt_text(prompt_name)

    outputs = transcribe_file(audio_path, summarize, template)

    click.echo(outputs.status_markdown)
    click.echo(f"Transcript: {outputs.txt_path}")
    click.echo(f"Captions: {outputs.srt_path}")
    click.echo(f"Verbose JSON: {outputs.json_path}")
    if outputs.summary_path:
        click.echo(f"Summary: {outputs.summary_path}")


if __name__ == "__main__":
    main()
