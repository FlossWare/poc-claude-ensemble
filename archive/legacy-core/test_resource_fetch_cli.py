from __future__ import annotations

from pathlib import Path
import subprocess
import sys

from resource_fetch import ResourceFetcher


def test_file_uri_percent_encoding_is_decoded_once(tmp_path: Path) -> None:
    source = tmp_path / "percent%20name.txt"
    source.write_text("encoded filename", encoding="utf-8")

    resource = ResourceFetcher().fetch(source.as_uri())

    assert resource.filename == "percent%20name.txt"
    assert resource.content == b"encoded filename"


def test_cli_invocation_fetches_local_file(tmp_path: Path) -> None:
    source = tmp_path / "cli.txt"
    source.write_text("CLI test content", encoding="utf-8")
    script = Path(__file__).resolve().parent / "tools" / "fetch-resource.py"

    result = subprocess.run(
        [sys.executable, str(script), source.as_uri()],
        cwd=script.parent.parent,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert result.stdout == "CLI test content"
    assert "fetched 16 bytes" in result.stderr
