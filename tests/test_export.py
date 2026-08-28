import shutil
import types

import pytest

from video2md.render.export import export_markdown


def test_export_skips_when_pandoc_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(shutil, "which", lambda name: None)
    md = tmp_path / "SOP.md"
    md.write_text("# x", encoding="utf-8")
    with pytest.raises(RuntimeError, match="pandoc"):
        export_markdown(str(md), "pdf")


def test_export_rejects_unsupported_format(tmp_path):
    md = tmp_path / "SOP.md"
    md.write_text("# x", encoding="utf-8")
    with pytest.raises(ValueError, match="Unsupported format"):
        export_markdown(str(md), "txt")


def test_export_raises_on_pandoc_failure(tmp_path, monkeypatch):
    md = tmp_path / "SOP.md"
    md.write_text("# x", encoding="utf-8")

    def fake_run(*args, **kwargs):
        return types.SimpleNamespace(returncode=1, stderr="boom")

    monkeypatch.setattr("video2md.render.export.shutil.which", lambda name: "/usr/bin/pandoc")
    monkeypatch.setattr("video2md.render.export.subprocess.run", fake_run)
    with pytest.raises(RuntimeError, match="pandoc failed"):
        export_markdown(str(md), "pdf")
