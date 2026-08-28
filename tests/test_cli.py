from pathlib import Path

from video2md.compose.schema import SOPDocument
from video2md.config import Config
from video2md.pipeline import PipelineResult


def test_main_runs(tmp_path, monkeypatch):
    import video2md.cli as cli

    monkeypatch.setattr(cli.Config, "load", lambda p=None: Config())

    def fake_run(video, out, config, progress=None):
        Path(out).mkdir(parents=True, exist_ok=True)
        (Path(out) / "SOP.md").write_text("# done", encoding="utf-8")
        return PipelineResult(
            markdown_path=str(Path(out) / "SOP.md"),
            mermaid_path=str(Path(out) / "flowchart.mmd"),
            sop=SOPDocument(title="T"),
            frames_count=0,
            segments_count=0,
            step_windows_count=0,
            understanding_count=0,
            click_events_count=0,
        )

    monkeypatch.setattr(cli, "run_pipeline", fake_run)

    video = tmp_path / "in.mp4"
    video.write_bytes(b"x")
    rc = cli.main([str(video), "-o", str(tmp_path / "out")])
    assert rc == 0


def test_main_passes_config_path(tmp_path, monkeypatch):
    import video2md.cli as cli

    calls = []

    def fake_load(p=None):
        calls.append(p)
        return Config()

    monkeypatch.setattr(cli.Config, "load", fake_load)

    def fake_run(video, out, config, progress=None):
        Path(out).mkdir(parents=True, exist_ok=True)
        return PipelineResult(
            markdown_path=str(Path(out) / "SOP.md"),
            mermaid_path=str(Path(out) / "flowchart.mmd"),
            sop=SOPDocument(title="T"),
            frames_count=0, segments_count=0, step_windows_count=0, understanding_count=0, click_events_count=0,
        )

    monkeypatch.setattr(cli, "run_pipeline", fake_run)
    cfg_path = str(tmp_path / "my.yaml")
    video = tmp_path / "in.mp4"
    video.write_bytes(b"x")
    rc = cli.main([str(video), "-o", str(tmp_path / "out"), "-c", cfg_path])
    assert rc == 0
    assert calls == [cfg_path]


def test_main_bad_config_path_returns_nonzero(tmp_path, monkeypatch, capsys):
    import video2md.cli as cli

    monkeypatch.setattr(
        cli.Config, "load",
        lambda p=None: (_ for _ in ()).throw(FileNotFoundError("nope")),
    )
    video = tmp_path / "in.mp4"
    video.write_bytes(b"x")
    rc = cli.main([str(video), "-o", str(tmp_path / "out"), "-c", str(tmp_path / "nope.yaml")])
    assert rc == 1
    assert "错误" in capsys.readouterr().err


def test_main_pipeline_failure_returns_nonzero(tmp_path, monkeypatch, capsys):
    import video2md.cli as cli

    monkeypatch.setattr(cli.Config, "load", lambda p=None: Config())

    def boom(video, out, config, progress=None):
        raise RuntimeError("boom")

    monkeypatch.setattr(cli, "run_pipeline", boom)
    video = tmp_path / "in.mp4"
    video.write_bytes(b"x")
    rc = cli.main([str(video), "-o", str(tmp_path / "out")])
    assert rc == 1
    assert "错误" in capsys.readouterr().err


def test_main_export_calls_export_markdown(tmp_path, monkeypatch):
    import video2md.cli as cli
    from unittest import mock

    monkeypatch.setattr(cli.Config, "load", lambda p=None: Config())

    def fake_run(video, out, config, progress=None):
        Path(out).mkdir(parents=True, exist_ok=True)
        (Path(out) / "SOP.md").write_text("# done", encoding="utf-8")
        return PipelineResult(
            markdown_path=str(Path(out) / "SOP.md"),
            mermaid_path=str(Path(out) / "flowchart.mmd"),
            sop=SOPDocument(title="T"),
            frames_count=0, segments_count=0, step_windows_count=0, understanding_count=0, click_events_count=0,
        )

    monkeypatch.setattr(cli, "run_pipeline", fake_run)
    fake_export = mock.Mock(return_value=str(tmp_path / "out" / "SOP.pdf"))
    monkeypatch.setattr(cli, "export_markdown", fake_export)

    video = tmp_path / "in.mp4"
    video.write_bytes(b"x")
    rc = cli.main([str(video), "-o", str(tmp_path / "out"), "--export", "pdf"])
    assert rc == 0
    fake_export.assert_called_once()
