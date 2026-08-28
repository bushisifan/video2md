from pathlib import Path

from video2md.asr.sensevoice import Segment
from video2md.compose.schema import SOPDocument
from video2md.compose.step_detector import StepWindow
from video2md.config import Config
from video2md.pipeline import PipelineResult, run_pipeline
from video2md.preprocess.frames import KeyFrame


class FakeExtractor:
    def __init__(self, **kwargs):
        pass

    def _write(self, output_dir, ts):
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        (Path(output_dir) / "frame_0001.png").write_bytes(b"png")
        return [KeyFrame(timestamp=ts, image_path=f"{output_dir}/frame_0001.png")]

    def extract(self, video_path, output_dir):
        return self._write(output_dir, 1.0)

    def extract_at_timestamps(self, video_path, timestamps, output_dir):
        return self._write(output_dir, timestamps[0] if timestamps else 0.0)


class FakeTranscriber:
    def __init__(self, **kwargs):
        pass

    def transcribe(self, audio_path):
        return [Segment(start=0.0, end=1.0, text="打开设置")]


class FakeStepDetector:
    def __init__(self, **kwargs):
        pass

    def detect(self, segments):
        return [StepWindow(order=1, title="打开设置", start=0.0, end=1.0)]


class FakeVision:
    def __init__(self, **kwargs):
        pass

    def understand_frame(self, image_path, timestamp):
        return {
            "action": "click",
            "target": "设置",
            "detail": "点击设置",
            "needs_review": False,
            "timestamp": timestamp,
            "screenshot": image_path,
        }


class FakeSynthesizer:
    def __init__(self, **kwargs):
        pass

    def synthesize(self, segments, understandings, cursor_events, step_windows=None):
        return SOPDocument(title="测试SOP", purpose="目的", steps=[], troubleshooting=[])


def _patch_common(monkeypatch, tmp_path):
    monkeypatch.setattr("video2md.pipeline.FrameExtractor", FakeExtractor)
    monkeypatch.setattr("video2md.pipeline.SenseVoiceTranscriber", FakeTranscriber)
    monkeypatch.setattr("video2md.pipeline.StepDetector", FakeStepDetector)
    monkeypatch.setattr("video2md.pipeline.VisionClient", FakeVision)
    monkeypatch.setattr("video2md.pipeline.StepSynthesizer", FakeSynthesizer)
    monkeypatch.setattr("video2md.pipeline.CursorDetector", lambda **k: None)
    monkeypatch.setattr(
        "video2md.pipeline.extract_audio",
        lambda *a, **k: str(tmp_path / "out" / "audio.wav"),
    )


def test_run_pipeline(tmp_path, monkeypatch):
    _patch_common(monkeypatch, tmp_path)

    (tmp_path / "out").mkdir(parents=True, exist_ok=True)
    (tmp_path / "out" / "audio.wav").write_bytes(b"wav")
    video = tmp_path / "in.mp4"
    video.write_bytes(b"fake")

    cfg = Config()
    cfg.cursor.enabled = False

    result = run_pipeline(str(video), str(tmp_path / "out"), cfg)

    assert isinstance(result, PipelineResult)
    assert (tmp_path / "out" / "SOP.md").exists()
    assert (tmp_path / "out" / "flowchart.mmd").exists()
    assert "测试SOP" in (tmp_path / "out" / "SOP.md").read_text(encoding="utf-8")
    assert result.frames_count == 1
    assert result.segments_count == 1
    assert result.step_windows_count == 1
    assert result.understanding_count == 1


def test_run_pipeline_vision_failure_degrades(tmp_path, monkeypatch):
    class BoomVision:
        def __init__(self, **kwargs):
            pass

        def understand_frame(self, image_path, timestamp):
            raise RuntimeError("conn reset")

    _patch_common(monkeypatch, tmp_path)
    monkeypatch.setattr("video2md.pipeline.VisionClient", BoomVision)

    (tmp_path / "out").mkdir(parents=True, exist_ok=True)
    (tmp_path / "out" / "audio.wav").write_bytes(b"wav")
    video = tmp_path / "in.mp4"
    video.write_bytes(b"fake")

    cfg = Config()
    cfg.cursor.enabled = False

    result = run_pipeline(str(video), str(tmp_path / "out"), cfg)
    assert (tmp_path / "out" / "SOP.md").exists()
    assert result.understanding_count == 1


def test_run_pipeline_asr_failure_degrades(tmp_path, monkeypatch):
    class FailTranscriber:
        def __init__(self, **kwargs):
            pass

        def transcribe(self, audio_path):
            raise RuntimeError("model failed")

    _patch_common(monkeypatch, tmp_path)
    monkeypatch.setattr("video2md.pipeline.SenseVoiceTranscriber", FailTranscriber)

    (tmp_path / "out").mkdir(parents=True, exist_ok=True)
    (tmp_path / "out" / "audio.wav").write_bytes(b"wav")
    video = tmp_path / "in.mp4"
    video.write_bytes(b"fake")

    cfg = Config()
    cfg.cursor.enabled = False

    result = run_pipeline(str(video), str(tmp_path / "out"), cfg)
    assert (tmp_path / "out" / "SOP.md").exists()
    assert result.segments_count == 0
    assert result.step_windows_count == 0


def test_run_pipeline_silent_video(tmp_path, monkeypatch):
    _patch_common(monkeypatch, tmp_path)
    monkeypatch.setattr(
        "video2md.pipeline.extract_audio",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("no audio")),
    )

    video = tmp_path / "in.mp4"
    video.write_bytes(b"fake")

    cfg = Config()
    cfg.cursor.enabled = False

    result = run_pipeline(str(video), str(tmp_path / "out"), cfg)
    assert (tmp_path / "out" / "SOP.md").exists()
    assert result.segments_count == 0
    assert result.step_windows_count == 0


def test_run_pipeline_sanitizes_screenshots(tmp_path, monkeypatch):
    class HallucinatingSynthesizer:
        def __init__(self, **kwargs):
            pass

        def synthesize(self, segments, understandings, cursor_events, step_windows=None):
            return SOPDocument.model_validate({
                "title": "T",
                "steps": [{
                    "order": 1,
                    "title": "A",
                    "action": "a",
                    "screenshot": "images/nope.png",
                    "branch": {"condition": "c", "children": [
                        {"order": 2, "title": "B", "action": "b", "screenshot": "images/frame_0001.png"}
                    ]},
                }],
            })

    _patch_common(monkeypatch, tmp_path)
    monkeypatch.setattr("video2md.pipeline.StepSynthesizer", HallucinatingSynthesizer)

    (tmp_path / "out").mkdir(parents=True, exist_ok=True)
    (tmp_path / "out" / "audio.wav").write_bytes(b"wav")
    video = tmp_path / "in.mp4"
    video.write_bytes(b"fake")

    cfg = Config()
    cfg.cursor.enabled = False

    result = run_pipeline(str(video), str(tmp_path / "out"), cfg)
    md = (tmp_path / "out" / "SOP.md").read_text(encoding="utf-8")
    assert "images/nope.png" not in md
    assert "images/frame_0001.png" in md
    assert result.sop.steps[0].screenshot == ""


def test_run_pipeline_step_detection_failure_falls_back(tmp_path, monkeypatch):
    """步骤切分失败 → 退化为场景变化抽帧（extract），管线不中断。"""

    class FailStepDetector:
        def __init__(self, **kwargs):
            pass

        def detect(self, segments):
            raise RuntimeError("detection failed")

    _patch_common(monkeypatch, tmp_path)
    monkeypatch.setattr("video2md.pipeline.StepDetector", FailStepDetector)

    (tmp_path / "out").mkdir(parents=True, exist_ok=True)
    (tmp_path / "out" / "audio.wav").write_bytes(b"wav")
    video = tmp_path / "in.mp4"
    video.write_bytes(b"fake")

    cfg = Config()
    cfg.cursor.enabled = False

    result = run_pipeline(str(video), str(tmp_path / "out"), cfg)
    assert (tmp_path / "out" / "SOP.md").exists()
    assert result.step_windows_count == 0
    assert result.frames_count == 1
