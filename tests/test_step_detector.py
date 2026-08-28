import json

import pytest

from video2md.asr.sensevoice import Segment
from video2md.compose.step_detector import StepDetector, StepWindow


class FakeMessage:
    def __init__(self, content):
        self.content = content


class FakeChoice:
    def __init__(self, content):
        self.message = FakeMessage(content)


class FakeResponse:
    def __init__(self, content):
        self.choices = [FakeChoice(content)]


class FakeCompletions:
    def __init__(self, responses):
        self.responses = list(responses)

    def create(self, **kwargs):
        return FakeResponse(self.responses.pop(0))


class FakeChat:
    def __init__(self, responses):
        self.completions = FakeCompletions(responses)


class FakeClient:
    def __init__(self, responses):
        self.chat = FakeChat(responses)


def _detector(responses, max_input_tokens=24000):
    return StepDetector(
        base_url="http://x", api_key="y", model="m", client=FakeClient(responses),
        max_input_tokens=max_input_tokens,
    )


def _segments():
    return [
        Segment(0.27, 2.97, "一条视频教会你"),
        Segment(3.05, 5.37, "把多个工具"),
        Segment(5.39, 6.45, "统一管理"),
    ]


def test_detect_returns_windows():
    content = json.dumps({
        "steps": [
            {"order": 1, "title": "安装工具", "start": 0.27, "end": 3.05},
            {"order": 2, "title": "统一管理", "start": 3.05, "end": 6.45},
        ]
    })
    det = _detector([content])
    windows = det.detect(_segments())
    assert len(windows) == 2
    assert isinstance(windows[0], StepWindow)
    assert windows[0].title == "安装工具"
    assert windows[1].start == 3.05


def test_detect_snaps_to_segment_boundaries():
    # LLM 给出边界之外的时间，应吸附到最近转写片段边界
    content = json.dumps({
        "steps": [
            {"order": 1, "title": "步骤A", "start": 0.1, "end": 3.1},
        ]
    })
    det = _detector([content])
    windows = det.detect(_segments())
    # 0.1 → 最近边界 0.27；3.1 → 最近边界 3.05
    assert windows[0].start == 0.27
    assert windows[0].end == 3.05


def test_detect_chunks_when_input_over_limit():
    long_text = "这是一段足够长的语音转写文本，用来触发输入超限切分，重复多次。" * 6
    segments = [
        Segment(0.0, 2.0, long_text),
        Segment(2.0, 4.0, long_text),
        Segment(4.0, 6.0, long_text),
    ]
    contents = [
        json.dumps({"steps": [{"order": 1, "title": "第一步", "start": 0.0, "end": 2.0}]}),
        json.dumps({"steps": [{"order": 1, "title": "第二步", "start": 2.0, "end": 4.0}]}),
        json.dumps({"steps": [{"order": 1, "title": "第三步", "start": 4.0, "end": 6.0}]}),
    ]
    det = _detector(contents, max_input_tokens=50)
    windows = det.detect(segments)
    assert [w.title for w in windows] == ["第一步", "第二步", "第三步"]
    assert [w.order for w in windows] == [1, 2, 3]
    # 3 段 → 3 块 → 3 次调用（全部响应已消费，证明走了切分）
    assert len(det.client.chat.completions.responses) == 0


def test_detect_retries_then_raises():
    det = _detector(["not json", "not json", "not json"])
    with pytest.raises(RuntimeError, match="步骤切分失败"):
        det.detect(_segments())
