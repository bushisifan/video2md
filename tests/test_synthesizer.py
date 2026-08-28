import json

import pytest

from video2md.asr.sensevoice import Segment
from video2md.compose.schema import SOPDocument
from video2md.compose.step_detector import StepWindow
from video2md.compose.synthesizer import StepSynthesizer


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
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return FakeResponse(self.responses.pop(0))


class FakeChat:
    def __init__(self, responses):
        self.completions = FakeCompletions(responses)


class FakeClient:
    def __init__(self, responses):
        self.chat = FakeChat(responses)


def _make_synthesizer(responses, max_input_tokens=24000):
    return StepSynthesizer(
        base_url="http://x", api_key="y", model="m", client=FakeClient(responses),
        max_input_tokens=max_input_tokens,
    )


def test_synthesize_returns_document():
    content = json.dumps({"title": "T", "purpose": "P", "prerequisites": [], "steps": []})
    syn = _make_synthesizer([content])
    doc = syn.synthesize(
        segments=[Segment(0.0, 1.0, "打开设置")],
        frame_understandings=[{"timestamp": 1.0, "action": "click", "target": "设置"}],
        cursor_events=[],
    )
    assert isinstance(doc, SOPDocument)
    assert doc.title == "T"


def test_synthesize_chunks_and_merges_when_input_over_limit():
    long_text = "这是一段足够长的语音转写文本，用来触发合成输入超限切分，重复多次。" * 6
    segments = [
        Segment(0.0, 2.0, long_text),
        Segment(2.0, 4.0, long_text),
        Segment(4.0, 6.0, long_text),
    ]
    vis = [
        {"timestamp": 1.0, "action": "click", "target": "A"},
        {"timestamp": 3.0, "action": "click", "target": "B"},
        {"timestamp": 5.0, "action": "click", "target": "C"},
    ]
    windows = [
        StepWindow(order=1, title="打开A", start=0.0, end=2.0),
        StepWindow(order=2, title="打开B", start=2.0, end=4.0),
        StepWindow(order=3, title="打开C", start=4.0, end=6.0),
    ]
    parts = [
        json.dumps({
            "title": "T", "purpose": "P", "prerequisites": ["x"],
            "steps": [{"order": 1, "title": "a", "action": "doA"}],
            "troubleshooting": [{"issue": "q1", "solution": "s1"}],
        }),
        json.dumps({
            "title": "T2", "purpose": "P2", "prerequisites": [],
            "steps": [{"order": 1, "title": "b", "action": "doB"}],
            "troubleshooting": [{"issue": "q1", "solution": "s1"}, {"issue": "q2", "solution": "s2"}],
        }),
        json.dumps({
            "title": "T3", "purpose": "P3", "prerequisites": [],
            "steps": [{"order": 1, "title": "c", "action": "doC"}],
            "troubleshooting": [],
        }),
    ]
    syn = _make_synthesizer(parts, max_input_tokens=50)
    doc = syn.synthesize(
        segments=segments, frame_understandings=vis, cursor_events=[], step_windows=windows
    )
    assert [s.title for s in doc.steps] == ["a", "b", "c"]
    assert [s.order for s in doc.steps] == [1, 2, 3]
    assert doc.title == "T"                       # 元数据取首块
    assert len(doc.troubleshooting) == 2          # 疑难解答去重合并
    assert len(syn.client.chat.completions.calls) == 3  # 3 个时间窗 → 3 次分块调用


def test_synthesize_retries_then_raises():
    syn = _make_synthesizer(["not json", "also not json", "still not json"])
    with pytest.raises(RuntimeError, match="步骤合成失败"):
        syn.synthesize(segments=[], frame_understandings=[], cursor_events=[])


def test_synthesize_retries_then_recovers():
    valid = json.dumps({"title": "T", "purpose": "P", "prerequisites": [], "steps": []})
    syn = _make_synthesizer(["not json", valid])
    doc = syn.synthesize(segments=[], frame_understandings=[], cursor_events=[])
    assert isinstance(doc, SOPDocument)
    assert doc.title == "T"
    assert len(syn.client.chat.completions.calls) == 2


def test_synthesize_retries_on_schema_validation_error():
    schema_invalid = json.dumps(
        {"title": "T", "steps": [{"order": 0, "title": "a", "action": "b"}]}
    )
    valid = json.dumps({"title": "T", "purpose": "P", "prerequisites": [], "steps": []})
    syn = _make_synthesizer([schema_invalid, valid])
    doc = syn.synthesize(segments=[], frame_understandings=[], cursor_events=[])
    assert isinstance(doc, SOPDocument)
    assert doc.title == "T"
    assert len(syn.client.chat.completions.calls) == 2
