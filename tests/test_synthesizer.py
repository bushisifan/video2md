import json

import pytest

from video2md.asr.sensevoice import Segment
from video2md.compose.schema import SOPDocument
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


def _make_synthesizer(responses):
    return StepSynthesizer(
        base_url="http://x", api_key="y", model="m", client=FakeClient(responses)
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
