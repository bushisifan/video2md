from video2md.asr.sensevoice import Segment, SenseVoiceTranscriber


def test_clean_strips_tags():
    assert (
        SenseVoiceTranscriber._clean("<|zh|>首先打开设置<|withitn|>")
        == "首先打开设置"
    )
    assert (
        SenseVoiceTranscriber._clean("<|zh|><|nospeech|>先打开设置<|Speech|>")
        == "先打开设置"
    )


def test_parse_single_segment():
    t = SenseVoiceTranscriber(model="fake", device="cpu")
    result = [{"text": "<|zh|>你好<|withitn|>"}]
    segments = t._parse(result)
    assert len(segments) == 1
    assert segments[0].text == "你好"


def test_parse_with_sentence_info():
    t = SenseVoiceTranscriber(model="fake", device="cpu")
    result = [{
        "text": "<|zh|>先打开设置然后点击保存<|withitn|>",
        # funasr sentence_info 时间戳单位是毫秒，_parse 应转成秒
        "sentence_info": [
            {"start": 270.0, "end": 2970.0, "text": "<|zh|>先打开设置<|withitn|>"},
            {"start": 3200.0, "end": 8100.0, "text": "<|zh|>然后点击保存<|withitn|>"},
        ],
    }]
    segments = t._parse(result)
    assert len(segments) == 2
    assert segments[0].start == 0.27
    assert segments[0].end == 2.97
    assert segments[1].text == "然后点击保存"


def test_transcribe_passes_sentence_timestamp(monkeypatch):
    calls = {}

    class FakeModel:
        def generate(self, **kwargs):
            calls.update(kwargs)
            return [{"text": "测试"}]

    def fake_auto_model(**kwargs):
        return FakeModel()

    monkeypatch.setattr("video2md.asr.sensevoice.AutoModel", fake_auto_model)
    t = SenseVoiceTranscriber(model="fake", device="cpu", sentence_timestamp=True)
    t.transcribe("audio.wav")
    assert calls.get("sentence_timestamp") is True
    assert calls.get("input") == "audio.wav"


def test_load_model_passes_vad_punc(monkeypatch):
    captured = {}

    class FakeModel:
        def generate(self, **kwargs):
            return [{"text": "x"}]

    def fake_auto_model(**kwargs):
        captured.update(kwargs)
        return FakeModel()

    monkeypatch.setattr("video2md.asr.sensevoice.AutoModel", fake_auto_model)
    t = SenseVoiceTranscriber(
        model="paraformer-zh", device="cpu",
        vad_model="fsmn-vad", punc_model="ct-punc-c",
    )
    t.transcribe("audio.wav")
    assert captured.get("model") == "paraformer-zh"
    assert captured.get("vad_model") == "fsmn-vad"
    assert captured.get("punc_model") == "ct-punc-c"


def test_transcribe_uses_model(monkeypatch):
    class FakeModel:
        def generate(self, input, language):
            return [{"text": "<|zh|>打开了设置<|withitn|>"}]

    def fake_auto_model(**kwargs):
        return FakeModel()

    monkeypatch.setattr("video2md.asr.sensevoice.AutoModel", fake_auto_model)
    t = SenseVoiceTranscriber(model="fake", device="cpu")
    segments = t.transcribe("audio.wav")
    assert segments[0].text == "打开了设置"


def test_parse_empty_returns_empty():
    t = SenseVoiceTranscriber(model="fake", device="cpu")
    assert t._parse([]) == []


def test_parse_falls_back_when_sentence_info_empty():
    t = SenseVoiceTranscriber(model="fake", device="cpu")
    result = [{
        "text": "<|zh|>完整文本<|withitn|>",
        "sentence_info": [{"start": 0.0, "end": 1.0, "text": "<|zh|><|nospeech|><|withitn|>"}],
    }]
    segments = t._parse(result)
    assert len(segments) == 1
    assert segments[0].text == "完整文本"
