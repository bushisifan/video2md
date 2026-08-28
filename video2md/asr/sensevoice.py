"""Chinese ASR wrapper (local, via FunASR).

FunASR is imported lazily inside `_load_model` so the rest of the package and
all unit tests work without `funasr`/`modelscope` installed.

Sentence-level timestamps (语音驱动抽帧) require paraformer-zh with vad/punc
models and `sentence_timestamp=True`:
    SenseVoiceTranscriber(model="paraformer-zh", vad_model="fsmn-vad",
                          punc_model="ct-punc-c", sentence_timestamp=True)
NOTE: funasr returns `sentence_info` start/end in MILLISECONDS; `_parse`
converts them to seconds.
"""
import re
from dataclasses import dataclass
from typing import List, Optional

# Module-level placeholder filled lazily on first `_load_model` call. Exposed
# so tests can monkeypatch `video2md.asr.sensevoice.AutoModel` without funasr
# installed; kept out of the public API.
AutoModel = None


@dataclass
class Segment:
    start: float
    end: float
    text: str


class SenseVoiceTranscriber:
    def __init__(
        self,
        model: str = "iic/SenseVoiceSmall",
        device: str = "cpu",
        sample_rate: int = 16000,
        vad_model: Optional[str] = None,
        punc_model: Optional[str] = None,
        sentence_timestamp: bool = False,
    ):
        self.model_name = model
        self.device = device
        # FunASR 从 WAV 头读取采样率；此处仅为配置对称保留
        self.sample_rate = sample_rate
        self.vad_model = vad_model
        self.punc_model = punc_model
        self.sentence_timestamp = sentence_timestamp
        self._model = None

    def _load_model(self):
        global AutoModel
        if self._model is None:
            if AutoModel is None:
                from funasr import AutoModel as _AutoModel  # lazy import

                AutoModel = _AutoModel
            kwargs = {"model": self.model_name, "device": self.device}
            if self.vad_model:
                kwargs["vad_model"] = self.vad_model
            if self.punc_model:
                kwargs["punc_model"] = self.punc_model
            self._model = AutoModel(**kwargs)
        return self._model

    def transcribe(self, audio_path: str) -> List[Segment]:
        model = self._load_model()
        kwargs = {"input": audio_path, "language": "zh"}
        if self.sentence_timestamp:
            kwargs["sentence_timestamp"] = True
        result = model.generate(**kwargs)
        return self._parse(result)

    def _parse(self, result: list) -> List[Segment]:
        """Normalize FunASR output into Segment list.

        Prefers per-sentence timestamps when present; otherwise falls back to a
        single segment with the whole transcript.
        """
        segments: List[Segment] = []
        if not result:
            return segments
        rec = result[0]
        text = self._clean(rec.get("text", "")) if isinstance(rec, dict) else ""
        sentence_info = rec.get("sentence_info", []) if isinstance(rec, dict) else []
        if sentence_info:
            for s in sentence_info:
                t = self._clean(s.get("text", ""))
                if t:
                    segments.append(
                        Segment(
                            # funasr sentence_info 时间戳单位为毫秒，转成秒
                            start=float(s.get("start", 0)) / 1000,
                            end=float(s.get("end", 0)) / 1000,
                            text=t,
                        )
                    )
        elif text:
            segments.append(Segment(start=0.0, end=0.0, text=text))
        if not segments and text:
            segments.append(Segment(start=0.0, end=0.0, text=text))
        return segments

    @staticmethod
    def _clean(text: str) -> str:
        return re.sub(r"<\|[^|]+\|>", "", text).strip()
