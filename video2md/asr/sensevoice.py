"""中文语音转写封装（本地，基于 FunASR）。

`_load_model` 内部惰性导入 FunASR，这样包的其他部分与全部单元测试在未安装
`funasr`/`modelscope` 时也能正常工作。

逐句时间戳（语音驱动抽帧）需要 paraformer-zh 配合 vad/punc 模型并开启
`sentence_timestamp=True`：
    SenseVoiceTranscriber(model="paraformer-zh", vad_model="fsmn-vad",
                          punc_model="ct-punc-c", sentence_timestamp=True)
注意：funasr 的 `sentence_info` 里 start/end 单位是毫秒；`_parse` 会转成秒。
"""
import re
from dataclasses import dataclass
from typing import List, Optional

# 模块级占位符：首次调用 `_load_model` 时惰性填充。暴露出来是为了测试可以在
# 未安装 funasr 的情况下 monkeypatch `video2md.asr.sensevoice.AutoModel`；
# 不纳入公开 API。
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
        """把 FunASR 输出规范化成 Segment 列表。

        优先使用逐句时间戳；否则回退为整段转写的单个 segment。
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
