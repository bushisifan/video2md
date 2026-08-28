"""video2md 的配置模型。

把 config.yaml 加载到类型化的 pydantic 模型；缺失的 section 回退到模块级
默认值，因此即使 YAML 很精简也能得到合法 Config。
"""
from pathlib import Path
from typing import Union

import yaml
from pydantic import BaseModel, ConfigDict

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "config.yaml"


class ASRConfig(BaseModel):
    model: str = "paraformer-zh"
    device: str = "cpu"
    sample_rate: int = 16000
    # 逐句时间戳（语音驱动抽帧）：需 vad/punc 模型配合
    vad_model: str = "fsmn-vad"
    punc_model: str = "ct-punc-c"
    sentence_timestamp: bool = True


class LLMEndpointConfig(BaseModel):
    base_url: str = "http://localhost:8000/v1"
    api_key: str = "EMPTY"
    model: str = "Qwen/Qwen2.5-VL-7B-Instruct"
    temperature: float = 0.1
    max_tokens: int = 2048
    timeout: float = 120
    # 输入 token 上限：提示词超过该值会切分成多块调用，避免依赖模型自带的大上下文
    max_input_tokens: int = 24000


class PreprocessConfig(BaseModel):
    scene_threshold: float = 30.0
    interval_seconds: float = 2.0
    resize_width: int = 768
    min_frames: int = 5
    audio_sample_rate: int = 16000


class CursorConfig(BaseModel):
    enabled: bool = False


class RenderConfig(BaseModel):
    images_dir: str = "images"


class Config(BaseModel):
    model_config = ConfigDict(extra="forbid")

    asr: ASRConfig = ASRConfig()
    vision: LLMEndpointConfig = LLMEndpointConfig()
    compose: LLMEndpointConfig = LLMEndpointConfig(
        model="Qwen/Qwen2.5-7B-Instruct", max_tokens=8192
    )
    preprocess: PreprocessConfig = PreprocessConfig()
    cursor: CursorConfig = CursorConfig()
    render: RenderConfig = RenderConfig()

    @classmethod
    def load(cls, path: Union[str, Path] = DEFAULT_CONFIG_PATH) -> "Config":
        """从 YAML 文件加载配置，与默认值合并。"""
        config_path = Path(path)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        with open(config_path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        return cls(**raw)
