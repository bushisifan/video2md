"""LLM 驱动的步骤时间窗切分（方案B）。

先把带时间戳的语音转写交给 LLM 切成"操作步骤 + 时间区间"，随后按这些
时间区间抽帧，保证"步骤 ↔ 截图"按时间对齐——避免同画面多操作导致的
文图不一致。时间点最后吸附到最近的转写片段边界，防止出现转写里没有的
时间。
"""
import json
from typing import List, Optional

import httpx
from openai import OpenAI
from pydantic import BaseModel, ConfigDict, Field

from video2md.asr.sensevoice import Segment


class StepWindow(BaseModel):
    model_config = ConfigDict(extra="forbid")
    order: int = Field(ge=1)
    title: str = Field(min_length=1)
    start: float = Field(ge=0)
    end: float = Field(ge=0)


class StepWindowList(BaseModel):
    model_config = ConfigDict(extra="forbid")
    steps: List[StepWindow] = Field(min_length=1)


STEP_DETECTION_SYSTEM_PROMPT = (
    "你是一个步骤切分助手。给你一段带时间戳的中文语音转写（来自屏幕操作录屏），"
    "请把它切成若干\"操作步骤\"，每步给出时间区间（单位：秒）。\n"
    "输出严格 JSON：\n"
    '{"steps": [{"order": 1, "title": "步骤标题", "start": 0.0, "end": 10.0}, ...]}\n'
    "规则：\n"
    "- start/end 必须是秒，且尽量对齐下面转写片段的时间戳\n"
    "- 步骤按时间顺序排列，合并后覆盖整个转写时长（前一步 end ≈ 后一步 start）\n"
    "- 每步一个动作，用祈使句（如\"点击添加供应商\"）\n"
    "- 4-20 步，视内容复杂度\n"
)


def build_step_detection_messages(segments: List[Segment]) -> list:
    seg_text = (
        "\n".join(f"[{s.start:.2f}-{s.end:.2f}] {s.text}" for s in segments)
        if segments
        else "(无转写)"
    )
    user = f"带时间戳的转写片段：\n{seg_text}\n\n请输出步骤列表 JSON。"
    return [
        {"role": "system", "content": STEP_DETECTION_SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


class StepDetector:
    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        temperature: float = 0.1,
        max_tokens: int = 2048,
        timeout: float = 120,
        client: Optional[OpenAI] = None,
    ):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        # trust_env=False：忽略系统/环境代理（本地与可直连的云端 API 都不需要代理）
        self.client = client or OpenAI(
            base_url=base_url,
            api_key=api_key,
            http_client=httpx.Client(trust_env=False),
        )

    def detect(
        self, segments: List[Segment], max_retries: int = 2
    ) -> List[StepWindow]:
        messages = build_step_detection_messages(segments)
        last_err: Optional[Exception] = None
        for _ in range(max_retries + 1):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    response_format={"type": "json_object"},
                )
                content = resp.choices[0].message.content or ""
                data = json.loads(content.strip())
                result = StepWindowList.model_validate(data)
                return self._snap_to_segments(result.steps, segments)
            except Exception as e:  # noqa: BLE001 - 解析/校验失败时重试
                last_err = e
        raise RuntimeError(f"步骤切分失败: {last_err}")

    @staticmethod
    def _snap_to_segments(
        steps: List[StepWindow], segments: List[Segment]
    ) -> List[StepWindow]:
        """把 LLM 给的秒数吸附到最近的转写片段边界（去重后）。"""
        if not segments:
            return steps
        bounds = sorted({round(s.start, 2) for s in segments} | {round(s.end, 2) for s in segments})

        def snap(t: float) -> float:
            return min(bounds, key=lambda b: abs(b - t))

        out: List[StepWindow] = []
        for s in steps:
            start = snap(s.start)
            end = max(snap(s.end), start)
            out.append(StepWindow(order=s.order, title=s.title, start=start, end=end))
        return out
