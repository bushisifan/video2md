"""Step synthesis using a local text LLM (vLLM, OpenAI-compatible)."""
import json
import logging
from typing import Any, Dict, List, Optional

import httpx
from openai import OpenAI

from video2md.compose.prompt import build_compose_messages
from video2md.compose.schema import SOPDocument

logger = logging.getLogger(__name__)


class StepSynthesizer:
    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        temperature: float = 0.1,
        max_tokens: int = 8192,
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

    def synthesize(
        self,
        segments: List[Any],
        frame_understandings: List[Dict[str, Any]],
        cursor_events: List[Any] | None = None,
        step_windows: List[Any] | None = None,
        max_retries: int = 2,
    ) -> SOPDocument:
        messages = build_compose_messages(
            segments, frame_understandings, cursor_events or [], step_windows
        )
        last_err: Optional[Exception] = None
        for attempt in range(max_retries + 1):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    timeout=self.timeout,
                    response_format={"type": "json_object"},
                )
                content = resp.choices[0].message.content or ""
                data = self._extract_json(content)
                return SOPDocument.model_validate(data)
            except Exception as e:  # noqa: BLE001 - retry on any parse/validation failure
                last_err = e
                if isinstance(e, json.JSONDecodeError):
                    logger.warning("步骤合成 JSON 解析失败 (第 %d 次)：%r", attempt + 1, content[:500])
                else:
                    logger.warning("步骤合成失败 (第 %d 次)：%s", attempt + 1, e)
        raise RuntimeError(f"步骤合成失败: {last_err}")

    @staticmethod
    def _extract_json(content: str) -> dict:
        text = content.strip()
        if text.startswith("```"):
            text = text.strip("`")
            if text[:4].lower() == "json":
                text = text[4:]
        return json.loads(text.strip())
