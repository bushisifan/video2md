"""Qwen2.5-VL 客户端：通过 vLLM 的 OpenAI 兼容接口调用。"""
import json
import logging
from typing import Any, Dict, List, Optional

import httpx
from openai import OpenAI

from video2md.vision.prompt import build_vision_messages

logger = logging.getLogger(__name__)

VALID_ACTIONS = {"click", "input", "navigate", "scroll", "toggle", "select", "open", "other"}


class VisionClient:
    """用本地视觉模型（vLLM/Ollama）理解屏幕关键帧。"""

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
        # trust_env=False：忽略系统/环境代理。本地(localhost)与可直连的云端
        # API 都不需要代理；否则 SakuraCat 等 TUN/代理会拦截 localhost 的
        # 大图片请求并返回 502。
        self.client = client or OpenAI(
            base_url=base_url,
            api_key=api_key,
            http_client=httpx.Client(trust_env=False),
        )

    def understand_frame(self, image_path: str, timestamp: float) -> Dict[str, Any]:
        messages = build_vision_messages(image_path, timestamp)
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            timeout=self.timeout,
            response_format={"type": "json_object"},
        )
        return self._parse(resp.choices[0].message.content)

    def understand_frames(self, frames: List[Any]) -> List[Dict[str, Any]]:
        """批量理解关键帧：单帧失败不拖垮整批。"""
        results: List[Dict[str, Any]] = []
        for f in frames:
            try:
                results.append(self.understand_frame(f.image_path, f.timestamp))
            except Exception as e:  # noqa: BLE001 - 逐帧降级
                logger.exception("视觉理解失败，frame 时间 %.1fs", f.timestamp)
                results.append(
                    {
                        "action": "other",
                        "target": "",
                        "detail": f"视觉理解失败: {e}",
                        "needs_review": True,
                    }
                )
        return results

    @staticmethod
    def _parse(content: Optional[str]) -> Dict[str, Any]:
        if not content:
            return {"action": "other", "target": "", "detail": "", "needs_review": True}
        text = content.strip()
        if text.startswith("```"):
            text = text.strip("`")
            text = text[4:] if text[:4].lower() == "json" else text
        try:
            data = json.loads(text.strip())
        except json.JSONDecodeError:
            return {"action": "other", "target": "", "detail": text, "needs_review": True}
        action = data.get("action", "other")
        needs_review = data.get("needs_review") in (True, "true")
        if action not in VALID_ACTIONS:
            action = "other"
            needs_review = True
        return {
            "action": action,
            "target": data.get("target", ""),
            "detail": data.get("detail", ""),
            "needs_review": needs_review,
        }
