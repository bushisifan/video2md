"""步骤合成：用本地文本 LLM（vLLM，OpenAI 兼容接口）生成 SOP 步骤树。"""
import json
import logging
from typing import Any, Dict, List, Optional

import httpx
from openai import OpenAI

from video2md.compose.prompt import build_compose_messages
from video2md.compose.schema import SOPDocument
from video2md.compose.tokens import estimate_tokens

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
        max_input_tokens: int = 24000,
    ):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.max_input_tokens = max_input_tokens
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
        """合成 SOP；输入超过 max_input_tokens 时按时间切分、分块合成再合并。"""
        cursor = cursor_events or []
        if self._input_tokens(segments, frame_understandings, cursor, step_windows) <= self.max_input_tokens:
            return self._synthesize_once(
                segments, frame_understandings, cursor, step_windows, max_retries
            )
        parts = []
        for segs, vis, curs, wins in self._iter_chunks(
            segments, frame_understandings, cursor, step_windows
        ):
            parts.append(self._synthesize_once(segs, vis, curs, wins, max_retries))
        return self._merge_sops(parts)

    def _input_tokens(
        self,
        segments: List[Any],
        frame_understandings: List[Dict[str, Any]],
        cursor_events: List[Any],
        step_windows: List[Any] | None,
    ) -> int:
        messages = build_compose_messages(segments, frame_understandings, cursor_events, step_windows)
        return sum(estimate_tokens(m.get("content", "")) for m in messages)

    def _iter_chunks(
        self,
        segments: List[Any],
        frame_understandings: List[Dict[str, Any]],
        cursor_events: List[Any],
        step_windows: List[Any] | None,
    ):
        """把超限输入按时间切分成若干组；组内输入 ≤ max_input_tokens。"""
        if step_windows:
            for wins in self._chunk_windows(segments, frame_understandings, step_windows):
                segs, vis = self._slice(segments, frame_understandings, wins)
                yield segs, vis, cursor_events, wins
        elif segments:
            for seg_chunk in self._chunk_segments(segments, frame_understandings):
                t0, t1 = seg_chunk[0].start, seg_chunk[-1].end
                vis = [
                    v for v in frame_understandings if t0 - 2 <= v["timestamp"] <= t1 + 2
                ]
                yield seg_chunk, vis, cursor_events, None
        else:
            for vis_chunk in self._chunk_vis(frame_understandings):
                yield [], vis_chunk, cursor_events, None

    def _chunk_windows(
        self,
        segments: List[Any],
        frame_understandings: List[Dict[str, Any]],
        step_windows: List[Any],
    ) -> List[List[Any]]:
        """把步骤时间窗切成若干组，使每组切片后的输入不超过上限。"""
        chunks: List[List[Any]] = []
        current: List[Any] = []
        for w in step_windows:
            candidate = current + [w]
            segs, vis = self._slice(segments, frame_understandings, candidate)
            if self._input_tokens(segs, vis, [], candidate) > self.max_input_tokens and current:
                chunks.append(current)
                current = [w]
            else:
                current = candidate
        if current:
            chunks.append(current)
        return chunks

    def _chunk_segments(
        self,
        segments: List[Any],
        frame_understandings: List[Dict[str, Any]],
    ) -> List[List[Any]]:
        """无步骤时间窗时，按片段切组；组内只带该时间段内的帧理解。"""
        chunks: List[List[Any]] = []
        current: List[Any] = []
        for s in segments:
            candidate = current + [s]
            if not candidate:
                continue
            t0, t1 = candidate[0].start, candidate[-1].end
            vis = [
                v for v in frame_understandings if t0 - 2 <= v["timestamp"] <= t1 + 2
            ]
            if self._input_tokens(candidate, vis, [], None) > self.max_input_tokens and current:
                chunks.append(current)
                current = [s]
            else:
                current = candidate
        if current:
            chunks.append(current)
        return chunks

    def _chunk_vis(self, frame_understandings: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
        """无声且无时间窗时，仅按帧理解切组。"""
        chunks: List[List[Dict[str, Any]]] = []
        current: List[Dict[str, Any]] = []
        for v in frame_understandings:
            candidate = current + [v]
            if self._input_tokens([], candidate, [], None) > self.max_input_tokens and current:
                chunks.append(current)
                current = [v]
            else:
                current = candidate
        if current:
            chunks.append(current)
        return chunks

    @staticmethod
    def _slice(
        segments: List[Any],
        frame_understandings: List[Dict[str, Any]],
        windows: List[Any],
    ) -> tuple:
        """取落在给定步骤时间窗范围（含少量缓冲）内的片段与帧理解。"""
        if not windows:
            return segments, frame_understandings
        t0 = windows[0].start
        t1 = windows[-1].end
        segs = [s for s in segments if s.end >= t0 - 0.5 and s.start <= t1 + 0.5]
        vis = [v for v in frame_understandings if t0 - 2 <= v["timestamp"] <= t1 + 2]
        return segs, vis

    @staticmethod
    def _merge_sops(parts: List[SOPDocument]) -> SOPDocument:
        """合并分块合成的 SOP：步骤顺序拼接并重编号，标题/目的/前置取首块，疑难解答去重。"""
        if len(parts) == 1:
            return parts[0]
        base = parts[0]
        steps = []
        for p in parts:
            steps.extend(p.steps)
        for i, s in enumerate(steps, 1):
            s.order = i
        issues: dict = {}
        for p in parts:
            for t in p.troubleshooting:
                issues.setdefault(t.issue, t)
        title = next((p.title for p in parts if p.title), base.title)
        return SOPDocument(
            title=title,
            purpose=base.purpose,
            prerequisites=base.prerequisites,
            steps=steps,
            troubleshooting=list(issues.values()),
        )

    def _synthesize_once(
        self,
        segments: List[Any],
        frame_understandings: List[Dict[str, Any]],
        cursor_events: List[Any],
        step_windows: List[Any] | None,
        max_retries: int,
    ) -> SOPDocument:
        messages = build_compose_messages(
            segments, frame_understandings, cursor_events, step_windows
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
            except Exception as e:  # noqa: BLE001 - 解析/校验失败时重试
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
