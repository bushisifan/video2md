"""光标点击检测（可选，启发式）。

相邻两帧间某块局部区域变化剧烈，视为一次可能的鼠标点击。该功能刻意做成
尽力而为：仅作为步骤合成的弱信号，且默认通过配置关闭。
"""
import math
from dataclasses import dataclass
from typing import List

import cv2


@dataclass
class ClickEvent:
    timestamp: float
    x: int
    y: int


class CursorDetector:
    def __init__(
        self,
        change_ratio: float = 0.35,
        region_ratio: float = 0.05,
        min_click_gap: float = 0.5,
    ):
        self.change_ratio = change_ratio
        self.region_ratio = region_ratio
        self.min_click_gap = min_click_gap

    def detect(self, video_path: str) -> List[ClickEvent]:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        try:
            fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
            if fps <= 0 or math.isnan(fps):
                return []

            events: List[ClickEvent] = []
            prev = None
            prev_event_ts = -self.min_click_gap
            frame_idx = 0
            max_region_area = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                if prev is not None:
                    diff = cv2.absdiff(prev, gray)
                    min_x, max_x, min_y, max_y = self._hot_region(diff)
                    region_area = (max_x - min_x) * (max_y - min_y)
                    ts = frame_idx / fps
                    if (
                        0 < region_area <= max_region_area
                        and ts - prev_event_ts >= self.min_click_gap
                    ):
                        events.append(
                            ClickEvent(
                                timestamp=round(ts, 3),
                                x=(min_x + max_x) // 2,
                                y=(min_y + max_y) // 2,
                            )
                        )
                        prev_event_ts = ts
                else:
                    height, width = gray.shape
                    max_region_area = self.region_ratio * height * width
                prev = gray
                frame_idx += 1
            return events
        finally:
            cap.release()

    def _hot_region(self, diff):
        """变化超过 change_ratio × 最大变化的像素的外接矩形框。"""
        max_val = int(diff.max())
        threshold = int(max_val * self.change_ratio) or 1
        ys, xs = (diff > threshold).nonzero()
        if len(xs) == 0:
            return 0, 0, 0, 0
        return int(xs.min()), int(xs.max()), int(ys.min()), int(ys.max())
