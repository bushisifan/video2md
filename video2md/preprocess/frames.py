"""Keyframe extraction with scene-change detection.

Pipeline step ①: reduce a long screen recording to the frames that actually
matter. Scene changes (page flips, dialogs, state changes) are the main step
boundaries; interval sampling is a fallback so we never end up with too few
frames.
"""
import math
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import List

import cv2


@dataclass
class KeyFrame:
    timestamp: float
    image_path: str


class FrameExtractor:
    def __init__(
        self,
        scene_threshold: float = 30.0,
        interval_seconds: float = 2.0,
        resize_width: int = 768,
        min_frames: int = 5,
        detection_sample_rate: float = 2.0,
    ):
        if interval_seconds <= 0:
            raise ValueError("interval_seconds must be > 0")
        if scene_threshold <= 0:
            raise ValueError("scene_threshold must be > 0")
        self.scene_threshold = scene_threshold
        self.interval_seconds = interval_seconds
        self.resize_width = resize_width
        self.min_frames = min_frames
        self.detection_sample_rate = detection_sample_rate

    def _detect_scene_timestamps(self, video_path: str) -> List[float]:
        """Return timestamps (seconds) where a scene change was detected.

        Mean absolute difference of consecutive downsampled grayscale frames;
        a jump above `scene_threshold` marks a new scene. Timestamp 0 is always
        included as the first frame. The diff is computed only on sampled frames
        (`detection_sample_rate` samples per second) to cut decode/compute cost.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        if math.isnan(fps):
            fps = 0.0
        if fps <= 0:
            return [0.0]

        sample_every = max(1, round(fps / self.detection_sample_rate))

        timestamps = [0.0]
        prev_small = None
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % sample_every == 0:
                small = cv2.resize(frame, (128, 72))
                gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
                if prev_small is not None:
                    diff = float(cv2.absdiff(prev_small, gray).mean())
                    if diff > self.scene_threshold:
                        timestamps.append(frame_idx / fps)
                prev_small = gray
            frame_idx += 1
        cap.release()
        return timestamps

    def _ensure_min_frames(self, timestamps: List[float], duration: float) -> List[float]:
        """Pad with interval-sampled timestamps so we have at least min_frames."""
        if len(timestamps) >= self.min_frames or duration <= 0:
            return timestamps
        step = self.interval_seconds
        ts = 0.0
        while len(timestamps) < self.min_frames and ts < duration:
            if not any(abs(t - ts) < 0.1 for t in timestamps):
                timestamps.append(round(ts, 3))
            ts += step
        return sorted(timestamps)

    def extract(self, video_path: str, output_dir: str) -> List[KeyFrame]:
        """Extract keyframes to `output_dir`, return metadata list.

        `image_path` mirrors the real write path (`out_path / filename`) so the
        pipeline can later relativize it against the output root for the
        Markdown embeds.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0.0
        cap.release()

        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        for stale in out_path.glob("frame_*.png"):
            stale.unlink()

        timestamps = self._detect_scene_timestamps(video_path)
        timestamps = self._ensure_min_frames(timestamps, duration)

        keyframes: List[KeyFrame] = []
        cap = cv2.VideoCapture(video_path)
        for i, ts in enumerate(timestamps, start=1):
            cap.set(cv2.CAP_PROP_POS_MSEC, ts * 1000)
            ret, frame = cap.read()
            if not ret:
                warnings.warn(f"failed to read frame at {ts}s")
                continue
            frame = self._resize(frame)
            filename = f"frame_{i:04d}.png"
            ok = cv2.imwrite(str(out_path / filename), frame)
            if not ok:
                raise RuntimeError(f"failed to write frame: {out_path / filename}")
            keyframes.append(
                KeyFrame(timestamp=round(ts, 3), image_path=str(out_path / filename))
            )
        cap.release()
        return keyframes

    def extract_at_timestamps(
        self, video_path: str, timestamps: List[float], output_dir: str
    ) -> List[KeyFrame]:
        """Extract a frame at each given timestamp (seconds).

        语音驱动抽帧：按 LLM 切分出的步骤时间点取帧，保证"步骤 ↔ 截图"
        按时间对齐（方案B）。时间戳去重、排序；`image_path` 与 `extract`
        保持一致（真实写路径）。
        """
        if not timestamps:
            return []
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        for stale in out_path.glob("frame_*.png"):
            stale.unlink()

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        keyframes: List[KeyFrame] = []
        for i, ts in enumerate(sorted(set(round(t, 3) for t in timestamps)), start=1):
            cap.set(cv2.CAP_PROP_POS_MSEC, ts * 1000)
            ret, frame = cap.read()
            if not ret:
                warnings.warn(f"failed to read frame at {ts}s")
                continue
            frame = self._resize(frame)
            filename = f"frame_{i:04d}.png"
            ok = cv2.imwrite(str(out_path / filename), frame)
            if not ok:
                raise RuntimeError(f"failed to write frame: {out_path / filename}")
            keyframes.append(
                KeyFrame(timestamp=round(ts, 3), image_path=str(out_path / filename))
            )
        cap.release()
        return keyframes

    def _resize(self, frame):
        height, width = frame.shape[:2]
        if width <= self.resize_width:
            return frame
        ratio = self.resize_width / width
        return cv2.resize(frame, (self.resize_width, int(height * ratio)))
