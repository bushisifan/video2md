import cv2
import numpy as np

from video2md.preprocess.cursor import ClickEvent, CursorDetector


def test_detect_fps_zero_returns_empty(monkeypatch):
    # 提供两帧真实帧：若 fps 保护被移除，第二帧会执行 frame_idx / fps (1/0.0)
    # 抛出 ZeroDivisionError，从而该测试能真正起到回归保护作用。
    class FakeCapture:
        def __init__(self):
            self.frames = [
                np.zeros((180, 320, 3), dtype=np.uint8),
                np.zeros((180, 320, 3), dtype=np.uint8),
            ]

        def isOpened(self):
            return True

        def get(self, prop):
            if prop == cv2.CAP_PROP_FPS:
                return 0.0
            return 0

        def read(self):
            if self.frames:
                return True, self.frames.pop(0)
            return False, None

        def release(self):
            pass

    monkeypatch.setattr("video2md.preprocess.cursor.cv2.VideoCapture", lambda *a, **k: FakeCapture())
    assert CursorDetector().detect("whatever.mp4") == []


def test_detect_localized_click(click_video):
    detector = CursorDetector(change_ratio=0.35, region_ratio=0.05, min_click_gap=0.5)
    events = detector.detect(click_video)
    assert len(events) >= 1
    assert isinstance(events[0], ClickEvent)
    # 小方块中心 (160, 90)，允许一定误差
    assert 140 <= events[0].x <= 180
    assert 70 <= events[0].y <= 110
