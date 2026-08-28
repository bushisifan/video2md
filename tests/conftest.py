import shutil
import subprocess

import cv2
import numpy as np
import pytest


@pytest.fixture
def sample_video(tmp_path):
    """9 秒合成视频：3 个明显不同场景，用于抽帧测试。"""
    path = str(tmp_path / "sample.mp4")
    fps = 10
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (320, 180))
    assert writer.isOpened(), "VideoWriter failed to open (missing codec?)"
    for _ in range(30):  # 场景1：黑底白方块 (0-3s)
        img = np.zeros((180, 320, 3), dtype=np.uint8)
        img[50:130, 100:220] = (255, 255, 255)
        writer.write(img)
    for _ in range(30):  # 场景2：灰底红圆 (3-6s)
        img = np.full((180, 320, 3), 80, dtype=np.uint8)
        cv2.circle(img, (160, 90), 40, (0, 0, 255), -1)
        writer.write(img)
    for _ in range(30):  # 场景3：白底蓝方块 (6-9s)
        img = np.full((180, 320, 3), 255, dtype=np.uint8)
        img[30:150, 60:260] = (255, 0, 0)
        writer.write(img)
    writer.release()
    return path


@pytest.fixture
def video_with_audio(tmp_path):
    """2 秒带 440Hz 音的视频，用于抽音测试（需 ffmpeg）。"""
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg not installed")
    path = str(tmp_path / "with_audio.mp4")
    cmd = [
        "ffmpeg", "-y",
        "-f", "lavfi", "-i", "color=c=black:s=320x180:d=2",
        "-f", "lavfi", "-i", "sine=frequency=440:duration=2",
        "-c:v", "libx264", "-c:a", "aac", "-pix_fmt", "yuv420p", path,
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    return path


@pytest.fixture
def click_video(tmp_path):
    """1.5 秒视频：中间短暂出现一个小方块（模拟光标点击），用于光标检测测试。"""
    path = str(tmp_path / "clicks.mp4")
    fps = 10
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (320, 180))
    assert writer.isOpened(), "VideoWriter failed to open (missing codec?)"
    for _ in range(5):  # 0.0-0.5s 稳定
        img = np.zeros((180, 320, 3), dtype=np.uint8)
        writer.write(img)
    for _ in range(5):  # 0.5-1.0s 出现 10x10 小方块
        img = np.zeros((180, 320, 3), dtype=np.uint8)
        img[85:95, 155:165] = (0, 255, 0)
        writer.write(img)
    for _ in range(5):  # 1.0-1.5s 稳定
        img = np.zeros((180, 320, 3), dtype=np.uint8)
        writer.write(img)
    writer.release()
    return path
