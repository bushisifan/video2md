"""用 ffmpeg 从视频中抽取音频。"""
import subprocess
from pathlib import Path
from typing import Union


def extract_audio(
    video_path: str,
    output_path: Union[str, Path],
    sample_rate: int = 16000,
) -> str:
    """从视频抽取单声道 16kHz WAV 音频。

    若 ffmpeg 失败或未产出文件（如无音轨的视频）则抛 RuntimeError；
    管线把这种情况当作"无声视频"降级处理。
    """
    cmd = [
        "ffmpeg", "-y", "-nostdin", "-loglevel", "error",
        "-i", str(video_path),
        "-vn", "-ac", "1", "-ar", str(sample_rate),
        "-acodec", "pcm_s16le",
        str(output_path),
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, errors="replace", timeout=300)
    except FileNotFoundError:
        raise RuntimeError("ffmpeg not found in PATH; install ffmpeg to extract audio") from None
    except subprocess.TimeoutExpired:
        raise RuntimeError(f"ffmpeg timed out extracting audio from: {video_path}") from None
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {result.stderr}")
    if not Path(output_path).exists():
        raise RuntimeError(f"ffmpeg produced no audio file: {output_path}")
    return str(output_path)
