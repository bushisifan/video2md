"""Audio extraction from video files using ffmpeg."""
import subprocess
from pathlib import Path
from typing import Union


def extract_audio(
    video_path: str,
    output_path: Union[str, Path],
    sample_rate: int = 16000,
) -> str:
    """Extract mono 16kHz WAV audio from a video file.

    Raises RuntimeError if ffmpeg fails or produces no file (e.g. a video with
    no audio track). The pipeline treats that as a silent-video fallback.
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
