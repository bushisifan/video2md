import shutil

import pytest

from video2md.preprocess.audio import extract_audio

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")


def test_extract_audio_wav(video_with_audio, tmp_path):
    import wave

    out = tmp_path / "audio.wav"
    result = extract_audio(video_with_audio, str(out), sample_rate=16000)
    assert result == str(out)
    assert out.exists()
    assert out.stat().st_size > 0
    with wave.open(str(out), "rb") as w:
        assert w.getnchannels() == 1
        assert w.getframerate() == 16000
        assert w.getsampwidth() == 2


def test_extract_audio_no_audio_track_raises(sample_video, tmp_path):
    with pytest.raises(RuntimeError):
        extract_audio(sample_video, str(tmp_path / "x.wav"))
