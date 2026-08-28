from pathlib import Path

from video2md.preprocess.frames import FrameExtractor, KeyFrame


def test_extract_finds_scene_changes(sample_video, tmp_path):
    out = tmp_path / "images"
    extractor = FrameExtractor(
        scene_threshold=20.0,
        interval_seconds=1.0,
        resize_width=160,
        min_frames=2,
    )
    frames = extractor.extract(sample_video, str(out))
    assert len(frames) >= 2
    assert all(isinstance(f, KeyFrame) for f in frames)
    for f in frames:
        assert (out / Path(f.image_path).name).exists()
    timestamps = [f.timestamp for f in frames]
    assert timestamps == sorted(timestamps)


def test_min_frames_padding(click_video, tmp_path):
    # click_video 场景单一，几乎无场景变化 → 用间隔采样兜底到 min_frames
    out = tmp_path / "images"
    extractor = FrameExtractor(
        scene_threshold=50.0,
        interval_seconds=0.5,
        resize_width=128,
        min_frames=3,
    )
    frames = extractor.extract(click_video, str(out))
    assert len(frames) >= 3


def test_resize_applies(sample_video, tmp_path):
    import cv2
    out = tmp_path / "images"
    extractor = FrameExtractor(scene_threshold=20.0, resize_width=160, min_frames=2)
    frames = extractor.extract(sample_video, str(out))
    img = cv2.imread(str(out / Path(frames[0].image_path).name))
    assert img.shape[1] == 160


def test_extract_at_timestamps(sample_video, tmp_path):
    out = tmp_path / "images"
    extractor = FrameExtractor(resize_width=160)
    frames = extractor.extract_at_timestamps(
        sample_video, [1.0, 4.0, 7.0], str(out)
    )
    assert len(frames) == 3
    assert [f.timestamp for f in frames] == [1.0, 4.0, 7.0]
    for f in frames:
        assert (out / Path(f.image_path).name).exists()


def test_extract_at_timestamps_dedupes(sample_video, tmp_path):
    out = tmp_path / "images"
    extractor = FrameExtractor(resize_width=160)
    frames = extractor.extract_at_timestamps(
        sample_video, [1.0, 1.0, 4.0], str(out)
    )
    assert len(frames) == 2
    assert [f.timestamp for f in frames] == [1.0, 4.0]


def test_extract_at_timestamps_empty(sample_video, tmp_path):
    extractor = FrameExtractor(resize_width=160)
    assert extractor.extract_at_timestamps(sample_video, [], str(tmp_path / "x")) == []
