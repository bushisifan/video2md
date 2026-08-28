import pytest
from pydantic import ValidationError

from video2md.config import Config


def test_config_unknown_section_raises(tmp_path):
    cfg_yaml = tmp_path / "config.yaml"
    cfg_yaml.write_text("preproces:\n  scene_threshold: 10\n", encoding="utf-8")
    with pytest.raises(ValidationError):
        Config.load(str(cfg_yaml))


def test_config_loads_defaults(tmp_path):
    cfg_yaml = tmp_path / "config.yaml"
    cfg_yaml.write_text("vision:\n  model: my-vl-model\n", encoding="utf-8")
    cfg = Config.load(str(cfg_yaml))
    assert cfg.vision.model == "my-vl-model"
    assert cfg.vision.base_url == "http://localhost:8000/v1"
    assert cfg.vision.api_key == "EMPTY"
    assert cfg.asr.device == "cpu"
    assert cfg.preprocess.scene_threshold == 30.0
    assert cfg.cursor.enabled is False


def test_config_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        Config.load(str(tmp_path / "nope.yaml"))


def test_bundled_config_loads():
    cfg = Config.load()
    assert cfg.vision.model.startswith("Qwen")
    assert cfg.compose.model.startswith("Qwen")
