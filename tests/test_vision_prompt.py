import base64

from video2md.vision.prompt import VISION_SYSTEM_PROMPT, build_vision_messages


def test_system_prompt_has_valid_json_template():
    content = VISION_SYSTEM_PROMPT
    for key in ("action", "target", "detail", "needs_review"):
        assert f'"{key}"' in content
    assert "click|input" in content
    assert '"needs_review": true}' in content


def test_build_messages_contains_image(tmp_path):
    img = tmp_path / "frame.png"
    raw = b"\x89PNG\r\n\x1a\n" + b"x" * 10
    img.write_bytes(raw)
    msgs = build_vision_messages(str(img), 3.2)
    assert msgs[0]["role"] == "system"
    assert "click|input" in msgs[0]["content"]
    user = msgs[1]
    assert user["role"] == "user"
    assert user["content"][0]["type"] == "text"
    assert user["content"][0]["text"] == "这是录屏视频时间点 3.2 秒的关键帧。请分析屏幕上的操作并输出 JSON。"
    assert "3.2 秒" in user["content"][0]["text"]
    image_url = user["content"][1]["image_url"]["url"]
    assert image_url.startswith("data:image/png;base64,")
    encoded = image_url.split("data:image/png;base64,", 1)[1]
    assert base64.b64decode(encoded) == raw
