class FakeMessage:
    def __init__(self, content):
        self.content = content


class FakeChoice:
    def __init__(self, content):
        self.message = FakeMessage(content)


class FakeResponse:
    def __init__(self, content):
        self.choices = [FakeChoice(content)]


class FakeCompletions:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return FakeResponse(self.responses.pop(0))


class FakeChat:
    def __init__(self, responses):
        self.completions = FakeCompletions(responses)


class FakeClient:
    def __init__(self, responses):
        self.chat = FakeChat(responses)


def _make_client(responses):
    from video2md.vision.client import VisionClient

    return VisionClient(
        base_url="http://x", api_key="y", model="m", client=FakeClient(responses)
    )


def test_understand_frame_parses(tmp_path):
    img = tmp_path / "f.png"
    img.write_bytes(b"\x89PNG\r\n\x1a\n" + b"y" * 10)
    vc = _make_client([
        '{"action": "click", "target": "设置 > 齿轮图标", '
        '"detail": "用户点击齿轮图标", "needs_review": false}'
    ])
    out = vc.understand_frame(str(img), 3.2)
    assert out["action"] == "click"
    assert out["target"] == "设置 > 齿轮图标"
    assert out["needs_review"] is False


def test_understand_frame_invalid_json(tmp_path):
    img = tmp_path / "f.png"
    img.write_bytes(b"z")
    vc = _make_client(["not json"])
    out = vc.understand_frame(str(img), 1.0)
    assert out["needs_review"] is True


def test_understand_frame_with_markdown_wrapper(tmp_path):
    img = tmp_path / "f.png"
    img.write_bytes(b"z")
    vc = _make_client(['```json\n{"action": "open", "target": "菜单", "detail": "打开", "needs_review": false}\n```'])
    out = vc.understand_frame(str(img), 5.0)
    assert out["action"] == "open"


def test_understand_frames_recovers_on_error(tmp_path):
    img = tmp_path / "f.png"
    img.write_bytes(b"z")

    class BoomClient:
        def __init__(self):
            self.chat = type("Chat", (), {
                "completions": type("Completions", (), {
                    "create": lambda self, **kw: (_ for _ in ()).throw(RuntimeError("conn refused"))
                })()
            })()

    from video2md.vision.client import VisionClient

    vc = VisionClient(base_url="http://x", api_key="y", model="m", client=BoomClient())
    frames = [type("KF", (), {"image_path": str(img), "timestamp": 1.0})()]
    results = vc.understand_frames(frames)
    assert len(results) == 1
    assert results[0]["needs_review"] is True
    assert "视觉理解失败" in results[0]["detail"]


def test_understand_frame_string_boolean_needs_review(tmp_path):
    img = tmp_path / "f.png"
    img.write_bytes(b"z")
    vc = _make_client(['{"action": "click", "target": "x", "detail": "d", "needs_review": "false"}'])
    out = vc.understand_frame(str(img), 2.0)
    assert out["needs_review"] is False

    vc = _make_client(['{"action": "click", "target": "x", "detail": "d", "needs_review": "true"}'])
    out = vc.understand_frame(str(img), 2.0)
    assert out["needs_review"] is True


def test_understand_frame_unknown_action_falls_back(tmp_path):
    img = tmp_path / "f.png"
    img.write_bytes(b"z")
    vc = _make_client(['{"action": "hack", "target": "x", "detail": "d", "needs_review": false}'])
    out = vc.understand_frame(str(img), 2.0)
    assert out["action"] == "other"
    assert out["needs_review"] is True


def test_understand_frame_uppercase_fence(tmp_path):
    img = tmp_path / "f.png"
    img.write_bytes(b"z")
    vc = _make_client([
        '```JSON\n{"action": "open", "target": "t", "detail": "d", "needs_review": false}\n```'
    ])
    out = vc.understand_frame(str(img), 2.0)
    assert out["action"] == "open"
