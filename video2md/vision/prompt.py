"""Qwen2.5-VL 视觉理解提示词。"""
import base64

VISION_SYSTEM_PROMPT = (
    "你是一个专业的屏幕操作分析助手。你会收到一张来自屏幕操作录屏视频的关键帧截图。"
    "请分析这一刻屏幕上发生了什么操作，并输出严格 JSON，不要输出任何其他内容：\n"
    '{"action": "click|input|navigate|scroll|toggle|select|open|other", '
    '"target": "被操作按钮/菜单/输入框的名称与路径", '
    '"detail": "这一刻屏幕上发生了什么", '
    '"needs_review": true}\n'
    "其中 action 只能是上面枚举之一；needs_review 是布尔值："
    "无法确定屏幕上发生了什么时填 true，否则填 false（不要用字符串）。"
)


def _encode_image(image_path: str) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def build_vision_messages(image_path: str, timestamp: float) -> list:
    """构造 OpenAI 风格聊天消息，把关键帧以 data URI 内嵌进去。"""
    image_data = _encode_image(image_path)
    data_uri = f"data:image/png;base64,{image_data}"
    user_content = (
        f"这是录屏视频时间点 {timestamp:.1f} 秒的关键帧。"
        "请分析屏幕上的操作并输出 JSON。"
    )
    return [
        {"role": "system", "content": VISION_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": user_content},
                {"type": "image_url", "image_url": {"url": data_uri}},
            ],
        },
    ]
