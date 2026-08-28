"""Prompt for step synthesis (merges transcript + frame understanding + cursor)."""
from typing import Any, Dict, List

SYSTEM_PROMPT = """你是一名专业的技术文档工程师，擅长将屏幕操作录屏转换成标准操作程序（SOP）。

你会收到：
1. 视频语音转写片段（含时间戳）
2. 每个关键帧的视觉理解结果（动作/目标/细节）
3. 可选的光标点击事件

请综合这些信息，输出一个完整的 SOP 步骤树，严格按以下 JSON 结构，不要输出任何多余内容：
{
  "title": "SOP 标题",
  "purpose": "本流程目的",
  "prerequisites": ["前置条件1", "前置条件2"],
  "steps": [
    {
      "order": 1,
      "title": "步骤标题",
      "action": "具体动作描述",
      "ui_element": "界面元素路径",
      "screenshot": "images/frame_0001.png",
      "timestamp": "00:01:05",
      "warnings": ["注意点"],
      "sub_steps": ["子步骤1"],
      "branch": {"condition": "分支条件", "children": [{"order": 2, "title": "...", "action": "..."}]}
    }
  ],
  "troubleshooting": [{"issue": "常见问题", "solution": "解决方法"}]
}

要求：
- 步骤用祈使句，动作清晰可执行
- 步骤数量、顺序应尽量对齐下面"已识别步骤时间窗"（若有）
- screenshot 字段只填上面给出的关键帧路径；每步的 screenshot 必须选"时间戳落在该步时间窗内（或最近）"的帧路径，禁止跨时间窗选帧；没有合适的一帧就填空字符串
- timestamp 用 HH:MM:SS 格式
- 无法从材料中确定的内容，在 warnings 里标注“需人工复核”，不要臆造
- 分支识别不出来时，branch 置为 null
"""


def build_compose_messages(
    segments: List[Any],
    frame_understandings: List[Dict[str, Any]],
    cursor_events: List[Any],
    step_windows: List[Any] | None = None,
) -> List[Dict[str, Any]]:
    step_text = ""
    if step_windows:
        step_text = (
            "\n".join(
                f"步骤{w.order} [{w.start:.1f}-{w.end:.1f}s]: {w.title}"
                for w in step_windows
            )
            + "\n\n"
        )
    seg_text = (
        "\n".join(f"[{s.start:.1f}s - {s.end:.1f}s]: {s.text}" for s in segments)
        if segments
        else "(无语音转写，视频可能无声)"
    )
    vis_lines = []
    for fu in frame_understandings:
        vis_lines.append(
            f"[{fu.get('timestamp', 0):.1f}s] {fu.get('screenshot', '')} "
            f"action={fu.get('action', '')} target={fu.get('target', '')} "
            f"detail={fu.get('detail', '')} needs_review={fu.get('needs_review', False)}"
        )
    vis_text = "\n".join(vis_lines) if vis_lines else "(无视觉理解结果)"
    cursor_text = (
        "\n".join(f"[{c.timestamp:.1f}s] 点击 ({c.x}, {c.y})" for c in cursor_events)
        if cursor_events
        else "(无光标事件)"
    )
    user_content = (
        f"已识别步骤时间窗：\n{step_text if step_windows else '(无，按内容自行切分)'}\n\n"
        f"语音转写片段：\n{seg_text}\n\n"
        f"关键帧视觉理解：\n{vis_text}\n\n"
        f"光标点击事件：\n{cursor_text}\n\n"
        "请综合以上信息输出完整 SOP JSON。"
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]
