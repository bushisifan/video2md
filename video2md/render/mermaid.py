"""把 SOPDocument 渲染成 Mermaid 流程图（flowchart TD）。"""
from video2md.compose.schema import SOPDocument, Step


def render_mermaid(doc: SOPDocument) -> str:
    """生成包含主流程、分支、子步骤的 `flowchart TD`。

    节点 id 是确定性的（主步骤 MS1..MSn，子步骤/分支节点 SUB/BR/BC），
    保证测试与 diff 稳定。

    分支递归渲染：分支子步骤自身又有的子步骤与嵌套分支，会从该子节点的
    角度继续渲染，因此嵌套分支不会被丢弃。
    """
    lines = ["flowchart TD"]
    steps = doc.steps
    if not steps:
        lines.append("    START([开始])")
        lines.append("    END([结束])")
        lines.append("    START --> END")
        return "\n".join(lines) + "\n"

    main_ids = [f"MS{i + 1}" for i in range(len(steps))]
    counter = len(steps)

    def new_id(prefix: str) -> str:
        nonlocal counter
        counter += 1
        return f"{prefix}{counter}"

    def emit_attachments(step, parent_id, else_target):
        # else_target: 节点 id，或 None 表示用一个新的 END 节点
        for sub in step.sub_steps:
            if not sub.strip():
                continue
            sid = new_id("SUB")
            lines.append(f"    {sid}([{_escape(sub)}])")
            lines.append(f"    {parent_id} --> {sid}")
        if step.branch:
            condition = step.branch.condition or "条件"
            bid = new_id("BR")
            lines.append(f"    {bid}{{{_escape(condition)}}}")
            lines.append(f"    {parent_id} --> {bid}")
            for child in step.branch.children:
                cid = new_id("BC")
                lines.append(f"    {cid}[{_label(child)}]")
                lines.append(f"    {bid} -->|是| {cid}")
                emit_attachments(child, cid, None)
            if else_target is None:
                end_id = new_id("END")
                lines.append(f"    {end_id}([结束])")
                else_target = end_id
            lines.append(f"    {bid} -->|否| {else_target}")

    for i, step in enumerate(steps):
        lines.append(f"    {main_ids[i]}[{_label(step)}]")
        if i > 0:
            lines.append(f"    {main_ids[i - 1]} --> {main_ids[i]}")
        else_target = main_ids[i + 1] if i + 1 < len(steps) else None
        emit_attachments(step, main_ids[i], else_target)

    return "\n".join(lines) + "\n"


def _label(step: Step) -> str:
    return _escape(f"{step.order}. {step.title}")


def _escape(text: str) -> str:
    # 带引号标签可接受 () | {} ; 等字符；双引号转义为单引号，换行折叠为 <br/>
    inner = (text or "…").replace('"', "'").replace("\n", "<br/>")
    return f'"{inner}"'
