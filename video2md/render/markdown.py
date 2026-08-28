"""把 SOPDocument 渲染成 Markdown（完整流程文档模板）。

结构：文档信息 / 目的 / 前置条件 / 流程图(Mermaid) / 操作步骤(速览表+详情) /
疑难解答 / 检查表 / 版本历史。
"""
from datetime import datetime

from video2md.compose.schema import SOPDocument


def _single_line(text: str) -> str:
    return " ".join(text.split())


def _cell(text: str) -> str:
    return _single_line(text).replace("|", "\\|")


def render_markdown(
    doc: SOPDocument,
    mermaid_code: str | None = None,
    doc_number: str = "SOP-001",
    version: str = "1.0",
    author: str = "",
    scope: str = "",
    generated_date: str | None = None,
) -> str:
    """渲染完整流程文档模板。"""
    if generated_date is None:
        generated_date = datetime.now().strftime("%Y-%m-%d")
    author_disp = author or "待填写"
    scope_disp = scope or "待填写"

    lines = [f"# {_single_line(doc.title)}", ""]

    # 文档信息
    lines += ["## 文档信息", ""]
    lines += [
        "| 项 | 值 |",
        "|---|---|",
        f"| 编号 | {_cell(doc_number)} |",
        f"| 版本 | {_cell(version)} |",
        f"| 编写人 | {_cell(author_disp)} |",
        f"| 日期 | {_cell(generated_date)} |",
        f"| 适用范围 | {_cell(scope_disp)} |",
        "",
    ]

    # 目的 / 前置条件
    if doc.purpose:
        lines += ["## 目的", doc.purpose, ""]
    if doc.prerequisites:
        lines += ["## 前置条件"]
        lines += [f"- {_single_line(p)}" for p in doc.prerequisites]
        lines += [""]

    # 流程图
    if mermaid_code:
        lines += ["## 流程图", ""]
        lines += ["```mermaid"]
        lines += mermaid_code.strip().split("\n")
        lines += ["```", ""]

    # 操作步骤：速览表 + 详情
    lines += ["## 操作步骤", ""]
    if doc.steps:
        lines += ["### 步骤速览", ""]
        lines += ["| 步骤 | 操作 | 界面元素 | 时间 |", "|---|---|---|---|"]
        for step in doc.steps:
            lines.append(
                f"| {step.order} | {_cell(step.action)} | {_cell(step.ui_element)} | {_cell(step.timestamp)} |"
            )
        lines += [""]
    for step in doc.steps:
        _render_step(lines, step, level=3)

    # 疑难解答
    if doc.troubleshooting:
        lines += ["## 疑难解答", ""]
        lines += ["| 问题 | 解决方法 |", "|---|---|"]
        lines += [f"| {_cell(t.issue)} | {_cell(t.solution)} |" for t in doc.troubleshooting]

    # 检查表
    if doc.steps:
        lines += ["## 检查表", ""]
        lines += [f"- [ ] 步骤 {step.order}: {_single_line(step.title)}" for step in doc.steps]
        lines += [""]

    # 版本历史
    lines += ["## 版本历史", ""]
    lines += ["| 版本 | 日期 | 变更说明 | 作者 |", "|---|---|---|---|"]
    lines += [f"| {_cell(version)} | {_cell(generated_date)} | 初版 | {_cell(author_disp)} |"]

    return "\n".join(lines).rstrip() + "\n"


def _render_step(lines, step, level):
    heading = "#" * level
    lines.append(f"{heading} 步骤 {step.order}: {_single_line(step.title)}")
    lines.append("")
    lines.append(f"- **动作**: {_single_line(step.action)}")
    if step.ui_element:
        lines.append(f"- **界面元素**: {_single_line(step.ui_element)}")
    if step.timestamp:
        lines.append(f"- **时间戳**: {_single_line(step.timestamp)}")
    if step.warnings:
        lines += ["- **注意**:"]
        lines += [f"  - {_single_line(w)}" for w in step.warnings]
    lines.append("")
    if step.screenshot:
        lines.append(f"![截图]({step.screenshot})")
        lines.append("")
    if step.sub_steps:
        lines.append("**子步骤**:")
        lines += [f"{i}. {s}" for i, s in enumerate(step.sub_steps, 1)]
        lines.append("")
    if step.branch:
        lines.append(f"**分支**: 如果 {_single_line(step.branch.condition)}".rstrip())
        lines.append("")
        for child in step.branch.children:
            _render_step(lines, child, level=level + 1)
