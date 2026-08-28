from video2md.compose.schema import SOPDocument
from video2md.render.markdown import render_markdown


def _doc():
    return SOPDocument.model_validate({
        "title": "导出报表",
        "purpose": "将数据导出为PDF",
        "prerequisites": ["已登录", "有导出权限"],
        "steps": [{
            "order": 1,
            "title": "打开设置",
            "action": "点击齿轮图标",
            "ui_element": "设置 > 齿轮图标",
            "screenshot": "images/frame_0001.png",
            "timestamp": "00:01:05",
            "warnings": ["需管理员权限"],
            "sub_steps": ["打开系统设置"],
            "branch": {
                "condition": "如果需要导出",
                "children": [
                    {"order": 2, "title": "导出", "action": "点击导出按钮"}
                ],
            },
        }],
        "troubleshooting": [{"issue": "导出失败", "solution": "重试"}],
    })


def test_render_markdown():
    md = render_markdown(_doc())
    assert md.startswith("# 导出报表")
    assert "## 目的" in md
    assert "- 已登录" in md
    assert "### 步骤 1: 打开设置" in md
    assert "- **动作**: 点击齿轮图标" in md
    assert "- **界面元素**: 设置 > 齿轮图标" in md
    assert "![截图](images/frame_0001.png)" in md
    assert "**子步骤**:" in md
    assert "如果 如果需要导出" in md
    assert "## 疑难解答" in md
    assert "| 导出失败 | 重试 |" in md


def test_render_markdown_escapes_table_cells():
    doc = SOPDocument.model_validate({
        "title": "T",
        "troubleshooting": [
            {"issue": "问题 A | 带管道", "solution": "重试\n换行"}
        ],
    })
    md = render_markdown(doc)
    assert "| 问题 A \\| 带管道 | 重试 换行 |" in md


def test_render_markdown_numbers_sub_steps():
    doc = SOPDocument.model_validate({
        "title": "T",
        "steps": [{"order": 1, "title": "步骤A", "action": "动作", "sub_steps": ["第一步", "第二步", "第三步"]}],
    })
    md = render_markdown(doc)
    assert "1. 第一步" in md
    assert "2. 第二步" in md
    assert "3. 第三步" in md


def test_render_markdown_full_template():
    """完整流程文档模板：文档信息 / 流程图 / 步骤速览 / 检查表 / 版本历史。"""
    mermaid = "flowchart TD\n    A[开始] --> B[结束]\n"
    md = render_markdown(_doc(), mermaid_code=mermaid, generated_date="2026-08-28")
    assert "## 文档信息" in md
    assert "| 编号 | SOP-001 |" in md
    assert "| 版本 | 1.0 |" in md
    assert "| 日期 | 2026-08-28 |" in md
    assert "## 流程图" in md
    assert "```mermaid" in md
    assert "flowchart TD" in md
    assert "### 步骤速览" in md
    assert "| 步骤 | 操作 | 界面元素 | 时间 |" in md
    assert "| 1 | 点击齿轮图标 | 设置 > 齿轮图标 | 00:01:05 |" in md
    assert "## 检查表" in md
    assert "- [ ] 步骤 1: 打开设置" in md
    assert "## 版本历史" in md
    assert "| 1.0 | 2026-08-28 | 初版 |" in md


def test_render_markdown_metadata():
    """自定义编号/编写人/适用范围反映到文档信息。"""
    md = render_markdown(
        _doc(),
        doc_number="SOP-007",
        author="张三",
        scope="IT 运维部门",
        generated_date="2026-08-28",
    )
    assert "| 编号 | SOP-007 |" in md
    assert "| 编写人 | 张三 |" in md
    assert "| 适用范围 | IT 运维部门 |" in md
