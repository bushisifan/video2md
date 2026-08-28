from video2md.compose.schema import SOPDocument
from video2md.render.mermaid import render_mermaid


def _doc(with_branch=False):
    steps = [
        {
            "order": 1,
            "title": "打开设置",
            "action": "点击齿轮图标",
            "screenshot": "images/frame_0001.png",
            "timestamp": "00:01:05",
            "sub_steps": ["打开系统设置"],
        },
        {"order": 2, "title": "输入账号", "action": "在输入框输入账号"},
    ]
    if with_branch:
        steps[0]["branch"] = {
            "condition": "如果需要导出",
            "children": [{"order": 3, "title": "导出", "action": "点击导出按钮"}],
        }
    return SOPDocument.model_validate({
        "title": "导出报表",
        "purpose": "",
        "prerequisites": [],
        "steps": steps,
    })


def test_render_mermaid_linear():
    mmd = render_mermaid(_doc())
    assert mmd.startswith("flowchart TD")
    assert "MS1[" in mmd and "MS2[" in mmd
    assert "MS1 --> MS2" in mmd
    assert "SUB" in mmd  # 子步骤节点
    assert "1. 打开设置" in mmd


def test_render_mermaid_branch():
    mmd = render_mermaid(_doc(with_branch=True))
    assert "BR" in mmd
    assert "-->|是|" in mmd
    assert "-->|否|" in mmd


def test_render_mermaid_empty():
    doc = SOPDocument(title="空流程")
    mmd = render_mermaid(doc)
    assert "flowchart TD" in mmd
    assert "START" in mmd


def test_render_mermaid_quotes_special_chars():
    doc = SOPDocument.model_validate({
        "title": "T",
        "steps": [{
            "order": 1,
            "title": "设置(齿轮图标)",
            "action": "点击",
            "sub_steps": ["确认 (版本信息)"],
            "branch": {"condition": "如果窗口|弹窗", "children": [{"order": 2, "title": "B", "action": "x"}]},
        }],
    })
    mmd = render_mermaid(doc)
    assert '"1. 设置(齿轮图标)"' in mmd
    assert '"确认 (版本信息)"' in mmd
    assert '"如果窗口|弹窗"' in mmd


def test_render_mermaid_branch_on_last_step():
    doc = SOPDocument.model_validate({
        "title": "T",
        "steps": [
            {"order": 1, "title": "A", "action": "a"},
            {"order": 2, "title": "B", "action": "b", "branch": {"condition": "需要", "children": [{"order": 3, "title": "C", "action": "c"}]}},
        ],
    })
    mmd = render_mermaid(doc)
    assert "-->|否|" in mmd
    assert "结束" in mmd


def test_render_mermaid_sub_and_branch_same_step():
    doc = SOPDocument.model_validate({
        "title": "T",
        "steps": [{
            "order": 1,
            "title": "A",
            "action": "a",
            "sub_steps": ["s1"],
            "branch": {"condition": "需要", "children": [{"order": 2, "title": "C", "action": "c"}]},
        }],
    })
    mmd = render_mermaid(doc)
    assert "SUB" in mmd
    assert "BR" in mmd
    assert "BC" in mmd


def test_render_mermaid_nested_branch():
    doc = SOPDocument.model_validate({
        "title": "T",
        "steps": [{
            "order": 1,
            "title": "A",
            "action": "a",
            "branch": {
                "condition": "需要导出",
                "children": [{
                    "order": 2,
                    "title": "B",
                    "action": "b",
                    "sub_steps": ["子步骤1", "子步骤2"],
                    "branch": {
                        "condition": "是否需要确认",
                        "children": [{"order": 3, "title": "C", "action": "c"}],
                    },
                }],
            },
        }],
    })
    mmd = render_mermaid(doc)
    # 分支子步骤被渲染为 SUB 节点（此处共 2 个）
    assert mmd.count('(["子步骤') == 2
    # 外层与内层分支条件都渲染为 BR 节点
    assert '"需要导出"' in mmd
    assert '"是否需要确认"' in mmd
    # 两层分支各自都有 -->|是| 边与 -->|否| 边
    assert mmd.count("-->|是|") == 2
    assert mmd.count("-->|否|") == 2
    # 两层分支的否边都终止于一个 结束 节点
    assert mmd.count("结束") == 2
