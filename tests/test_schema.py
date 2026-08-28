import pytest
from pydantic import ValidationError

from video2md.compose.schema import SOPDocument


def test_minimal_document():
    doc = SOPDocument(title="测试")
    assert doc.steps == []
    assert doc.prerequisites == []


def test_full_document_with_branch():
    data = {
        "title": "导出报表",
        "purpose": "将数据导出为PDF",
        "prerequisites": ["已登录"],
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
    }
    doc = SOPDocument.model_validate(data)
    assert doc.steps[0].branch.children[0].title == "导出"
    assert doc.troubleshooting[0].issue == "导出失败"


def test_missing_title_raises():
    with pytest.raises(ValidationError):
        SOPDocument.model_validate({})


def test_empty_title_rejected():
    with pytest.raises(ValidationError):
        SOPDocument.model_validate({"title": "", "steps": [{"order": 1, "title": "a", "action": "b"}]})


def test_empty_action_rejected():
    with pytest.raises(ValidationError):
        SOPDocument.model_validate({"title": "t", "steps": [{"order": 1, "title": "a", "action": ""}]})


def test_extra_key_rejected():
    with pytest.raises(ValidationError):
        SOPDocument.model_validate({"title": "t", "steps": [{"order": 1, "title": "a", "action": "b", "bogus": 1}]})


def test_zero_order_rejected():
    with pytest.raises(ValidationError):
        SOPDocument.model_validate({"title": "t", "steps": [{"order": 0, "title": "a", "action": "b"}]})
