"""SOP 步骤树的数据模型（pydantic 强校验 LLM 输出）。"""
from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Branch(_StrictModel):
    condition: str = ""
    children: List["Step"] = Field(default_factory=list)


class Step(_StrictModel):
    order: int = Field(ge=1)
    title: str = Field(min_length=1)
    action: str = Field(min_length=1)
    ui_element: str = ""
    screenshot: str = ""
    timestamp: str = ""
    warnings: List[str] = Field(default_factory=list)
    sub_steps: List[str] = Field(default_factory=list)
    branch: Optional[Branch] = None


class TroubleshootingItem(_StrictModel):
    issue: str = Field(min_length=1)
    solution: str = Field(min_length=1)


class SOPDocument(_StrictModel):
    title: str = Field(min_length=1)
    purpose: str = ""
    prerequisites: List[str] = Field(default_factory=list)
    steps: List[Step] = Field(default_factory=list)
    troubleshooting: List[TroubleshootingItem] = Field(default_factory=list)


Step.model_rebuild()
