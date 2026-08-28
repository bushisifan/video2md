"""输入 token 启发式估算，用于判断 LLM 输入是否超限并切分。"""


def estimate_tokens(text: str) -> int:
    """启发式估算文本的 token 数（无需额外依赖）。

    中文约 1 字/token，其余字符约 0.3 字/token。用于判断提示词是否超过
    `max_input_tokens`，精度足以指导是否切分。
    """
    if not text:
        return 0
    cjk = sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff")
    other = len(text) - cjk
    return int(cjk + other * 0.3)
