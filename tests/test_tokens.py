from video2md.compose.tokens import estimate_tokens


def test_estimate_tokens_empty():
    assert estimate_tokens("") == 0


def test_estimate_tokens_chinese_one_per_char():
    # 中文约 1 字/token
    assert estimate_tokens("你好世界") == 4


def test_estimate_tokens_mixed():
    # "你好"=2 token，英文按 0.3 字/token
    n = estimate_tokens("你好 hello world")
    assert 2 < n < 8


def test_estimate_tokens_non_decreasing_with_length():
    a = estimate_tokens("短文本")
    b = estimate_tokens("这是更长的中文文本内容，用于验证估算不会随长度减少。")
    assert b >= a
