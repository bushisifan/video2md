"""用 pandoc 把 Markdown 导出为 PDF/Word/HTML（可选依赖）。"""
import shutil
import subprocess
from pathlib import Path

SUPPORTED = {"pdf", "docx", "html"}


def export_markdown(markdown_path: str, output_format: str = "pdf") -> str:
    """用 pandoc 转换 Markdown 文件；pandoc 不可用时抛异常。

    返回生成文件的路径。
    """
    out_fmt = output_format.strip().lower().lstrip(".")
    if out_fmt not in SUPPORTED:
        raise ValueError(f"Unsupported format: {output_format}")
    if shutil.which("pandoc") is None:
        raise RuntimeError("pandoc not found in PATH; install pandoc or skip export")
    src = Path(markdown_path)
    out = src.with_suffix(f".{out_fmt}")
    try:
        result = subprocess.run(
            ["pandoc", str(src), "-o", str(out)],
            capture_output=True, text=True, errors="replace", timeout=300,
        )
    except subprocess.TimeoutExpired:
        raise RuntimeError(f"pandoc timed out converting: {src}") from None
    if result.returncode != 0:
        raise RuntimeError(f"pandoc failed: {result.stderr}")
    if not out.exists():
        raise RuntimeError(f"pandoc produced no output: {out}")
    return str(out)
