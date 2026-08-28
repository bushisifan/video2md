"""video2md 命令行入口。"""
import argparse
import re
import subprocess
import sys
import time

from video2md.config import Config
from video2md.pipeline import run_pipeline
from video2md.render.export import export_markdown

# 运行前内存预检阈值(GB)：ASR 峰值 ~3.5GB + 视觉模型常驻 ~4.5GB + 系统开销，低于此值易触发 swap。
MEM_WARN_THRESHOLD_MB = 6 * 1024


def _peak_rss_mb():
    """本进程峰值内存(MB)；仅 Unix 可用，Windows 返回 None（避免引入非标准依赖）。"""
    try:
        import resource  # noqa: WPS433 - 仅 Unix 有，懒加载保证 Windows 兼容
        value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # macOS 单位为字节，Linux 为 KB
        return value / (1024 * 1024) if sys.platform == "darwin" else value / 1024
    except (ImportError, AttributeError, OSError):
        return None


def _available_memory_mb():
    """当前可用内存(MB)；支持 macOS/Linux，探测失败返回 None。"""
    try:
        if sys.platform == "darwin":
            out = subprocess.run(
                ["vm_stat"], capture_output=True, text=True, check=False
            ).stdout
            page_size = None
            pages = 0
            for line in out.splitlines():
                m = re.search(r"page size of (\d+) bytes", line)
                if m:
                    page_size = int(m.group(1))
                    continue
                m = re.match(r"Pages (free|inactive|speculative):\s+(\d+)", line.strip())
                if m:
                    pages += int(m.group(2))
            if page_size is None:
                return None
            return pages * page_size / (1024 * 1024)
        if sys.platform.startswith("linux"):
            with open("/proc/meminfo", encoding="utf-8") as f:
                for line in f:
                    if line.startswith("MemAvailable:"):
                        return int(line.split()[1]) / 1024  # kB -> MB
    except (OSError, ValueError, subprocess.SubprocessError):
        return None
    return None


def _warn_low_memory():
    """可用内存低于阈值时向 stderr 给出建议（只警告，不阻断）。"""
    avail = _available_memory_mb()
    if avail is None or avail >= MEM_WARN_THRESHOLD_MB:
        return
    print(
        f"警告: 当前可用内存约 {avail / 1024:.1f} GB，低于推荐的 {MEM_WARN_THRESHOLD_MB / 1024:.0f} GB。\n"
        "  ASR 阶段峰值约 3.5 GB、视觉模型常驻约 4.5 GB，内存不足会触发 swap 导致 ASR 停滞。\n"
        "  建议先关闭 Chrome/Safari/IDE 等大内存应用，或换用更小的视觉模型（如 qwen2.5vl:1.5b）。",
        file=sys.stderr,
    )


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="将已有屏幕录屏视频转换为带截图的逐步SOP文档与Mermaid流程图（纯本地）"
    )
    parser.add_argument("video", help="输入视频文件路径 (MP4/MOV/AVI)")
    parser.add_argument("-o", "--output", default="sop_output", help="输出目录 (默认: sop_output)")
    parser.add_argument(
        "-c", "--config", default=None,
        help="配置文件路径 (默认: 包内 config.yaml)",
    )
    parser.add_argument(
        "--export", choices=["pdf", "docx", "html"], default=None,
        help="用 pandoc 将 Markdown 导出为 PDF/DOCX/HTML（需安装 pandoc）",
    )
    args = parser.parse_args(argv)

    try:
        # config.yaml 位于包内；未指定 --config 时使用默认路径
        config = Config.load(args.config) if args.config else Config.load()
    except Exception as e:  # noqa: BLE001 - 面向用户的 CLI 需给出友好错误
        print(f"错误: 配置加载失败: {e}", file=sys.stderr)
        return 1

    # 运行前内存预检（仅警告）
    _warn_low_memory()

    # 各阶段计时：progress 回调在 stage 切换时结算上一阶段耗时
    timings: dict[str, float] = {}
    order: list[str] = []
    cur_stage: str | None = None
    stage_start: float = 0.0

    def progress(stage, i, total):
        nonlocal cur_stage, stage_start
        now = time.time()
        if stage != cur_stage:
            if cur_stage is not None:
                timings[cur_stage] = now - stage_start
            cur_stage = stage
            order.append(stage)
            stage_start = now
        if total:
            print(f"[{stage}] {i}/{total}", flush=True)

    try:
        start = time.time()
        print(f"处理视频: {args.video}")
        result = run_pipeline(args.video, args.output, config, progress=progress)
        elapsed = time.time() - start
        if cur_stage is not None:
            timings[cur_stage] = time.time() - stage_start
        print(f"完成! 用时 {int(elapsed // 60)}m {int(elapsed % 60)}s")
        print(f"  Markdown:  {result.markdown_path}")
        print(f"  Mermaid:   {result.mermaid_path}")
        print(
            f"  关键帧: {result.frames_count}, 转写片段: {result.segments_count}, "
            f"步骤时间窗: {result.step_windows_count}, "
            f"视觉理解: {result.understanding_count}, 点击事件: {result.click_events_count}"
        )
        if timings:
            print("  各阶段耗时:")
            for s in order:
                print(f"    {s:<18} {timings[s]:>8.1f}s")
        peak = _peak_rss_mb()
        if peak is not None:
            print(f"  本进程峰值内存: {peak:.0f} MB（不含 Ollama/vLLM 等独立进程）")
        if args.export:
            export_path = export_markdown(result.markdown_path, args.export)
            print(f"  导出:     {export_path}")
        return 0
    except Exception as e:  # noqa: BLE001 - 面向用户的 CLI 需给出友好错误
        print(f"错误: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
