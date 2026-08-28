"""video2md command-line interface."""
import argparse
import sys
import time

from video2md.config import Config
from video2md.pipeline import run_pipeline
from video2md.render.export import export_markdown


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

    def progress(stage, i, total):
        if total:
            print(f"[{stage}] {i}/{total}", flush=True)

    try:
        start = time.time()
        print(f"处理视频: {args.video}")
        result = run_pipeline(args.video, args.output, config, progress=progress)
        elapsed = time.time() - start
        print(f"完成! 用时 {int(elapsed // 60)}m {int(elapsed % 60)}s")
        print(f"  Markdown:  {result.markdown_path}")
        print(f"  Mermaid:   {result.mermaid_path}")
        print(
            f"  关键帧: {result.frames_count}, 转写片段: {result.segments_count}, "
            f"步骤时间窗: {result.step_windows_count}, "
            f"视觉理解: {result.understanding_count}, 点击事件: {result.click_events_count}"
        )
        if args.export:
            export_path = export_markdown(result.markdown_path, args.export)
            print(f"  导出:     {export_path}")
        return 0
    except Exception as e:  # noqa: BLE001 - 面向用户的 CLI 需给出友好错误
        print(f"错误: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
