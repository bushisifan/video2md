"""video2md 端到端管线编排。

方案B（语音驱动抽帧）：抽音 → ASR(逐句时间戳) → LLM 步骤时间点检测 →
按步骤时间点抽帧（场景变化兜底）→ 视觉理解 → 合成(带时间窗约束) → 渲染。
这样"步骤 ↔ 截图"按时间对齐，规避同画面多操作导致的文图不一致。
"""
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional

from video2md.asr.sensevoice import SenseVoiceTranscriber
from video2md.compose.schema import SOPDocument
from video2md.compose.step_detector import StepDetector, StepWindow
from video2md.compose.synthesizer import StepSynthesizer
from video2md.config import Config
from video2md.preprocess.audio import extract_audio
from video2md.preprocess.cursor import CursorDetector
from video2md.preprocess.frames import FrameExtractor, KeyFrame
from video2md.render.markdown import render_markdown
from video2md.render.mermaid import render_mermaid
from video2md.vision.client import VisionClient

logger = logging.getLogger(__name__)

Progress = Optional[Callable[[str, int, int], None]]


@dataclass
class PipelineResult:
    markdown_path: str
    mermaid_path: str
    sop: SOPDocument
    frames_count: int
    segments_count: int
    step_windows_count: int
    understanding_count: int
    click_events_count: int


def run_pipeline(
    video_path: str,
    output_dir: str,
    config: Config,
    progress: Progress = None,
) -> PipelineResult:
    """方案B：语音驱动抽帧。抽音→ASR→步骤时间点→按时间抽帧(场景兜底)→视觉→合成→渲染。"""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    images_dir = out / config.render.images_dir

    def report(stage, i=0, total=0):
        if progress:
            progress(stage, i, total)

    # ① preprocess: audio (+ cursor)。抽帧延后到步骤时间点确定后。
    audio_path: Optional[str] = None
    try:
        audio_path = extract_audio(
            video_path, str(out / "audio.wav"), config.preprocess.audio_sample_rate
        )
    except RuntimeError:
        audio_path = None  # 无声视频 → 优雅降级

    cursor_events = []
    if config.cursor.enabled:
        try:
            detector = CursorDetector()
            cursor_events = detector.detect(video_path)
        except Exception as e:  # noqa: BLE001 - 光标检测为可选启发式信号
            logger.warning("光标检测失败，忽略：%s", e)
            cursor_events = []

    # ② ASR（带逐句时间戳；失败降级为无转写）
    report("transcribe", 0, 1)
    segments = []
    if audio_path:
        try:
            transcriber = SenseVoiceTranscriber(
                model=config.asr.model,
                device=config.asr.device,
                sample_rate=config.asr.sample_rate,
                vad_model=config.asr.vad_model,
                punc_model=config.asr.punc_model,
                sentence_timestamp=config.asr.sentence_timestamp,
            )
            segments = transcriber.transcribe(audio_path)
        except Exception as e:  # noqa: BLE001 - 语音转写为可选信号
            logger.warning("语音转写失败，降级为无转写：%s", e)
            segments = []
    report("transcribe", 1, 1)

    # ③ LLM 步骤时间点检测（失败则退化为场景变化抽帧）
    report("detect_steps", 0, 1)
    step_windows: List[StepWindow] = []
    if segments:
        try:
            detector = StepDetector(
                base_url=config.compose.base_url,
                api_key=config.compose.api_key,
                model=config.compose.model,
                temperature=config.compose.temperature,
                max_tokens=config.compose.max_tokens,
                timeout=config.compose.timeout,
            )
            step_windows = detector.detect(segments)
        except Exception as e:  # noqa: BLE001 - 步骤切分为可选环节
            logger.warning("步骤切分失败，退化为场景变化抽帧：%s", e)
            step_windows = []
    report("detect_steps", 1, 1)

    # ④ 抽帧：语音步骤时间点优先，场景变化兜底
    report("extract_frames", 0, 1)
    extractor = FrameExtractor(
        scene_threshold=config.preprocess.scene_threshold,
        interval_seconds=config.preprocess.interval_seconds,
        resize_width=config.preprocess.resize_width,
        min_frames=config.preprocess.min_frames,
    )
    if step_windows:
        timestamps = [round((w.start + w.end) / 2, 3) for w in step_windows]
        keyframes = extractor.extract_at_timestamps(
            video_path, timestamps, str(images_dir)
        )
        if not keyframes:
            logger.warning("按步骤时间点抽帧失败，退化为场景变化抽帧")
            keyframes = extractor.extract(video_path, str(images_dir))
    else:
        keyframes = extractor.extract(video_path, str(images_dir))
    report("extract_frames", 1, 1)

    # 截图路径相对化，供 Markdown 嵌入
    for kf in keyframes:
        kf.image_path = Path(kf.image_path).relative_to(out).as_posix()

    # ⑤ 视觉理解：逐个理解关键帧
    vision = VisionClient(
        base_url=config.vision.base_url,
        api_key=config.vision.api_key,
        model=config.vision.model,
        temperature=config.vision.temperature,
        max_tokens=config.vision.max_tokens,
        timeout=config.vision.timeout,
    )
    understandings = []
    for i, kf in enumerate(keyframes):
        report("understand_frame", i + 1, len(keyframes))
        abs_path = str(out / Path(kf.image_path))
        try:
            parsed = vision.understand_frame(abs_path, kf.timestamp)
        except Exception as e:  # noqa: BLE001 - 视觉理解为可选信号
            logger.warning("视觉理解失败，frame %.1fs：%s", kf.timestamp, e)
            parsed = {
                "action": "other",
                "target": "",
                "detail": f"视觉理解失败: {e}",
                "needs_review": True,
            }
        parsed["timestamp"] = kf.timestamp
        parsed["screenshot"] = kf.image_path
        understandings.append(parsed)

    # ⑥ compose（带步骤时间窗约束选帧）
    report("synthesize", 0, 1)
    synthesizer = StepSynthesizer(
        base_url=config.compose.base_url,
        api_key=config.compose.api_key,
        model=config.compose.model,
        temperature=config.compose.temperature,
        max_tokens=config.compose.max_tokens,
        timeout=config.compose.timeout,
    )
    sop = synthesizer.synthesize(segments, understandings, cursor_events, step_windows)
    _sanitize_screenshots(sop, {kf.image_path for kf in keyframes})
    report("synthesize", 1, 1)

    # ⑦ 渲染：写 Markdown 与 Mermaid 流程图
    report("render", 0, 1)
    mermaid_text = render_mermaid(sop)
    markdown_path = out / "SOP.md"
    markdown_path.write_text(
        render_markdown(sop, mermaid_code=mermaid_text), encoding="utf-8"
    )
    mermaid_path = out / "flowchart.mmd"
    mermaid_path.write_text(mermaid_text, encoding="utf-8")
    report("render", 1, 1)

    return PipelineResult(
        markdown_path=str(markdown_path),
        mermaid_path=str(mermaid_path),
        sop=sop,
        frames_count=len(keyframes),
        segments_count=len(segments),
        step_windows_count=len(step_windows),
        understanding_count=len(understandings),
        click_events_count=len(cursor_events),
    )


def _iter_steps(steps):
    for step in steps:
        yield step
        if step.branch:
            yield from _iter_steps(step.branch.children)


def _sanitize_screenshots(sop: SOPDocument, valid: set) -> None:
    """清掉指向未抽取帧的截图引用。

    LLM 被要求只使用给定的帧路径，但仍可能幻觉出别的路径；我们把未知路径
    置空，避免 Markdown 里出现裂图。
    """
    for step in _iter_steps(sop.steps):
        if step.screenshot not in valid:
            step.screenshot = ""
