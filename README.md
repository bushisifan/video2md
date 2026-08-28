# video2md

把已有屏幕录屏视频（MP4/MOV/AVI）自动转换为**带截图的逐步 SOP 文档 + Mermaid 流程图**。中文优先、模型主要本地部署、视频不出内网。

```
录屏视频 ──► ① 抽音 + ASR(逐句时间戳) ──► ② LLM 步骤时间窗切分
               │
               ▼
   ③ 按步骤时间点抽帧(场景变化兜底) ──► ④ 视觉理解 ──► ⑤ 步骤合成 ──► ⑥ SOP.md + flowchart.mmd
```

## 核心特性

- **语音驱动抽帧（方案B）**：先用本地 ASR（FunASR paraformer-zh）拿到逐句时间戳，再用 LLM 把转写切成"操作步骤 + 时间区间"，**按步骤时间点抽帧**——解决"同画面多次操作、场景变化抽帧漏图"导致的文图不一致。
- **完整流程文档模板**：文档信息（编号/版本/编写人/适用范围）、目的、前置条件、流程图（Mermaid）、步骤速览表 + 步骤详情、疑难解答、检查表、版本历史。
- **优雅降级**：无声视频/ASR 失败 → 退化为场景变化抽帧；视觉理解失败 → 该步标记"需人工复核"。
- **纯本地优先**：ASR、视觉理解均本地部署；步骤合成可用本地 LLM 或云端 OpenAI 兼容 API。
- 输出 Markdown + Mermaid，可选 pandoc 导出 PDF / DOCX / HTML。

## 快速开始

完整跨平台（macOS / Windows）部署见 **[docs/部署指南.md](docs/部署指南.md)**。以下为 macOS 快捷路径：

```bash
# 1. 依赖：Python 3.10+、ffmpeg、Ollama
brew install ffmpeg ollama
ollama pull qwen2.5vl:3b

# 2. 安装 Python 包（含 torch/funasr）
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
pip install torch torchaudio funasr modelscope   # funasr 模型 ~1.3GB 首次自动下载

# 3. 配置模型端点（写到一个不入库的本地配置，API Key 放这里）
cp video2md/config.yaml config.local.yaml
#    编辑 config.local.yaml：
#      vision.base_url   = http://localhost:11434/v1     # Ollama
#      vision.model      = qwen2.5vl:3b
#      compose.base_url  = https://api.deepseek.com/v1    # 或用本地 vLLM
#      compose.api_key   = <你的密钥>
#   然后运行：
video2md 你的录屏.mp4 -o output -c config.local.yaml
```

输出到 `output/`：

- `SOP.md` —— 完整流程文档（图片相对路径 `images/`）
- `flowchart.mmd` —— Mermaid 流程图
- `images/frame_*.png` —— 每步代表性截图

```bash
# 可选：用 pandoc 导出 PDF/DOCX/HTML
brew install pandoc
video2md 你的录屏.mp4 -o output -c config.local.yaml --export pdf
```

## 配置

默认配置在 `video2md/config.yaml`（包内），通过 `-c <path>` 覆盖；未配置的 section 使用代码内默认值。

| Section | 默认 | 说明 |
|---|---|---|
| `asr` | `paraformer-zh` + `fsmn-vad` + `ct-punc-c`，`sentence_timestamp: true` | 本地 CPU 转写，逐句时间戳（方案B 必需） |
| `vision` | `http://localhost:8000/v1` / `Qwen/Qwen2.5-VL-7B-Instruct` | 视觉理解端点（可指向 Ollama / vLLM） |
| `compose` | `http://localhost:8000/v1` / `Qwen/Qwen2.5-7B-Instruct` | 步骤合成 LLM 端点；`max_input_tokens: 24000`（输入超限自动切分，不依赖模型大上下文） |
| `preprocess` | `scene_threshold: 30`, `interval_seconds: 2`, `resize_width: 768` | 抽帧/场景变化参数 |
| `cursor` | `enabled: false` | 光标点击检测（启发式，默认关） |
| `render` | `images_dir: images` | 截图目录名 |

> **API Key 安全**：`config.local.yaml` 已加入 `.gitignore`。切勿把真实密钥写进提交的配置。

## 运行时资源占用（实测）

> 实测环境：16GB 内存 / 10 核 M 系列 Mac / CPU 推理（无 GPU 加速）。资源占用与视频时长、模型大小强相关。

| 阶段 | 峰值内存 | CPU | 说明 |
|---|---|---|---|
| ASR（FunASR paraformer-zh + vad + punc，CPU） | **约 3.5 GB** | 单核~多核，约 **27× 实时**（30s 音频 → 1.1s） | 模型加载约 8s；整段视频一次性载入内存 |
| 视觉理解（Ollama qwen2.5vl:3b，CPU） | **约 4.5 GB**（常驻） | 每帧 **约 6s**（768px PNG） | 模型加载后常驻内存 |
| 步骤合成（DeepSeek API） | 本地近 0（网络请求） | — | 时长取决于 token，约 0.5–1 min |
| 抽帧 / 渲染（ffmpeg + OpenCV） | < 1 GB | 秒级 | — |

**端到端实测**：16 分钟 CC-Switch 录屏（17 步、17 帧）全程 **约 3m23s**，视觉理解 0 退化。

> CLI 运行前会自动检查可用内存（低于 6GB 警告并建议关闭大内存应用）；运行后会打印
> **各阶段耗时**与**本进程峰值内存**，便于小组排障与调参。

**内存建议**：

- **16GB 机型**：可跑 qwen2.5vl:3b 视觉 + 本地 ASR，但**运行前请关闭 Chrome/Safari/IDE 等大内存应用**——ASR 峰值 ~3.5GB + 视觉模型常驻 ~4.5GB，再加系统占用容易触发 swap。
- **8GB 机型**：改用更小视觉模型（`qwen2.5vl:1.5b`），或视觉/合成走云端 API。
- **GPU 机型**：Ollama/vLLM 会自动使用 GPU，视觉每帧从 ~6s 降至毫秒级，内存压力集中在显存。

## 项目结构

```
video2md/
├── video2md/
│   ├── asr/sensevoice.py      # FunASR 转写（逐句时间戳）
│   ├── compose/
│   │   ├── schema.py          # SOPDocument / Step 数据模型（pydantic 强校验）
│   │   ├── step_detector.py   # LLM 步骤时间窗切分（方案B）
│   │   ├── synthesizer.py     # 步骤合成（带时间窗约束选帧）
│   │   └── prompt.py
│   ├── preprocess/            # 抽音 / 抽帧 / 场景变化 / 光标检测
│   ├── vision/                # 视觉理解客户端（OpenAI 兼容）
│   ├── render/                # Markdown 模板 + Mermaid + pandoc 导出
│   ├── pipeline.py            # 端到端管线编排（方案B）
│   ├── cli.py                 # video2md 命令行
│   └── config.py / config.yaml
├── tests/                     # pytest 单元测试
└── pyproject.toml
```

## 开发

```bash
pip install -e ".[dev]"
pytest
```

## 边界情况

- 无声视频 → 退化为场景变化抽帧 + 纯视觉/光标推断
- ASR/视觉失败 → 单步降级，文档标注"需人工复核"，不臆造内容
- 长视频 → 建议分块处理（超过 30 分钟注意内存峰值）
- 截图引用 → 只保留实际抽到的帧，LLM 幻觉路径会被自动清空（见 `pipeline._sanitize_screenshots`）
