# ComfyUI FunASR — 离线中文语音转文字节点

基于 [FunASR](https://github.com/modelscope/FunASR) 的 ComfyUI 离线中文语音识别自定义节点。无需联网，本地推理，支持 VAD（语音端点检测）、标点恢复和热词增强。

## 功能特性

- **完全离线** — 模型下载后无需联网即可使用，数据不离开本地
- **自动下载模型** — 首次运行时自动从 ModelScope 下载所需模型，后续运行直接加载本地缓存
- **高精度识别** — 基于 Paraformer-Large，中文识别准确率业界领先
- **VAD 语音端点检测** — 自动检测语音片段，过滤静音，提升长音频处理效率
- **标点恢复** — 自动为转录文本添加标点符号，输出可直接阅读
- **热词增强** — 支持自定义热词列表，提升特定领域词汇的识别准确率
- **多格式支持** — WAV、MP3、FLAC、OGG、OPUS、M4A、AAC、WMA
- **自动重采样** — 输入音频自动转为 16kHz 单声道，兼容任意采样率
- **GPU 加速** — 自动使用 CUDA GPU 加速推理（RTX 4090 上 RTF ≈ 0.004）

## 提供的节点

| 节点名称 | 类型 | 说明 |
|---------|------|------|
| **FunASR Speech to Text** | 接收 AUDIO 输入 | 连接 LoadAudio 等输出 AUDIO 的节点 |
| **FunASR Speech to Text (File)** | 文件选择输入 | 直接从 ComfyUI input 目录选择音频文件，支持拖拽上传 |

两个节点均输出 STRING 类型文本，可连接任意接受文本输入的下游节点。

## 使用的模型

| 用途 | ModelScope 模型 ID | 本地路径（相对于 ComfyUI 根目录） | 大小 |
|------|-------------------|--------------------------------|------|
| ASR 语音识别 | `damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch` | `models/funasr/paraformer-zh` | ~849 MB |
| VAD 语音端点检测 | `damo/speech_fsmn_vad_zh-cn-16k-common-pytorch` | `models/funasr/fsmn-vad` | ~3.9 MB |
| 标点恢复 | `damo/punc_ct-transformer_cn-en-common-vocab471067-large` | `models/funasr/ct-punc` | ~1.2 GB |

模型总计约 **2 GB**，首次运行时自动下载，之后离线使用。

## 安装

### 1. 克隆到 ComfyUI 的 custom_nodes 目录

```bash
cd /path/to/ComfyUI/custom_nodes
git clone https://github.com/<your-username>/ComfyUI-FunASR.git
# 或直接复制
cp -r /path/to/ComfyUI-FunASR /path/to/ComfyUI/custom_nodes/
```

### 2. 安装 Python 依赖

```bash
cd /path/to/ComfyUI/custom_nodes/ComfyUI-FunASR
pip install -r requirements.txt
```

依赖列表：
- `funasr>=1.0.0` — FunASR 语音识别框架
- `modelscope` — 模型下载（仅首次运行需要）
- `torchaudio` — 音频加载与重采样

### 3. 启动 ComfyUI

```bash
cd /path/to/ComfyUI
python main.py
```

首次使用节点时，会自动从 ModelScope 下载模型到 `models/funasr/` 目录。下载完成后后续运行不再需要联网。

## 使用方法

### 方法一：连接 LoadAudio 节点

1. 添加 **LoadAudio** 节点，选择或上传音频文件
2. 添加 **FunASR Speech to Text** 节点
3. 将 LoadAudio 的 `AUDIO` 输出连接到 FunASR 的 `audio` 输入
4. （可选）在 `hotword` 中输入热词，每行一个
5. 运行工作流，转录结果在节点 UI 中显示

### 方法二：直接选择文件

1. 将音频文件放入 ComfyUI 的 `input/` 目录（或通过节点拖拽上传）
2. 添加 **FunASR Speech to Text (File)** 节点
3. 在下拉菜单中选择音频文件
4. 运行工作流

### 工作流示例

```
┌──────────────┐      AUDIO      ┌─────────────────────────┐
│   LoadAudio  │ ──────────────▸ │  FunASR Speech to Text  │
│              │                 │                         │
│ test_audio   │                 │  hotword: (可选)        │
└──────────────┘                 └────────────┬────────────┘
                                              │ STRING
                                              ▼
                                    ┌─────────────────┐
                                    │  转录文本输出     │
                                    └─────────────────┘
```

预置工作流文件位于 `user/default/workflows/FunASR_Speech_to_Text.json`，可在 ComfyUI Web UI 的工作流菜单中直接加载。

## 热词使用

热词（Hotword）功能可以提升特定词汇的识别准确率，适用于专业术语、品牌名、人名等场景。

在节点的 `hotword` 输入框中，每行写一个热词：

```
ComfyUI
FunASR
Paraformer
语音识别
```

## 配置说明

### 模型目录

模型默认存储在 ComfyUI 的 `models/funasr/` 下。如需修改，编辑 `nodes.py` 中的 `MODELS` 字典：

```python
FUNASR_MODEL_DIR = os.path.join(folder_paths.models_dir, "funasr")
```

### 使用代理下载模型

如果网络环境需要代理，在启动 ComfyUI 前设置环境变量：

```bash
export http_proxy=http://your-proxy:port
export https_proxy=http://your-proxy:port
python main.py
```

### 禁用 FunASR 更新检查

FunASR 启动时会检查新版本。如需禁用，在代码中设置：

```python
model = AutoModel(..., disable_update=True)
```

## 性能参考

| 硬件 | 音频时长 | 推理耗时 | RTF |
|------|---------|---------|-----|
| RTX 4090 | 30.2s | 0.136s | 0.004 |
| CPU (参考) | 30.2s | ~3-5s | ~0.1-0.15 |

> RTF (Real-Time Factor) = 推理耗时 / 音频时长，越小越快。

## 项目结构

```
ComfyUI-FunASR/
├── __init__.py          # ComfyUI 入口，导出 NODE_CLASS_MAPPINGS
├── nodes.py             # 节点实现（自动下载、模型加载、转录逻辑）
├── requirements.txt     # Python 依赖
└── README.md            # 本文件
```

## 常见问题

### Q: 首次运行很慢？

A: 首次运行需要下载约 2GB 模型文件。下载完成后后续运行秒启动。可手动预下载：

```bash
modelscope download --model damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch \
  --local_dir /path/to/ComfyUI/models/funasr/paraformer-zh

modelscope download --model damo/punc_ct-transformer_cn-en-common-vocab471067-large \
  --local_dir /path/to/ComfyUI/models/funasr/ct-punc

modelscope download --model damo/speech_fsmn_vad_zh-cn-16k-common-pytorch \
  --local_dir /path/to/ComfyUI/models/funasr/fsmn-vad
```

### Q: 支持英文或其他语言吗？

A: 当前模型主要针对中文优化。英文识别有一定能力但效果不如中文。如需纯英文场景，可替换为其他 FunASR 模型。

### Q: 长音频支持吗？

A: 支持。内置 VAD 模型会自动将长音频切分为语音片段分别识别，没有时长限制。

### Q: 如何提升识别准确率？

1. 使用 `hotword` 热词功能添加领域专有词汇
2. 确保音频质量良好（16kHz+ 采样率，低噪声）
3. 对于特定领域，可考虑使用 FunASR 的热词文件功能

### Q: 模型下载失败怎么办？

检查网络连接，或使用代理：

```bash
export https_proxy=http://your-proxy:port
```

也可手动从 [ModelScope](https://modelscope.cn/models/iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch) 下载模型文件放到对应目录。

## 致谢

- [FunASR](https://github.com/modelscope/FunASR) — 阿里达摩院语音识别框架
- [ModelScope](https://modelscope.cn) — 模型托管平台
- [ComfyUI](https://github.com/comfyanonymous/ComfyUI) — 节点式 Stable Diffusion UI

## 许可证

本项目代码采用 MIT 许可证。FunASR 模型的使用请遵循 [ModelScope 模型许可协议](https://modelscope.cn/models/iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch)。
