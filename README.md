# 👨‍💻 智能内容生成 Agent

这是一个功能强大的内容处理 Agent，旨在将视频、音频或文本文档自动转换为结构化、高质量的 Markdown 笔记。

本应用集成了一个 Streamlit 交互界面，允许用户上传文件，选择不同的处理引擎（本地 Whisper 或 Qwen ASR），并根据内容类型（理工科STEM、 人文社科HASS或医学Medical）调用大语言模型（LLM）生成深度定制的笔记。

## 核心功能

* **多模态输入**: 支持多种文件格式：
    * **视频**: `.mp4`, `.mov`, `.webm` 等 (由 FFmpeg 支持)。
    * **音频**: `.mp3`, `.m4a`, `.wav`, `.mpga` 等。
    * **文档**: `.txt`, `.md`, `.pdf` 等。
* **双 ASR 引擎**: 用户可根据需求灵活选择语音转录服务：
    * **Local Whisper**: 在本地设备上运行，保障数据隐私，支持多种模型大小 (tiny, base, small ...)。
    * **Qwen API**: 调用阿里云 DashScope 提供的 `qwen3-asr-flash` 模型，速度快、精度高。
* **ASR 上下文增强**: 在使用 Qwen API 时，可提供“热词”（如专业术语、人名）来显著提升特定词汇的识别准确率。
* **智能笔记生成**: 利用配置的大语言模型（LLM）将转录稿或文本文档处理成笔记：
* **笔记精炼 (Refine)**: 支持对已生成的笔记进行迭代修改。用户可以提出“更简洁”、“更详细”或自定义指令，Agent 将结合原始转录稿重新生成笔记。
* **实时流式输出**: 在生成和精炼笔记时，支持流式（Streaming）输出，内容逐字显示，提供即时反馈。
* **配置验证**: 提供 `api_verifier.py` 脚本，用于在运行前检查 API 密钥和网络连通性。

## 技术栈

* **前端框架**: Streamlit
* **媒体处理**: FFmpeg (通过 `subprocess` 调用)
* **核心依赖**: Python 3.x
* **ASR (语音转录)**: `openai-whisper`, `dashscope`
* **LLM API**: `requests` (兼容 OpenAI 格式的 API), `dashscope`
* **配置管理**: `python-dotenv`

## 安装步骤

### 1、安装python

在python官网下载即可，建议版本>=3.10

### 2、运行run.bat文件

运行该文件后，所有安装过程一步完成

首次使用streamlit会提示输入邮箱，随便输一个即可，之后会跳转到网页版应用界面