**Multi-Agent Video & Image Understanding System**
**多智能体视频与图像理解系统**

A full-stack multimodal system that converts images and videos into structured captions, coherent stories, and natural speech audio.
一个将图像与视频转换为字幕、完整叙事文本与自然语音的端到端多模态系统。




Overview | 项目概述

This project is a multi-agent multimodal intelligence system that processes both images and videos.
It generates:

Captions (BLIP Image Captioning)

Coherent stories (Qwen 3-4B LLM)

Natural speech audio (Kokoro TTS)

Frame selection for videos (difference-based scoring)


本项目是一个 面向图像与视频的多智能体多模态系统，可自动完成：

图像字幕生成（BLIP）

剧情扩写 / 连贯故事生成（Qwen 3-4B）

自然语音合成（Kokoro TTS）

视频关键帧选取（差分评分算法）

并提供 完整的 Web 前端界面（Flask），支持文件上传、语言切换（中英）、音频下载等功能。

✨ Features | 功能特色
🔹 Image Processing | 图像处理

Upload image

BLIP captioning

Story generation via Qwen

TTS output (EN/CH)

Web UI preview

🔹 Video Processing | 视频处理

Upload video

Automatic frame extraction

Caption each selected frame

Generate full story based on all frames

Output narration audio

🔹 Web Application | 网页应用

Upload & preview

English / Chinese UI

Auto-clean temp files

Download MP3

🔹 Modular Multi-Agent Pipeline | 模块化多智能体流水线
Agents:

ImageConverter — BLIP caption generator

FrameSelector — extract top-N frames

StoryTeller — LLM narrative builder

TextReader — TTS pipeline

ImageReaderPipeline & VideoReaderPipeline orchestrate everything

🧱 System Architecture | 系统架构
User Upload → Flask Web UI

核心流程：
视觉 → 文本 → 故事 → 语音

🛠 Tech Stack | 技术栈
Component	Library / Model
Captioning	Salesforce BLIP
Story Generation	Qwen 3-4B
TTS	Kokoro TTS
Web	Flask + Jinja2
Video Processing	OpenCV
Image Processing	Pillow
Backend	Python 3.8+
📁 Project Structure | 项目结构
project/
│
├── app.py                   # Flask web application
│
├── pipeline/
│   ├── ImageReaderPipeline
│   └── VideoReaderPipeline
│
├── module/
│   ├── frame_selector/
│   ├── image_converter/
│   ├── story_teller/
│   └── text_reader/
│
├── utils/
│   ├── misc.py
│   └── score_fn.py
│
├── templates/               # HTML templates
└── static/
    ├── uploads/
    └── generated_audio/

🔧 Installation | 安装
git clone <repo-url>
cd <project-folder>

pip install -r requirements.txt


Ensure PyTorch (with CUDA) is installed for Qwen & BLIP acceleration.

▶️ Run the Web App | 运行 Web 应用
python app.py


Then open:

http://localhost:5000

🔄 Pipelines | 流水线设计
1. ImageReaderPipeline

Image → Caption → Story → Audio

Steps:

Load image

Run BLIP to generate caption

Use Qwen to expand caption into narrative

Convert story to TTS audio

2. VideoReaderPipeline

Video → Frames → Captions → Story → Audio

Steps:

Extract frames (FrameSelector)

Caption each frame (ImageConverter)

Merge captions into a logical story (StoryTeller)

Generate narration audio (TextReader)

📦 Module Details | 模块说明
🟥 FrameSelector

Selects top-N frames using frame_difference_score_fn

Computes difference between consecutive frames

🟦 ImageConverter

BLIP model

Supports single image, list, or file paths

Auto converts ndarray → PIL

🟩 StoryTeller

Uses Qwen/Qwen3-4B

Large custom prompting framework

Supports Chinese/English output

🟨 TextReader

Kokoro TTS wrapper

EN voice: bf_emma

CN voice: zf_xiaoxiao

Returns 24kHz waveform

📘 Examples | 使用示例
Image Pipeline
from pipeline import ImageReaderPipeline

pipeline = ImageReaderPipeline()
audio, intermediates = pipeline.run(
    image_path="example.jpg",
    lang_code="b",  # b=English, z=Chinese
    audio_save_dir="./audio"
)

print(intermediates["caption"])
print(intermediates["script_text"])

Video Pipeline
from pipeline import VideoReaderPipeline

pipeline = VideoReaderPipeline()
audio, intermediates = pipeline.run(
    "video.mp4",
    n_frames=10,
    score_fn="frame_difference",
    lang_code="z",
    audio_save_dir="./audio"
)

print(intermediates["story"])

📜 License | 许可证

MIT License (recommended).
MIT 许可证（建议使用）。

🙌 Acknowledgements | 致谢

Models used:

Salesforce BLIP

Qwen/Qwen3-4B

Kokoro TTS
