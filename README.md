# Multi-Agent Image & Video Reader<br>多智能体视频与图像理解系统

## A full-stack multimodal system that converts images and videos into structured captions, coherent stories, and natural speech audio.<br><br>一个将图像与视频转换为字幕、完整叙事文本与自然语音的端到端多模态系统。


## Usage

### Setup environment

```bash
pip install -r requirements.txt
git clone https://github.com/Gumssss/Videoreader
cd Videoreader
```

### Run the code

```bash
python app.py
```
---

## Convert image to audio

You can upload an image in two ways:

1. Upload a local image file  
2. Capture a photo using your computer’s camera (if supported by browser)

The system will:

- Generate a caption (BLIP)  
- Produce a coherent story (Qwen 3-4B)  
- Convert the story into audio (Kokoro TTS)

---

## Convert video to audio

The video reader will:

1. Extract informative frames from your video  
2. Generate captions for each selected frame  
3. Produce a story based on all captions  
4. Convert the story into speech  

Upload a `.mp4` file and wait for the results.

---

## Project Structure

```
project/
├── app.py
├── pipeline/
├── module/
├── templates/
└── static/
```

---

## Future works

- Add more language options  
- Support more TTS voices  
- Add GPU acceleration detection  
- Add batch processing mode  








































