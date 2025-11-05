from flask import Flask, render_template, request, redirect, url_for, jsonify
import os
from werkzeug.utils import secure_filename
from pathlib import Path
from pipeline import ImageReaderPipeline
from pipeline import VideoReaderPipeline

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['AUDIO_FOLDER'] = 'static/generated_audio'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'mp4'}

# 确保上传目录存在
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['AUDIO_FOLDER'], exist_ok=True)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def clear_uploads_and_audio():
    upload_path = Path(app.config['UPLOAD_FOLDER'])
    audio_path = Path(app.config['AUDIO_FOLDER'])
    
    # 清空上传目录
    for file in upload_path.glob('*'):
        try:
            file.unlink()
        except Exception as e:
            print(f"Error deleting file {file}: {e}")
    
    # 清空音频目录
    for file in audio_path.glob('*'):
        try:
            file.unlink()
        except Exception as e:
            print(f"Error deleting file {file}: {e}")

@app.route('/')
def index():
    # 清理上传和音频目录
    clear_uploads_and_audio()
    # 获取语言参数，默认为英文
    lang = request.args.get('lang', 'en')
    return render_template('index.html', lang=lang)

@app.route('/detail')
def detail():
    lang = request.args.get('lang', 'en')
    return render_template('detail.html', lang=lang)

@app.route('/image', methods=['GET', 'POST'])
def image_analysis():
    lang = request.args.get('lang', 'en')
    
    if request.method == 'POST':
        # 检查是否有文件上传
        if 'file' not in request.files:
            return redirect(request.url)
        
        file = request.files['file']
        
        if file.filename == '':
            return redirect(request.url)
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # 处理图像
            pipeline = ImageReaderPipeline()
            lang_code = "z" if lang == 'zh' else "b"
            
            try:
                audio, intermediates = pipeline.run(
                    image_path=filepath,
                    image_converter_prompt="",
                    lang_code=lang_code,
                    audio_save_dir=app.config['AUDIO_FOLDER'],
                )
                
                caption = intermediates[pipeline._CAPTION_KEY]
                script_text = intermediates[pipeline._SCRIPT_TEXT_KEY]  
                mp3_path = intermediates[pipeline._AUDIO_SAVE_KEY]
                
                # 确保音频路径是相对路径
                relative_audio_path = os.path.relpath(mp3_path, start='static')
                
                return render_template('image.html', 
                                    lang=lang,
                                    image_path=filepath,
                                    caption=caption,
                                    script_text=script_text,
                                    audio_path=relative_audio_path)
            
            except Exception as e:
                print(f"Error processing image: {e}")
                return render_template('image.html', lang=lang, error=str(e))
    
    return render_template('image.html', lang=lang)

@app.route('/video', methods=['GET', 'POST'])
def video_analysis():
    lang = request.args.get('lang', 'en')
    
    if request.method == 'POST':
        # 检查是否有文件上传
        if 'file' not in request.files:
            return redirect(request.url)
        
        file = request.files['file']
        
        if file.filename == '':
            return redirect(request.url)
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # 处理视频
            pipeline = VideoReaderPipeline()
            lang_code = "z" if lang == 'zh' else "b"
            
            try:
                audio, intermediates = pipeline.run(
                    filepath,
                    n_frames=10,
                    score_fn="frame_difference",
                    image_converter_prompt="",
                    lang_code=lang_code,
                    audio_save_dir=app.config['AUDIO_FOLDER'],
                )
                
                story = intermediates[pipeline._STORY_KEY]
                frame_desc_list = intermediates[pipeline._FRAME_DESC_KEY]
                video_info = intermediates[pipeline._VIDEO_INFO_KEY]
                mp3_path = intermediates[pipeline._AUDIO_SAVE_KEY]
                
                # 确保音频路径是相对路径
                relative_audio_path = os.path.relpath(mp3_path, start='static')
                
                return render_template('video.html', 
                                    lang=lang,
                                    video_path=filepath,
                                    story=story,
                                    frame_descriptions=frame_desc_list,
                                    video_info=video_info,
                                    audio_path=relative_audio_path)
            
            except Exception as e:
                print(f"Error processing video: {e}")
                return render_template('video.html', lang=lang, error=str(e))
    
    return render_template('video.html', lang=lang)

@app.route('/switch_language')
def switch_language():
    current_lang = request.args.get('current_lang', 'en')
    new_lang = 'zh' if current_lang == 'en' else 'en'
    return redirect(url_for(request.args.get('next', 'index'), lang=new_lang))

if __name__ == '__main__':
    app.run(debug=True)