from pipeline import ImageReaderPipeline
from pathlib import Path


if __name__ == "__main__":
    # 创建 pipeline 实例
    pipeline = ImageReaderPipeline()

    # 运行 pipeline，会返回生成的 MP3 文件路径和所有中间结果
    audio, intermediates = pipeline.run(
        image_path="static/image_test1.jpg",
        image_converter_prompt="",
        lang_code="z",  # 使用中文
        audio_save_dir="./generation/audio",
    )

    # 从 intermediates 中提取关键信息
    caption = intermediates[pipeline._CAPTION_KEY]
    script_text = intermediates[pipeline._SCRIPT_TEXT_KEY]
    mp3_path = intermediates[pipeline._AUDIO_SAVE_KEY]

    # 打印结果
    print("Image Caption:", caption)
    print("Script Text:", script_text)
    print("Saved Audio Path:", Path(mp3_path).resolve())

