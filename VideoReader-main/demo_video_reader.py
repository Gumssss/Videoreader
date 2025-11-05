from pipeline import VideoReaderPipeline
from pathlib import Path


if __name__ == "__main__":
    # 创建 pipeline 实例
    pipeline = VideoReaderPipeline()

    # 运行 pipeline，会返回生成的 MP3 文件路径和所有中间结果
    audio, intermediates = pipeline.run(
        "static/video_test.mp4",
        n_frames=10,
        score_fn="frame_difference",
        image_converter_prompt="",
        lang_code="z",  # 使用中文
        audio_save_dir="./generation/audio",
    )

    # 从 intermediates 中提取关键信息
    story = intermediates[pipeline._STORY_KEY]
    frame_desc_list = intermediates[pipeline._FRAME_DESC_KEY]
    video_info = intermediates[pipeline._VIDEO_INFO_KEY]
    mp3_path = intermediates[pipeline._AUDIO_SAVE_KEY]

    # 打印结果
    print("Generated Story:", story)
    print("Frame Descriptions:", frame_desc_list)
    print("Video Info:", video_info)
    print("Saved Audio Path:", Path(mp3_path).resolve())

    # 也可以直接查看缓存的中间帧数
    frames = intermediates["selected_frames"]
    print("Number of Selected Frames:", len(frames))
