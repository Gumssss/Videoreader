from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple, Union
import soundfile as sf

from base import BaseModule
from module import FrameSelector, ImageConverter, StoryTeller, TextReader


class VideoReaderPipelineV2(BaseModule):
    """VideoReaderPipelineV2
    =======================
    A lightweight pipeline that chains FrameSelector, ImageConverter, StoryTeller,
    and TextReaderV2 to go from video → frames → captions → story → audio.

    Tracks every intermediate artifact (frames, captions, story, audio, metadata).
    """

    _FRAME_KEY = "selected_frames"
    _FRAME_DESC_KEY = "frame_desc_list"
    _VIDEO_INFO_KEY = "video_info"
    _STORY_KEY = "story_text"
    _AUDIO_KEY = "story_audio"
    _AUDIO_SAVE_KEY = "audio_save_path"

    def __init__(
        self,
        frame_selector: FrameSelector | None = None,
        image_converter: ImageConverter | None = None,
        story_teller: StoryTeller | None = None,
        text_reader: TextReader | None = None,
        name: str = "VideoReaderPipelineV1",
    ) -> None:
        super().__init__(name=name)
        self.frame_selector = frame_selector or FrameSelector()
        self.image_converter = image_converter or ImageConverter()
        self.story_teller = story_teller or StoryTeller()
        self.text_reader = text_reader or TextReader()
        self.intermediates: Dict[str, Any] = {}

    def run(
        self,
        video_path: Union[str, Path],
        n_frames: int,
        audio_save_dir: Union[str, Path] = "./audio",
        score_fn: str = "frame_difference",
        image_converter_prompt: str | None = None,
        lang_code: str = "z",
        **caption_kwargs: Any,
    ) -> Tuple[Path, Dict[str, Any]]:
        """
        运行视频阅读管道，处理视频文件并生成音频。
        参数说明
        ----------
        video_path : str | Path
            视频文件路径。
        n_frames : int
            需要选择的帧数。
        audio_save_dir : str | Path
            音频保存目录，默认为 "./audio"。
        score_fn : str
            帧选择评分函数，默认为 "frame_difference"。
        image_converter_prompt : str | None
            图像转换器的提示词，默认为 None。
        lang_code : str
            语言代码，默认为 "z"（中文）。可选值包括 'b'（英式英语）和 'z'（中文）。
        caption_kwargs : Any
            传递给图像转换器的其他参数。
        """
        # 1) select frames
        selected_frames, video_info = self.frame_selector.run(
            str(video_path), n_frames=n_frames, score_fn=score_fn
        )
        self.intermediates[self._FRAME_KEY] = selected_frames
        self.intermediates[self._VIDEO_INFO_KEY] = video_info

        # 2) get frame captions
        frame_desc_list = self.image_converter.run(
            selected_frames, prompt=image_converter_prompt,
        )
        self.intermediates[self._FRAME_DESC_KEY] = frame_desc_list

        # 3) generate story text
        story_text = self.story_teller.run("\n".join(frame_desc_list), output_language=lang_code)
        self.intermediates[self._STORY_KEY] = story_text

        # 4) synthesize audio
        audio_array = self.text_reader.run(story_text, lang_code=lang_code)
        self.intermediates[self._AUDIO_KEY] = audio_array

        # 5) save mp3
        video_name = video_info["video_name"]
        audio_save_dir = Path(audio_save_dir)
        audio_save_dir.mkdir(parents=True, exist_ok=True)
        output_path = audio_save_dir / f"{video_name}.mp3"
        self.intermediates[self._AUDIO_SAVE_KEY] = str(output_path)
        
        # assume text_reader produces numpy audio at 24000 Hz
        sf.write(str(output_path), audio_array, 24000, format="MP3")

        if not frame_desc_list:
            raise ValueError("No frames selected. Check the video file and score function.")
        if not video_info:
            raise ValueError("No video info available. Check the video file.")

        return audio_array, self.intermediates

    def reset_cache(self) -> None:
        self.intermediates.clear()

    def __getattr__(self, item: str) -> Any:
        for module in (self.frame_selector, self.image_converter,
                       self.story_teller, self.text_reader):
            if hasattr(module, item):
                return getattr(module, item)
        raise AttributeError(item)

    def summary(self, verbose: bool = False) -> str:
        lines = [f"{self.__class__.__name__} (name={self.name})",
                 f" ├─ Frame selector : {self.frame_selector.__class__.__name__}",
                 f" ├─ Image converter: {self.image_converter.__class__.__name__}",
                 f" ├─ Story teller  : {self.story_teller.__class__.__name__}",
                 f" └─ Text reader   : {self.text_reader.__class__.__name__}"]
        if verbose and self.intermediates:
            lines.append("\nCached intermediates:")
            for k, v in self.intermediates.items():
                shape = None
                if isinstance(v, list): shape = f"list[{len(v)}]"
                elif isinstance(v, dict): shape = "dict"
                lines.append(f"  • {k}: {shape or type(v)}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return self.summary(verbose=False)
