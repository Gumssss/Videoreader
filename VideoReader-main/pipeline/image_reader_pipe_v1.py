from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple, Union
import soundfile as sf

from base import BaseModule  # Assuming same base as in original code
from module import ImageConverter, StoryTeller, TextReader  # Import required modules only

ImagePathLike = Union[str, Path]


class ImageReaderPipelineV1(BaseModule):
    _IMAGE_PATH_KEY = "image_path"
    _CAPTION_KEY = "caption"
    _SCRIPT_TEXT_KEY = "script_text"
    _AUDIO_KEY = "audio_array"
    _AUDIO_SAVE_KEY = "audio_save_path"

    def __init__(
        self,
        image_converter: ImageConverter | None = None,
        story_teller: StoryTeller | None = None,
        text_reader: TextReader | None = None,
        name: str = "ImageReaderPipelineV1",
    ) -> None:
        super().__init__(name=name)
        self.image_converter = image_converter or ImageConverter()
        self.story_teller = story_teller or StoryTeller()
        self.text_reader = text_reader or TextReader()
        self.intermediates: Dict[str, Any] = {}

    def run(
        self,
        image_path: Union[str, Path],
        audio_save_dir: Union[str, Path] = "./generation/audio",
        image_converter_prompt: str | None = None,
        lang_code: str = "z",
        **caption_kwargs: Any,
    ) -> Tuple[Any, Dict[str, Any]]:
        """运行图像阅读管道：image → caption → audio。

        参数（对齐 VideoReaderPipelineV1，去掉视频特有的参数）：
        ------------------------------------------------------------------
        image_path : str | Path
            单张图片路径。
        audio_save_dir : str | Path
            音频保存目录，默认为 "./audio"。
        image_converter_prompt : str | None
            传给 ImageConverter 的提示词。
        lang_code : str
            语言代码（与 VideoReaderPipelineV1 含义一致）。
        caption_kwargs : Any
            传递给 ImageConverter.run 的其它参数（如需要）。

        返回
        -----
        (audio_array, intermediates)
        audio_array : numpy.ndarray
            TTS 生成的音频数组（假设 24000 Hz）。
        intermediates : Dict[str, Any]
            缓存的中间结果：caption、script_text、audio_save_path 等。
        """
        # 1) 记录图片路径
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image file not found: {image_path}")
        self.intermediates[self._IMAGE_PATH_KEY] = str(image_path)

        # 2) 生成 caption（ImageConverter 接口期望列表，保持兼容）
        caption_list = self.image_converter.run([str(image_path)], prompt=image_converter_prompt, **caption_kwargs)
        if not isinstance(caption_list, list) or len(caption_list) == 0:
            raise ValueError("ImageConverter 未返回有效的 caption 列表。")
        caption = caption_list[0]
        self.intermediates[self._CAPTION_KEY] = caption

        # 3) 构造脚本文本（这里直接使用 caption，可按需扩展）
        script_text= self.story_teller.run(caption)
        self.intermediates[self._SCRIPT_TEXT_KEY] = script_text

        # 4) 文本转语音
        audio_array = self.text_reader.run(script_text, lang_code=lang_code)
        self.intermediates[self._AUDIO_KEY] = audio_array

        # 5) 保存音频
        audio_save_dir = Path(audio_save_dir)
        audio_save_dir.mkdir(parents=True, exist_ok=True)
        audio_name = f"{image_path.stem}.mp3"
        output_path = audio_save_dir / audio_name
        sf.write(str(output_path), audio_array, 24000, format="MP3")
        self.intermediates[self._AUDIO_SAVE_KEY] = str(output_path)

        return audio_array, self.intermediates

    def reset_cache(self) -> None:
        self.intermediates.clear()

    def __getattr__(self, item: str) -> Any:
        for module in (self.image_converter, self.text_reader):
            if hasattr(module, item):
                return getattr(module, item)
        raise AttributeError(item)

    def summary(self, verbose: bool = False) -> str:
        lines = [
            f"{self.__class__.__name__} (name={self.name})",
            f" ├─ Image converter: {self.image_converter.__class__.__name__}",
            f" └─ Text reader   : {self.text_reader.__class__.__name__}",
        ]
        if verbose and self.intermediates:
            lines.append("Cached intermediates:")
            for k, v in self.intermediates.items():
                shape = None
                if isinstance(v, list):
                    shape = f"list[{len(v)}]"
                elif isinstance(v, dict):
                    shape = "dict"
                lines.append(f"  • {k}: {shape or type(v)}")
        return "".join(lines)

    def __repr__(self) -> str:
        return self.summary(verbose=False)
