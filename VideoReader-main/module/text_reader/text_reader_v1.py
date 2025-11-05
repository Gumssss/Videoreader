from typing import Union
import numpy as np
import torch
from kokoro import KPipeline

from base import BaseModule


class TextReaderV1(BaseModule):
    """
    TextReaderV1
    ------------
    将 **文本** 转换为语音（audio waves）。

    使用开源 TTS 模型 Kokoro 进行语音合成。
    """

    VOICE_DICT = {
        'b': 'bf_emma',  # British English
        'z': 'zf_xiaoxiao',   # Chinese
    }

    def __init__(
        self,
        device: Union[str, torch.device, None] = None
    ):
        """
        参数说明
        -------
        lang_code : str
            语言代码，b 表示英式英语，z 表示中文。
        speed : float
            合成语速倍速，默认为 1.0。
        device : str | torch.device | None
            计算设备；若为 None 则自动检测 GPU，否则退回 CPU。
        """
        super().__init__(name="TextReader")
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    def run(
        self,
        texts: Union[str, list],
        lang_code: str,
        speed: float = 1.0
    ) -> np.ndarray:
        """
        接收单条文本或文本列表，若为列表则拼接为单条文本后合成。

        可在此处指定 lang_code 和 speed（优先于初始化参数）。
        参数说明
        ----------
        texts : str | list
            单条文本或文本列表，若为列表则拼接为单条文本后合成。
        lang_code : str
            语言代码，b 表示英式英语，z 表示中文。
        speed : float
            合成语速倍速，默认为 1.0。
        返回
        ----
        numpy.ndarray
            合成后的音频波形，采样率 24000 Hz。
        """
        # 更新语言和速度
        assert lang_code in self.VOICE_DICT, f"Unsupported lang_code: {lang_code}, available options: {list(self.VOICE_DICT.keys())}"
        
        # 通过 lang_code 获取 voice
        voice = self.VOICE_DICT.get(lang_code)

        # 拼接列表为单个字符串
        if isinstance(texts, list):
            if not texts:
                raise ValueError("texts 列表不能为空")
            text = " ".join(texts)
        elif isinstance(texts, str):
            text = texts
        else:
            raise TypeError("texts must be a string or a list of strings")

        # 重新配置 pipeline 以支持动态 lang_code
        self.pipeline = KPipeline(lang_code=lang_code, device=self.device)

        return self.text_to_audio(text, voice, speed)

    def text_to_audio(
        self,
        text: str,
        voice: str,
        speed: float
    ) -> np.ndarray:
        """
        将单条文本合成为音频波形。

        返回
        ----
        numpy.ndarray
            合成后的音频波形，采样率 24000 Hz。
        """
        generator = self.pipeline(text, 
                                  voice=voice, 
                                  speed=speed)
        # 生成器可能产出多个候选，取第一个
        _, _, audio = next(generator)
        return audio
