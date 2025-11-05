from typing import List, Union
import numpy as np
from PIL import Image
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration

from base import BaseModule


class ImageConverterV1(BaseModule):
    """
    ImageConverterV1
    ----------------
    将 **单张图片** 或 **图片列表** 转换为文字描述（caption）。

    使用预训练的 BLIP 图像字幕模型  
    （默认 "Salesforce/blip-image-captioning-base"）。
    """

    _DEFAULT_MODEL = "Salesforce/blip-image-captioning-base"
    _DEFAULT_PROMPT = "a photography of"

    def __init__(self,
                 model_name: str = _DEFAULT_MODEL,
                 device: Union[str, torch.device, None] = None,
                 prompt: str = _DEFAULT_PROMPT):
        """
        参数说明
        -------
        model_name : str  
            HuggingFace Hub 上的模型名称或路径。  
        device : str | torch.device | None  
            计算设备；若为 None 则自动检测 GPU，否则退回 CPU。  
        prompt : str  
            放在图片前的引导词，可为空字符串。  
        """
        super().__init__(name="ImageConverter")

        self.prompt = prompt
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # 载入 tokenizer + 模型
        self.processor = BlipProcessor.from_pretrained(model_name)
        self.model = (
            BlipForConditionalGeneration.from_pretrained(model_name)
            .to(self.device)
            .eval()
        )

    # ------------------------------------------------------------------ #
    # 核心接口
    # ------------------------------------------------------------------ #
    def run(self,
            images: Union[
                Image.Image,
                np.ndarray,
                List[Union[Image.Image, np.ndarray]]
            ],
            prompt: str = None) -> Union[str, List[str]]:
        """
        根据输入类型自动调用 `image_to_text()` 或 `frames_to_texts()`。

        参数
        ----
        images : Image | np.ndarray | list  
            - 单张图片：返回单条字幕字符串。  
            - 图片列表：返回等长字符串列表。  
        prompt, max_length, num_beams : 同下述两个函数。  

        返回
        ----
        str 或 list[str]  
            与输入结构对应的字幕结果。  
        """
        prompt = prompt if prompt is not None else self.prompt

        # 判断是单张还是多张
        if isinstance(images, (Image.Image, np.ndarray)):
            # 单张
            return self.image_to_text(images, prompt)
        elif isinstance(images, list):
            # 多张
            return self.frames_to_texts(images, prompt)
        else:
            raise TypeError("images 必须是 PIL.Image、np.ndarray 或它们的列表")

    # ------------------------------------------------------------------ #
    # 单张图片 → 文本
    # ------------------------------------------------------------------ #
    def image_to_text(self,
                      image: Union[Image.Image, np.ndarray],
                      prompt: str = "") -> str:
        """
        将单张图片转为字幕文本。

        返回
        ----
        str
        """
        prompt = prompt if prompt is not None else self.prompt
        return self._img_to_text_single(image, prompt)

    # ------------------------------------------------------------------ #
    # 多帧图片 → 文本列表
    # ------------------------------------------------------------------ #
    def frames_to_texts(self,
                        images: List[Union[Image.Image, np.ndarray]],
                        prompt: str = "") -> List[str]:
        """
        批量生成字幕，保持输入顺序。

        返回
        ----
        list[str]
        """
        assert images and isinstance(images, list), "images 需为非空列表"
        prompt = prompt if prompt is not None else self.prompt
        return [
            self._img_to_text_single(img, prompt)
            for img in images
        ]

    # ------------------------------------------------------------------ #
    # 内部工具函数
    # ------------------------------------------------------------------ #
    def _img_to_text_single(self,
                        img: Union[Image.Image, np.ndarray],
                        prompt: str) -> str:
        """
        对单张图片进行字幕生成；内部私有。
        """
        pil_img = self._ensure_pil(img)
        inputs = self.processor(pil_img, prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
            )
        caption = self.processor.decode(generated_ids[0], skip_special_tokens=True)
        return caption.strip()

    @staticmethod
    def _ensure_pil(img: Union[Image.Image, np.ndarray]) -> Image.Image:
        """
        保证返回 PIL.Image(RGB)。接受：
        * PIL.Image
        * np.ndarray (BGR / RGB)
        """
        if isinstance(img, Image.Image):
            return img.convert("RGB")

        if isinstance(img, np.ndarray):
            # 若为 (H, W, 3) 形状的数组
            if img.ndim == 3 and img.shape[-1] == 3:
                # 简单判断是否 BGR；若是则转换为 RGB
                if ImageConverterV1._looks_bgr(img):
                    img = img[:, :, ::-1]
            return Image.fromarray(img.astype(np.uint8), mode="RGB")

        raise TypeError(f"不支持的图片类型：{type(img)}")

    @staticmethod
    def _looks_bgr(arr: np.ndarray, thresh: float = 1.05) -> bool:
        """
        简易判断 numpy array 是否为 BGR：
        若第 0 通道（蓝）均值明显高于第 2 通道（红）则认为是 BGR。
        """
        if arr.ndim != 3 or arr.shape[-1] != 3:
            return False
        ch0_mean = arr[..., 0].mean()
        ch2_mean = arr[..., 2].mean()
        return ch0_mean > ch2_mean * thresh
