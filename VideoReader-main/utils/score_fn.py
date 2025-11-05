import cv2
import numpy as np
from typing import Dict, List


def frame_difference_score_fn(
    frames: list,
    group_size: int = 1,
    resize_shape: tuple = (64, 64),
    sort: bool = True,
    normalised: bool = False
) -> List[Dict[int, float]]:
    """
    计算帧间差异分数。

    Parameters
    ----------
    frames : list[np.ndarray]
        图像帧列表（BGR/RGB 均可）。
    group_size : int, default=1
        - 1 : 每帧与前一帧做差。
        - >1: 将序列按 group_size 划分小组，
              组内所有帧都与该组第 1 帧做差。
    resize_shape : tuple(int, int), default=(64, 64)
        先缩放再做差，可加速计算。
    sort : bool, default=True
        True 则按分数降序排序。
    normalised : bool, default=True
        True 则把分数归一化到 [0, 1]。
    
    Returns
    -------
    list[dict]  格式：[{"index": 0, "score": 0.0}, ...]
    """

    if not frames:
        return []

    # ------- 内部工具函数 -------
    def _normalise(scores):
        vals = [d["score"] for d in scores]
        v_min, v_max = min(vals), max(vals)
        if v_max == v_min:           # 全部相同
            return [{"index": d["index"], "score": 0.0} for d in scores]
        return [
            {"index": d["index"], "score": (d["score"] - v_min) / (v_max - v_min)}
            for d in scores
        ]

    def _sort(scores):
        return sorted(scores, key=lambda d: d["score"], reverse=True)

    # ------- 计算原始分数 -------
    scores = []
    if group_size <= 1:
        for idx, frame in enumerate(frames):
            if idx == 0:
                score = 255.0  # 首帧设定一个常数
            else:
                f_prev = cv2.resize(frames[idx - 1], resize_shape)
                f_cur = cv2.resize(frame, resize_shape)
                score = np.abs(f_prev - f_cur).mean()
            scores.append({"index": idx, "score": score})
    else:
        n = len(frames)
        for g_start in range(0, n, group_size):
            idx_group = range(g_start, min(g_start + group_size, n))
            ref = cv2.resize(frames[g_start], resize_shape)
            for idx in idx_group:
                if idx == g_start:
                    score = 0.0               # 组内第一帧
                else:
                    f_cur = cv2.resize(frames[idx], resize_shape)
                    score = np.abs(ref - f_cur).mean()
                scores.append({"index": idx, "score": score})

    # ------- 后处理 -------
    if sort:
        scores = _sort(scores)
    if normalised:
        scores = _normalise(scores)

    return scores
