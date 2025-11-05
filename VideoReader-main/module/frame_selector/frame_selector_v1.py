import cv2
import os

from base import BaseModule
from utils.score_fn import frame_difference_score_fn
from utils.misc import get_file_name


class FrameSelectorV1(BaseModule):
    """
    FrameSelectorV1 is a module that selects frames based on certain criteria.
    It inherits from BaseModule.
    """

    def __init__(self):
        """
        Initialize the FrameSelectorV1 module.
        """
        super().__init__(name="FrameSelector")
        self.support_score_fns = ["frame_difference"]
        self.video_info = {}

    def get_frames(self, video_path: str, n_frames: int, 
                   score_fn: str = "frame_difference"):
        # check input validation
        assert os.path.exists(video_path), f"Video path {video_path} does not exist."
        assert score_fn in self.support_score_fns, f"Score function {score_fn} is not supported. " \
                                                   f"Supported functions: {self.support_score_fns}"
        assert n_frames > 0, "Number of frames to select must be greater than 0."
        
        # get frames and video info
        frame_list, video_info = self._read_videos(video_path)

        # score frames
        frame_scores = self._score_frames(frame_list, score_fn)

        # select top n frames
        frame_scores = frame_scores[:n_frames]
        
        # get the frame indices and sorted them ascendingly
        frame_indices = sorted([d["index"] for d in frame_scores])

        # get the selected frames
        selected_frames = [frame_list[i] for i in frame_indices]
        return selected_frames, video_info

    def _score_frames(self, frame_list, score_fn):
        if score_fn == "frame_difference":
            frame_scores = frame_difference_score_fn(frame_list, sort=True, normalised=False)
        else:
            raise ValueError(f"Score function {score_fn} is not implemented.")
        
        return frame_scores

    def _read_videos(self, video_path:str):
        video_cap = cv2.VideoCapture(video_path)
        frame_list = []
        video_info = {}

        video_info["fps"] = video_cap.get(cv2.CAP_PROP_FPS)
        video_info["frame_shape"] = (video_cap.get(cv2.CAP_PROP_FRAME_WIDTH),
                                     video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # (width, height)
        video_info["video_name"] = get_file_name(video_path)

        # get frames
        while True:
            ret, frame = video_cap.read()
            if not ret:
                break
            frame_list.append(frame)
        return frame_list, video_info


    def run(self, video_path: str, n_frames: int, 
            score_fn: str = "frame_difference"):
        """
        Run the frame selection process.
        """
        return self.get_frames(video_path, n_frames, score_fn)
        