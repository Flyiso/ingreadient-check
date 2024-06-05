"""
uses images to try to create a panorama of all.
"""
import cv2
import numpy as np
from stitching import Stitcher, stitching_error


class ManagePanorama:
    """
    class responsible for building panorama image of
    output frames.
    """
    def __init__(self,
                 frame_manager) -> None:
        """
        parameters:
            interval(int): distance in frames between every merge default: 10
        """
        self.frames = []
        self.to_stitch = []
        self.selected = []
        self.base = False
        self.panorama_merged = False
        self.merge_counter = 0
        self.fail_counter = 0
        self.frame_manager = frame_manager

    def add_frame(self, frame, last_frame: bool = False) -> bool:
        """
        Adds a new frame to the mergers frames.
        Tries self.interval frames before if unsuccessful.
        Returns True False when first merge is successfull
        Returns True True when also second
        merge(of already merged frames) succeeds.
        returns False False else.
        Last return value tells if any merge was attempted.
        Parameters:
            frame(np.ndarray):  frame to attempt to merge in to panorama.
            last_frame(bool):   Defaults to False. When true, stitcher tries to
                                do a final merge with the new image even if
                                minimum number of frames to
                                do a merge is not reached.
        """
        self.frames.append(frame)
        if last_frame:
            if self.stitch_frames():
                cv2.imwrite('progress_images/base.png', self.base)
                return True
        return False

    def stitch_frames(self):
        """
        Attempts to stitch together already stitched frames.
        """
        self.to_stitch = self.frame_manager.get_most_different(
            self.frames, len(self.frames)//23)
        print('frames collected...')

        stitcher = Stitcher(detector='sift', finder='dp_color',
                            blender_type='multiband',
                            blend_strength=5, matcher_type='affine',
                            wave_correct_kind='no',
                            compensator='gain_blocks',
                            nr_feeds=1, block_size=5,
                            warper_type='cylindrical',
                            try_use_gpu=True, match_conf=0.45)
        try:
            panorama = stitcher.stitch(self.to_stitch)
        except stitching_error.StitchingError:
            print('stitching unsuccessful...')
            panorama = False
        if isinstance(panorama, np.ndarray):
            panorama = cv2.addWeighted(panorama, 0.5, panorama, 0.5, 0)
            cv2.imwrite('progress_images/FinalPanorama.png', panorama)
            self.base = panorama
            return True
        return False

    def detect_text(self) -> np.ndarray | bool:
        if isinstance(self.base, np.ndarray):
            self.frame_manager.find_text(self.base)
            return self.base
        return False
