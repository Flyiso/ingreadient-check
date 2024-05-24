"""
uses images to try to create a panorama of all.
"""
import cv2
import numpy as np


class ManagePanorama:
    """
    class responsible for building panorama image of
    output frames.
    """
    def __init__(self,
                 frame_manager,
                 interval: int = 10) -> None:
        """
        parameters:
            interval(int): distance in frames between every merge default: 10
        """
        self.interval = interval
        self.frames = []
        self.to_stitch = []
        self.stitched = []
        self.base = False
        self.merge_counter = 1
        self.fail_counter = 0
        self.stitcher = cv2.Stitcher.create(1)
        self.stitcher.setWaveCorrection(cv2.WARP_POLAR_LINEAR)
        self.stitcher.setCompositingResol(-1)
        self.stitcher.setInterpolationFlags(cv2.INTER_LANCZOS4)
        self.stitcher.setPanoConfidenceThresh(0.7)
        self.stitcher.setRegistrationResol(-1)
        self.stitcher.setSeamEstimationResol(4)  # fails/interval: 0
        self.frame_manager = frame_manager

    def add_frame(self, frame, last_frame: bool = False) -> bool:
        """
        Adds a new frame to the mergers frames.
        Tries self.interval frames before if unsuccessful.
        Returns True False when first merge is successfull
        Returns True True when also second merge(of already merged frames) succeeds.
        returns False False else.
        Parameters:
            frame(np.ndarray):  frame to attempt to merge in to panorama.
            last_frame(bool):   Defaults to False. When true, stitcher tries to
                                do a final merge with the new image even if
                                minimum number of frames to do a merge is not reached. 
        """
        self.frames.append(frame)
        if not len(self.frames) % self.interval == 0 and self.base is True:
            return False, False
        bool_a = self.add_more_frames(frame, last_frame)
        bool_b = self.stitch_stitched_frames(last_frame)
        return bool_a, bool_b

    def add_more_frames(self, frame: np.ndarray,
                        last_frame: bool) -> bool:
        """
        Modified version of frame stitching to test if
        more frames solve stitcher error 1 problems
        """
        if self.base is False:
            self.base = self.frame_manager.find_label(frame)
            print('New Merge: First frame')
            self.to_stitch.append(self.base)
            self.frame_manager.set_manager_values(self.base)
            return True
        status = None
        for frame in self.frames[::-1]:
            frame = self.frame_manager.find_label(frame)
            if frame is False:
                continue
            self.to_stitch.append(frame)
            if len(self.to_stitch) >= 3 or last_frame:
                status, result = \
                    self.stitcher.stitch(self.frame_manager.cut_images(
                        self.to_stitch),
                                         self.frame_manager.get_masks(
                                             self.to_stitch))
                if status == cv2.Stitcher_OK:
                    print('New Merge: Success')
                    cv2.imwrite('progress_images/merged.png', result)
                    self.base = result
                    self.stitched.append(result)
                    self.to_stitch = []
                    return True
                print('New Merge: Failed')
            return False

    def stitch_stitched_frames(self, last_frame: bool):
        """
        Attempts to stitch together already stitched frames.ii
        """
        if len(self.stitched) >= 3 or last_frame:
            status, result = self.stitcher.stitch(self.stitched)
            if status == cv2.Stitcher_OK:
                print('Stitching of stitched images Success')
                cv2.imwrite('progress_images/merged_2.png', result)
                return True
        return False

    def detect_text(self):
        data = self.frame_manager.find_text(self.base)
        print(
            ' '.join(data['text']))
        return self.base
