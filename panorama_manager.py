"""
uses images to try to create a panorama of all.
"""
import cv2


class ManagePanorama:
    """
    class responsible for building panorama image of
    output frames.
    """
    def __init__(self,
                 interval: int = 10,
                 merge_size: int = 4,) -> None:
        """
        parameters:
            interval(int): distance in frames between every merge default: 10 
            merge_size(int): How many frames to merge to panorama img.
        """
        self.interval = interval
        self.merge_size = merge_size
        self.image_management = None
        self.frames = []

    def add_frame(self, frame):
        """
        Adds a new frame to the mergers frames.
        """
        self.frames.append(frame)

    def process_frames(self, frames: list) -> list:
        """
        Use image manipulation methods
        to use the frames in panorama merge.
        """
        pass
