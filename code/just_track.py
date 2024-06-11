"""
This is to test the efficiency and performance of object tracking
used together with image segmentation.
In the purpose to to find alternative/better/faster
approaches.
"""
from image_manager import ManageFrames


class TrackVideo:
    def __init__(self) -> None:
        self.frame_manager = ManageFrames()
        # loop frames
        # get first possible contour
        # track that
        # Estimate changes/movements inside contour, add after threshold
        # merge the images.
        # compare result + performance.
        # (evaluate result when done- make sure panorama covers all of the label)
        pass
