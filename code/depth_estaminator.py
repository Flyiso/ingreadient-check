"""
File for test if depth estimation could help with
correcting the frames.

diff of max & min in label area = save, to not run this all the time
run this once or a few times? when?
"""
import numpy as np


class DepthCorrection:
    """
    Class that finds depth and
    correct the image's perspective/flatten it.
    init sets min/max diff,
    method just correct by values?
    """
    def __init__(self, frame: np.ndarray, roi_mask: np.ndarray) -> None:
        pass

    def _estimate_depth(self, frame, mask):
        pass

    def correct_image(self, frame: np.ndarray,
                      mask: np.ndarray) -> np.ndarray:
        """
        use estimated size to correct
        the images shape and perspective
        """
        pass
