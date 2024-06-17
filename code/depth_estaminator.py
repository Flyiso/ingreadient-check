"""
File for test if depth estimation could help with
correcting the frames.

diff of max & min in label area = save, to not run this all the time
run this once or a few times? when?
"""
import numpy as np
import cv2
from transformers import pipeline
from PIL import Image
import matplotlib as plt
image = Image.open('/home/mx-dex/schoolstuff/Courses/kurs-10-examensarbete-ide-1-ingredient-label-reader/ingreadient-check/outputs/vid_2/blurry_frame.png')
pipe = pipeline(task="depth-estimation", model="LiheYoung/depth-anything-small-hf")
depth = pipe(image)["depth"]
depth.save('depth.png')


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
