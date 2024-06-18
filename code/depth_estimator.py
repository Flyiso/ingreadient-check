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


class DepthCorrection:
    """
    Class that finds depth and
    correct the image's perspective/flatten it.
    init sets min/max diff
    method just correct by values?
    """
    def __init__(self, frame: np.ndarray) -> None:
        frame_org = frame
        frame_cv2 = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)
        alpha_channel = np.where(
            (frame[:, :, 0] == 0) &
            (frame[:, :, 1] == 0) &
            (frame[:, :, 2] == 0),
            0,
            255
        )
        frame_cv2[:, :, 3] = alpha_channel
        cv2.imwrite('masked.png', frame_cv2)
        frame = Image.fromarray(frame_cv2)
        pipe = pipeline(task="depth-estimation",
                        model="LiheYoung/depth-anything-large-hf")
        depth = np.array(pipe(frame)["depth"])
        depth = cv2.cvtColor(depth, cv2.COLOR_BGR2BGRA)
        depth[:, :, 3] = alpha_channel

        cv2.imwrite('depth.png', depth)

        _, _, _, a = cv2.split(depth)
        mask = a > 0
        gray = cv2.cvtColor(depth[:, :, :3],
                            cv2.COLOR_BGRA2GRAY)
        depth = np.where(mask, gray, 255)
        depth = cv2.bitwise_not(depth)
        min_val = np.min(depth[mask])
        max_val = np.max(depth[mask])
        print(min_val, max_val)
        print(np.mean(depth[mask]))
        print(np.median(depth[mask]))
        print('....')

        cv2.imwrite('depth_gs.png', depth)
        self.frame = False

    def _estimate_depth(self, frame, mask):
        pass

    def correct_image(self, frame: np.ndarray,
                      mask: np.ndarray) -> np.ndarray:
        """
        use estimated size to correct
        the images shape and perspective
        """
        pass
