"""
Manages and extracts data from
merged images.
"""
import pytesseract as pt
import numpy as np
import cv2
from skimage.segmentation import slic
from skimage.data import astronaut
from skimage.color import label2rgb


class ManageFrames:
    """
    Does the requested operations on frame.
    """
    def __init__(self) -> None:
        """
        Initializes the fame manager
        """
        pass

    def set_manager_values(self, frame):
        """
        set values for frame manager threshold
        based on where most frame is found in
        input frame.
        """
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_data = self.find_text(frame)
        self.set_threshold_values(frame, frame_data)

    def find_text(self, frame,
                  config: str = '--oem 3 --psm 6') -> dict:
        """
        Returns data dictionary of the text
        found in the frame.
        """
        data = pt.image_to_data(frame, config=config,
                                output_type='dict')
        return data

    def set_threshold_values(self, frame, frame_data: dict):
        """
        adjust threshold to the most text-populated area.
        """
        img_x = int(np.mean(frame_data['top']))
        img_y = int(np.mean(frame_data['left']))
        img_w = int(np.mean(frame_data['width']))
        img_h = int(np.mean(frame_data['height']))
        frame = frame[img_x:img_x+img_w,
                      img_y:img_y+img_h]
        frame = cv2.GaussianBlur(frame, (7, 7), 0)

    def extract_foreground(self, frame):
        frame_g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(frame_g, 0, 255,
                                    cv2.THRESH_BINARY_INV +
                                    cv2.THRESH_OTSU)
        kernel = np.ones((3, 3), np.uint8)
        closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE,
                                   kernel, iterations=2)

        # Background area using Dilation 
        bg = cv2.dilate(closing, kernel, iterations=1)

        # Finding foreground area 
        dist_transform = cv2.distanceTransform(closing, cv2.DIST_L2, 3)
        ret, fg = cv2.threshold(dist_transform, 0.02
                                * dist_transform.max(), 255, 0)
        return fg

    def segment_img_scikit(self, frame):
        frame_seg = slic(frame,
                         n_segments=2,
                         compactness=10)
        return frame_seg
