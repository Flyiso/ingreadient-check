"""
This is temporary file to test/find the best way to run the
program, using the sections of the original code with the
best performance. This will be deleted at a later stage, but
might be used as guidelines when designing the final version
of the label reader.
"""
import cv2
import numpy as np
# unique from main:
import os
import shutil
# unique from image_flattening:
from sklearn.linear_model import RANSACRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
# unique frm image_manager and text_reader:
from groundingdino.util.inference import Model
from segment_anything import sam_model_registry, SamPredictor
from typing import List
import pytesseract as pt
import supervision as sv
# unique from panorama_manager:
from stitching import Stitcher, stitching_error


class VideoFlow:
    """
    This class manages the flow of the frames.
    """
    def __innit__(self):
        self.frames = ManageImage()

    def start_video(self):
        self.capture = cv2.VideoCapture(self.video_path)
        while self.capture.isOpened():
            ret, frame = self.capture.read()
            if not ret:
                break
            if cv2.waitKey(25) & 0xFF == 27:
                break
            self.frames.add_image(frame)


class ManageImage:
    """
    This class extract roi and transforms the image
    to make it suitable for stitching.
    """
    def __init__(self) -> None:
        self.panorama = None
        self.last_frame = None

    def add_image(self, new_image: np.ndarray):
        """
        send new images here. Try to detect matching points
        of priory saved frame in new frame. call segmentation
        and processing when overlap threshold is reached.
        attempt to stitch new image to the panorama.
        """
        pass

    def segment_image(self):
        """
        Use dino and sam to extract ROI from image.
        """
        pass

    def map_roi_to_regression(self):
        """
        use RANSACRegressor to 'smoothen' uneven extracted
        ROI.
        """
        pass

    def evaluate_segmentation(self):
        pass

    def create_maps(self):
        pass

    def flatten_with_maps(self):
        pass


class ManageStitcher:
    """
    This class manages the stitcher and makes ensures
    quality and performance
    """
    def stitch_in_image(self):
        """
        Method to use the image manager to modify the image
        until it works for stitching.
        """
        pass

    def evaluate_result(self):
        """
        Evaluate/control performance, parameters, and processes
        of stitcher and input images.
        """
        pass

    def select_frame(self):
        """
        Method to select what frames to use in the panorama,
        to ensure balance between performance and quality.
        """
        pass
