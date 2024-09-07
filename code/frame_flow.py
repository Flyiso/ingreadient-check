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
import torch
# unique from panorama_manager:
from stitching import Stitcher, stitching_error


class VideoFlow:
    """
    This class manages the flow of the frames and decides what frames
    to create panorama from.
    """
    def __innit__(self):
        self.previous_frame = None
        self.saved_count = 0
        # self.panorama_manager = PanoramaManager()
        # self.panorama = self.panorama_manager.panorama

    def start_video(self):
        self.capture = cv2.VideoCapture(self.video_path)
        while self.capture.isOpened():
            ret, frame = self.capture.read()
            if not ret:
                break
            if cv2.waitKey(25) & 0xFF == 27:
                break
            self.check_image(frame)
            if isinstance(self.panorama, np.ndarray):
                cv2.imshow('frame', self.panorama)

    def check_image(self, frame: np.ndarray):
        """
        check frame and try to add it to panorama.

        :param frame: input frame- numpy array- 3 channel.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if not self.check_blur(gray):
            return
        if not self.check_difference(gray):
            return
        if not self.panorama_manager.add_image(frame):
            self.reset_to_previous()
            return
        self.panorama = self.panorama_manager.panorama

    def check_difference(self, frame: np.ndarray,
                         threshold: int = 67000000) -> bool:
        """
        Check if image difference threshold is reached, and
        save image if so.

        :param frame: np.ndarray- grayscale image to compare.
        :param threshold: int. threshold for image difference.
        :output: True if image is different enough.
        """
        self.memory = None
        if self.previous_frame is not None:
            diff = cv2.absdiff(self.previous_frame, frame).sum

            if diff > threshold:
                self.memory = self.previous_frame
                self.saved_count += 1
                self.previous_frame = frame
                return True
        else:
            self.previous_frame = frame
            self.saved_count += 1
            return True
        return False

    def reset_to_previous(self) -> None:
        """
        Reset images to previous state.
        """
        self.saved_count -= 1
        self.previous_frame = self.memory

    @staticmethod
    def check_blur(frame: np.ndarray,
                   threshold: float = 300) -> bool:
        """
        Check if frame is sharp enough to stitch.

        :param frame: np.ndarray, image(grayscale) to check blur on.
        :param threshold: float, optional-defaults to 300.
        :output: True if not to blurry.
        """
        laplacian_var = cv2.Laplacian(frame, cv2.CV_64F).var()
        return laplacian_var < threshold
