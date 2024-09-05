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
        self.previous_frame = None
        self.saved_count = 0
        self.panorama_manager = PanoramaManager()
        self.panorama = self.panorama_manager.panorama

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


class ManageImage:
    """
    This class extract roi and transforms the image
    to make it suitable for stitching.
    """
    def __init__(self) -> None:
        self.panorama = None
        self.last_frame = None
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        sam_encoder_version = 'vit_h'
        sam_checkpoint_path = 'weights/sam_vit_h_4b8939.pth'
        dino_dir = '.venv/lib/python3.11/site-packages/groundingdino/'
        self.pt_config = pt_config
        self.focal_length = False
        self.dino_model = Model(
            f'{dino_dir}config/GroundingDINO_SwinT_OGC.py',
            f'{dino_dir}weights/groundingdino_swint_ogc.pth')
        self.classes = ['Text or image']  # ['text or image']
        self.box_threshold = 0.25  # 0.35
        self.text_threshold = 0.25
        self.sam_model = sam_model_registry[sam_encoder_version](
            checkpoint=sam_checkpoint_path).to(device=device)
        self.sam_predictor = SamPredictor(self.sam_model)

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
        edges = [[edge_point[0] for edge_point in edge_points],
                 [edge_point[1] for edge_point in edge_points]]
        for point_index, points in enumerate(edges):
            first = points[:len(points)//3]
            second = points[len(points)//3:(len(points)//3)*2]
            third = points[(len(points)//3)*2:]
            x = np.linspace(0, len(points), len(points))
            X = x[:, np.newaxis]
            if np.median(first) < np.median(second) > np.median(third):
                model = make_pipeline(PolynomialFeatures(2), RANSACRegressor())
                model.fit(X, points)
                points = model.predict(X)
            elif np.median(first) > np.median(second) < np.median(third):
                model = make_pipeline(PolynomialFeatures(2), RANSACRegressor())
                model.fit(X, points)
                points = model.predict(X)
            else:
                points = np.linspace(np.mean(first),
                                     np.mean(third), len(points))
            edges[point_index] = points

        filtered = []
        for point_1, point_2 in zip(edges[0], edges[1]):
            filtered.append((point_1, point_2))
        return filtered

    def evaluate_segmentation(self):
        pass

    def create_maps(self):
        pixel_map = []
        for row_idx, (start, stop) in enumerate(pixel_pairs):
            pixel_map.append((np.linspace(start, stop, len_active)))
            if first_value:
                location_start = (int(start), row_idx)
                location_stop = (int(stop), row_idx)
            else:
                location_start = (row_idx, int(start))
                location_stop = (row_idx, int(stop))

            masked = cv2.circle(image, location_start,
                                1, color_1, 1)
            masked = cv2.circle(masked, location_stop,
                                1, color_2, 1)
        return pixel_map, masked

    def flatten_with_maps(self):
        pass


class PanoramaManager:
    """
    This class manages the stitcher and makes ensures
    quality and performance
    """
    def __init__(self):
        """
        stitch images to panorama.
        """
        self.panorama = None

    def add_image(self, image: np.ndarray) -> bool:
        """
        Attempt to pre-process image and add to panorama

        :param image: array, 3-channel img to attempt to stitch.
        :output: Bool- true if stitching successfull.
        """
        # pre-process image
        # check if panorama present/first frame
        # attempt stitching
        # return bool
        pass

    def evaluate_result(self):
        """
        Evaluate/control performance, parameters, and processes
        of stitcher and input images.
        """
        pass

    def stitch_to_panorama(self, frame: np.ndarray):
        """
        Stitch the new frame to the panorama.

        :param frame: array- 3channel image
        :return: Bool- True if stitcher successful
        """
        pass
