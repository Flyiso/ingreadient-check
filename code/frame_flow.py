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
    def __init__(self,
                 pt_config: str = 'oem-- 3 --psm 6') -> None:
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

    def find_label(self, frame) -> np.ndarray | bool:
        """
        Finds label on product, calls methods to try to
        correct its perspective, and enhances it.
        Returns corrected frame if successfull,
        else returns False

        returns mask for roi if found, else returns False
        """
        detections = self.dino_model.predict_with_classes(
            image=frame,
            classes=self.enhance_class_names(),
            box_threshold=self.box_threshold,
            text_threshold=self.text_threshold)
        detections.mask = self.segment_label(frame, detections.xyxy)
        mask_annotator = sv.MaskAnnotator()
        empty_image = np.zeros((frame.shape), dtype=np.uint8)
        roi_mask = cv2.cvtColor(mask_annotator.annotate(scene=empty_image,
                                                        detections=detections,
                                                        opacity=1),
                                cv2.COLOR_BGR2GRAY)
        _, binary_mask = cv2.threshold(roi_mask, 5, 255, cv2.THRESH_BINARY)
        cv2.imwrite('mask.png', roi_mask)
        contours, _ = cv2.findContours(binary_mask, 1, 2)
        img = False
        if len(contours) >= 1:
            x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
            img_to_depth = cv2.bitwise_and(frame,
                                           cv2.cvtColor(binary_mask,
                                                        cv2.COLOR_GRAY2RGB),
                                           binary_mask)
            img_to_depth = img_to_depth[y:y+h, x:x+w]
        perspective_corrected = FlattenImage(img_to_depth)
        img = perspective_corrected.frames
        if isinstance(img, np.ndarray):
            img = self.enhance_frame(img)
            return img
        return False

    def segment_image(self, frame, xyxy: np.ndarray):
        """
        Use sam to extract ROI from image.
        """
        self.sam_predictor.set_image(frame, image_format='RGB')
        result_masks = []
        for box in xyxy:
            masks, scores, _ = self.sam_predictor.predict(
                box=box, multimask_output=False)
            index = np.argmax(scores)
            result_masks.append(masks[index])
        return np.array(result_masks)

    def enhance_class_names(self) -> List[str]:
        """
        Enhances class names by specifying prompt details.
        Returns updated list.
        """
        return [
            f"{class_name}"
            for class_name
            in self.classes
        ]

    def set_manager_values(self, frame):
        """
        set values for frame manager threshold
        based on where most frame is found in
        input frame.
        Calls methods to get text on image and
        method to set threshold values for image
        enhancement
        """
        frame_data = self.find_text(frame)
        self.set_threshold_values(frame, frame_data)

    def find_text(self, frame: np.ndarray,
                  output_type: str = 'dict') -> dict:
        """
        Returns data dictionary of the text
        found in the frame.
        """
        print(type(frame))
        data = pt.image_to_data(image=frame, config=self.pt_config,
                                output_type=output_type)
        return data

    def enhance_frame(self,
                      frame: np.ndarray) -> np.ndarray:
        """
        Enhances frame to make it easier to process
        by stitcher and pytesseract.
        returns enhanced frame.
        """
        # hue thresh to correct light
        frame = self.enhance_text_lightness(frame)
        return frame

    def enhance_text_lightness(self, frame):
        """
        Enhance text by perceptual lightness
        """
        frame2 = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        clahe = cv2.createCLAHE(clipLimit=1.7, tileGridSize=(5, 5))
        frame_planes = list(cv2.split(frame2))
        frame_planes[0] = clahe.apply(frame_planes[0])
        frame2 = cv2.merge(frame_planes)
        frame2 = cv2.cvtColor(frame2, cv2.COLOR_LAB2BGR)
        return frame2

    def set_threshold_values(self, frame, frame_data: dict):
        """
        adjust threshold to the most text-populated area.
        get threshold for grayscale and HSV values.
        """
        img_x = int(np.mean(frame_data['top']))
        img_y = int(np.mean(frame_data['left']))
        img_w = int(np.mean(frame_data['width']))
        img_h = int(np.mean(frame_data['height']))
        frame = frame[img_x:img_x+img_w,
                      img_y:img_y+img_h]
        self.target_area = frame
        self.hue_threshold1 = np.min(cv2.cvtColor(self.target_area,
                                                  cv2.COLOR_BGR2HSV),
                                     axis=(0, 1)).astype(int)
        self.hue_threshold2 = np.max(cv2.cvtColor(self.target_area,
                                                  cv2.COLOR_BGR2HSV),
                                     axis=(0, 1)).astype(int)
        print(self.hue_threshold1, self.hue_threshold2)
        self.hue_threshold1[0] = int(self.hue_threshold1[0]*0.35)
        self.hue_threshold2[0] = int(self.hue_threshold2[0]*1.35)
        self.hue_threshold1[1] = int(self.hue_threshold1[1]*0.35)
        self.hue_threshold2[1] = int(self.hue_threshold2[1]*1.35)
        self.hue_threshold1[2] = int(self.hue_threshold1[2]*0.35)
        self.hue_threshold2[2] = int(self.hue_threshold2[2]*1.35)
        self.gs_threshold1 = int((np.min(cv2.cvtColor(self.target_area,
                                                      cv2.COLOR_BGR2GRAY),
                                         axis=(0, 1)).astype(int))*0.35)
        self.gs_threshold2 = int(np.max(cv2.cvtColor(self.target_area,
                                                     cv2.COLOR_BGR2GRAY),
                                        axis=(0, 1)).astype(int)*1.35)

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
        self.frame_manager = ManageImage()

    def add_image(self, image: np.ndarray) -> bool:
        """
        Attempt to pre-process image and add to panorama

        :param image: array, 3-channel img to attempt to stitch.
        :output: Bool- true if stitching successfull.
        """
        if self.panorama is None:
            self.frame_manager.set_manager_values(image)
        img = self.frame_manager.find_label(image)
        if img:
            img = self.frame_manager.find_label(image)

        stitched_success = self.stitch_to_panorama(img)
        return stitched_success

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
        if self.panorama is None:
            self.panorama = frame
            return True


class FlattenImage:
    """
    Class to correct an image to make it appear flat.
    """
    def __init__(self, frame: np.ndarray) -> None:
        """
        Attempt to match left and right edges of ROI to ml model.

        :param frame: numpy array of the image to create map from.
        :param evaluation_class: class for model evaluation. optional.
        """
        masked_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_bgra = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)
        self.correct_image(frame=frame_bgra, masked=masked_img)

    def correct_image(self, frame: np.ndarray,
                      masked: np.ndarray) -> np.ndarray:
        """
        use estimated size to correct the images shape and perspective

        :param frame: BGRA image
        :param masked: GRAYSCALE image
        """
        self.map_a, self.map_b = self.get_flattening_maps(masked)
        cv2.imwrite('map_b_vertical.png', self.map_b)
        cv2.imwrite('map_a_horizontal.png', self.map_a)
        flattened_image = cv2.remap(frame, self.map_a, self.map_b,
                                    interpolation=cv2.INTER_NEAREST,
                                    borderMode=cv2.BORDER_WRAP)

        gray = cv2.cvtColor(flattened_image, cv2.COLOR_BGRA2GRAY)
        flattened_image = cv2.cvtColor(flattened_image, cv2.COLOR_BGRA2BGR)
        _, mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY_INV)
        flattened_image = self.inpaint_img(flattened_image, mask)
        cv2.imwrite('flat_img.png', flattened_image)
        self.frame = flattened_image

    def get_flattening_maps(self, masked: np.ndarray):
        """
        Method to create 2 maps for flattening/remapping.

        :param masked: 2 dimensional numpy array where all except ROI is 0.
        :return: 2 arrays of same shape as masked, describing re-map values.
        """
        map_base = masked
        map_base_rotated = cv2.rotate(masked, cv2.ROTATE_90_COUNTERCLOCKWISE)

        edge_points = self.get_edge_points(map_base)
        edge_points = self.manage_outliers(edge_points)
        edge_points_rotated = self.get_edge_points(map_base_rotated)
        edge_points_rotated = self.manage_outliers(edge_points_rotated)

        pixel_map_a, masked = self.get_maps(True,
                                            edge_points,
                                            len(map_base[0]), masked,
                                            (255, 255, 0), (0, 0, 255))
        pixel_map_b, masked = self.get_maps(True,
                                            edge_points_rotated,
                                            len(map_base_rotated[0]), masked,
                                            (255, 0, 255), (0, 255, 0))
        cv2.imwrite('points.png', masked)

        pixel_map_a = np.array(pixel_map_a).astype(np.float32)
        pixel_map_b = cv2.rotate(np.array(pixel_map_b),
                                 cv2.ROTATE_90_CLOCKWISE).astype(np.float32)
        return pixel_map_a, pixel_map_b

    def get_edge_points(self, map_base: np.ndarray,
                        reverse: bool = False) -> list:
        """
        returns [list of (min/start, max/end) for each row.]
        indexes of where ROI of each row of the map start and end.
        Return as list of pairs for each row, starting with lowest value.
        """
        edge_points = []
        for pixel_row in map_base:
            roi = [idx_nr for idx_nr, pix in enumerate(pixel_row) if pix > 0]
            if len(roi) < 1:
                edge_points.append(edge_points[-1])
            else:
                edge_points.append((min(roi), max(roi)))
        if reverse:
            edge_points = [pair[::-1] for pair in edge_points]
        return edge_points

    @staticmethod
    def manage_outliers(edge_points: list):
        """
        Replace outlier points with RANSAC regressor and np.linspace.

        :param  edge_points: List of where  edges where detected
        :output: List where outlier edges have been replaced.
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

    def get_maps(self, first_value: bool,
                 pixel_pairs: list,
                 len_active: int, image: np.ndarray,
                 color_1: tuple, color_2: tuple):
        """
        generate the actual maps.
        TODO: make this take width/difference between numbers into account.
              for roi length index at the opposite direction.
        """
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

    def inpaint_img(self, img, mask):
        img = cv2.inpaint(img, mask, 3, cv2.INPAINT_TELEA)
        return img
