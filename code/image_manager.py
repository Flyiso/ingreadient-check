"""
image manipulation, such as
threshold, cropping, enhancement.
"""
from groundingdino.util.inference import Model
from segment_anything import sam_model_registry, SamPredictor
from typing import List
import pytesseract as pt
import supervision as sv
import numpy as np
import torch
import math
import cv2


class ManageFrames:
    """
    manages images for panorama merge.
    """
    def __init__(self, pt_config: str) -> None:
        """
        Initializes the fame manager
        and model for label detection.
        """
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        sam_encoder_version = 'vit_h'
        sam_checkpoint_path = 'weights/sam_vit_h_4b8939.pth'
        dino_dir = '.venv/lib/python3.11/site-packages/groundingdino/'
        self.pt_config = pt_config
        self.dino_model = Model(
            f'{dino_dir}config/GroundingDINO_SwinT_OGC.py',
            f'{dino_dir}weights/groundingdino_swint_ogc.pth')
        self.classes = ['all rows of text']
        self.box_threshold = 0.35
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
        img = self.distort_perspective(frame, roi_mask)
        if isinstance(img, np.ndarray):
            img = self.enhance_frame(img)
            return img
        return False

    def segment_label(self, frame, xyxy: np.ndarray) -> np.ndarray:
        """
        Use GroundedSAM to detect and get mask for text/label area
        of image.
        """
        # why this? copyright/license reasons?
        self.sam_predictor.set_image(frame, image_format='RGB')
        result_masks = []
        for box in xyxy:
            masks, scores, _ = self.sam_predictor.predict(
                box=box, multimask_output=False)
            index = np.argmax(scores)
            result_masks.append(masks[index])
        return np.array(result_masks)

    def distort_perspective(self, frame: np.ndarray,
                            mask: np.ndarray) -> np.ndarray | bool:
        """
        Uses mask to detect ROI and return ROI with perspective corrected.
        """
        _, mask_binary = cv2.threshold(mask, mask.mean(), mask.max(),
                                       cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(mask_binary, cv2.RETR_LIST,
                                       cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) < 1:
            return False
        cont_max = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(cont_max)

        frame = frame[y:y+h, x:x+w]
        f2 = frame.copy()
        if frame.size == 0:
            return False
        cv2.imwrite('progress_images/frame_ROI.png', frame)
        mask = mask_binary[y:y+h, x:x+w]
        cv2.imwrite('progress_images/mask.png', mask)
        contours, _ = cv2.findContours(mask, cv2.RETR_LIST,
                                       cv2.CHAIN_APPROX_SIMPLE)

        if len(contours) == 0:
            return False
        cont_max = max(contours, key=cv2.contourArea)

        points_1 = cont_max
        points_2 = self.get_correction_matrix(cont_max)

        if all(isinstance(p, np.ndarray) for p in [points_1, points_2]):
            homography, _ = cv2.findHomography(points_2, points_1,
                                               method=cv2.RANSAC)
            homography = homography.astype(np.float64)
            frame = cv2.warpPerspective(frame, homography,
                                        flags=cv2.INTER_CUBIC,
                                        borderMode=cv2.BORDER_CONSTANT,
                                        borderValue=(0, 0, 0),
                                        dsize=(w, h))
            print(frame.shape)
            for a, b in zip(points_1, points_2):
                a = tuple(a[0])
                b = tuple(b)
                cv2.circle(f2, a, 2, (255, 0, 0), 2)
                cv2.circle(f2, b, 2, (0, 255, 0), 2)
            cv2.imwrite('progress_images/frame_warped.png', frame)
            cv2.imwrite('progress_images/frame_corrections.png', f2)
        return frame

    def get_approx_corners(self, contour: np.ndarray) -> np.ndarray:
        """
        Approximates the corners of a given contour,
        using the Douglas-Peucker algorithm.

        Parameters:
        - contour (numpy.ndarray): A contour represented as a numpy array.

        Returns:
        - numpy.ndarray or None: An array containing the approximated corners
        of the contour if it is a quadrilateral shape; otherwise, None.
        """
        eps_vals = [0.0001, 0.0005, 0.001, 0.005, 0.009,
                    0.01, 0.015, 0.02, 0.025, 0.03,
                    0.035, 0.04, 0.045, 0.05, 0.055,
                    0.06, 0.065, 0.07, 0.075, 0.08,
                    0.085, 0.09, 0.095, 0.1, 0.13,
                    0.15, 0.2, 0.25, 0.3, 0.35,
                    0.4, 0.045, 0.5, 0.055, 0.9]
        for eps_val in eps_vals:
            epsilon = eps_val * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            if len(approx) == 4:
                return approx

    def get_corr_max_min(self, contour: np.ndarray) -> np.ndarray:
        """
        Return pairs where height and width is the largest and smallest.
        """
        max_x = max(contour, key=lambda x: x[0][0])
        min_x = min(contour, key=lambda x: x[0][0])
        max_y = max(contour, key=lambda x: x[0][1])
        min_y = min(contour, key=lambda x: x[0][1])
        return max_x, max_y, min_x, min_y

    def sort_correction_corners(self, corners, contour) -> list:
        """
        Make sure corners are in the same order before perspective transform.
        Returns  corners of square, clockwise, with top left corner first.
        """
        sorted_horizontal = sorted(corners, key=lambda x: x[0][0])
        top_left = sorted(sorted_horizontal[:2], key=lambda x: x[0][1])[0]
        bottom_left = sorted(sorted_horizontal[:2], key=lambda x: x[0][1])[1]
        top_right = sorted(sorted_horizontal[2:], key=lambda x: x[0][1])[0]
        bottom_right = sorted(sorted_horizontal[2:], key=lambda x: x[0][1])[1]

        return np.array([top_left, top_right, bottom_right, bottom_left])

    def get_correction_matrix(self, contour: np.ndarray) -> np.ndarray:
        """
        Get an np.ndarray of len(contour) to describe the correction matrix.
        Matches each contour point to closest line in min_area rect.
        matches contour points present in corners to corners in min_area rect.
        Matches along horizontal lines keep their width values and
        modifies their height value
        Matches along vertical lines keep their height value
        and modifies their width values.

        input values:
            corners(np.ndarray, len 4):
                approximation of corners, from self.get_approx_corners.
            contour(np.ndarray):
                contour of area to create correction points for.
        output:
            correction_points(np.ndarray, len(contour)):
                correction points to use in correction matrix creation.
        """
        left, up, w, h = cv2.boundingRect(contour)
        down = up+h
        right = left+w
        corners = self.get_approx_corners(contour)

        correction_points = []
        for point in contour:
            point = list(point[0])
            if point in corners:
                _, coord = min(
                    {'up_left':
                     [math.dist(point, [left, up]),
                      (left, up)],
                     'up_right':
                     [math.dist(point, [left+w, up]),
                      (left+w, up)],
                     'down_left':
                     [math.dist(point, [left, up+h]),
                      (left, up+h)],
                     'down_right':
                     [math.dist(point, [left+w, up+h]),
                      (left+w, up+h)]}.items(),
                    key=lambda item: item[1][0])
                correction_points.append(coord[1])
                continue
            else:
                _, coord = min({'d_left': [abs(point[0] - left),
                                           (left, point[1])],
                                'd_right': [abs(right - point[0]),
                                            (right, point[1])],
                                'd_up': [abs(point[1] - up),
                                         (point[0], up)],
                                'd_down': [abs(down - point[1]),
                                           (point[0], down)]}.items(),
                               key=lambda item: item[1][0])
                correction_points.append(coord[1])

        return np.array(correction_points)

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

    def find_text(self, frame,
                  output_type: str = 'dict',
                  lang: str = 'swe') -> dict:
        """
        Returns data dictionary of the text
        found in the frame.
        """
        data = pt.image_to_data(image=frame, config=self.pt_config,
                                lang=lang, output_type=output_type)
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
        # grayscale?
        # sharpen? bilateral blur?s
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
        print(self.gs_threshold1, self.gs_threshold2)

    @staticmethod
    def get_masks(frames: list, merge_counter: int = 0) -> list:
        """
        Return masks for input frames.
        """
        masks = []
        for frame_index, frame in enumerate(frames):
            mask = np.zeros_like(frame)
            if frame_index != 0:
                mask[:, frame.shape[1]//3: (frame.shape[1]//3)*2] = 1
                print('middle mask')
            else:
                mask[:, frame.shape[1]//4:] = 1
                print('end mask')

            masks.append(mask)

        return masks
