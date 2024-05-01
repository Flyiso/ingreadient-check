"""
image manipulation, such as
threshold, cropping, enhancement.
"""
from groundingdino.util.inference import Model
from segment_anything import sam_model_registry, SamPredictor
#from transformers import AutoProcessor, GroundingDinoForObjectDetection
from IPython.display import display, HTML
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
        self.pt_config = pt_config
        self.dino_model = \
            Model(
                '.venv/lib/python3.11/site-packages/groundingdino/config/GroundingDINO_SwinT_OGC.py',
                '.venv/lib/python3.11/site-packages/groundingdino/weights/groundingdino_swint_ogc.pth')
        self.classes = ['lines of text']
        self.box_threshold = 0.35
        self.text_threshold = 0.25
        self.sam_model = sam_model_registry[sam_encoder_version](
            checkpoint=sam_checkpoint_path).to(device=device)
        self.sam_predictor = SamPredictor(self.sam_model)

    def find_label(self, frame):
        cv2.imwrite('frame.jpeg', frame)
        detections = self.dino_model.predict_with_classes(
            image=frame,
            classes=self.enhance_class_names(),
            box_threshold=self.box_threshold,
            text_threshold=self.text_threshold)
        print(f'type:{type(detections)} len:{len(detections)}')
        print(f'detections: {detections}')
        detections.class_id
        detections.mask = self.segment_label(frame, detections.xyxy)
        mask_annotator = sv.MaskAnnotator()
        empty_image = np.zeros((frame.shape), dtype=np.uint8)
        roi_mask = cv2.cvtColor(mask_annotator.annotate(scene=empty_image,
                                                        detections=detections,
                                                        opacity=1),
                                cv2.COLOR_BGR2GRAY)
        img = self.distort_perspective(frame, roi_mask)
        cv2.imwrite('img_disorted.png', img)

    def segment_label(self, frame, xyxy: np.ndarray) -> np.ndarray:
        """
        Use GroundedSAM to detect and get mask for text/label area
        of image.
        """
        # why this? copyright/license reasons?
        self.sam_predictor.set_image(frame, image_format='RGB')
        result_masks = []
        for box in xyxy:
            masks, scores, logits = self.sam_predictor.predict(
                box=box, multimask_output=False)
            index = np.argmax(scores)
            result_masks.append(masks[index])
        return np.array(result_masks)

    def distort_perspective(self, frame: np.ndarray,
                            mask: np.ndarray) -> np.ndarray:
        """
        Uses mask to detect ROI and return ROI with perspective corrected.
        """
        # _, mask_binary = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY)
        _, mask_binary = cv2.threshold(mask, mask.mean(), mask.max(),
                                       cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(mask_binary, cv2.RETR_LIST,
                                       cv2.CHAIN_APPROX_SIMPLE)
        print(f'contours total: {len(contours)}')
        print(f'contour: {contours}')

        cont_max = max(contours, key=cv2.contourArea)
        points_2 = self.sort_correction_corners(
            np.array([(0, 0), (frame.shape[0], 0),
                      (frame.shape[:2]), (0, frame.shape[1])]))
        points_1 = self.get_approx_corners(cont_max)

        if isinstance(points_1, np.ndarray):
            points = [list(val[0]) for val in points_1]
            width = max(point[0] for point in points)
            height = max(point[1] for point in points)
        else:
            points_1 = False

        if isinstance(points_1, np.ndarray):
            print(f'p1-{type(points_1)}- {points_1}')
            print(f'p2-{type(points_2)}- {points_2}')
            matrix = cv2.getPerspectiveTransform(np.float32(points_1),
                                                 np.float32(points_2))

            frame = cv2.warpPerspective(frame, matrix,
                                        (width, height),
                                        flags=cv2.INTER_LINEAR)
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
        eps_vals = [0.99, 0.9, 0.5, 0.4, 0.3, 0.1,
                    0.09, 0.08, 0.07, 0.06,
                    0.05, 0.04, 0.025, 0.030,
                    0.02, 0.015, 0.01, 0.009,
                    0.005, 0.001, 0.0005, 0.0001]
        for eps_val in eps_vals:
            epsilon = eps_val * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            print(f'{eps_val}-{len(approx)}')
            print(approx)
            if len(approx) == 4:
                print(f'\napprox{type(approx)}: \n{approx}\n')
                approx = self.sort_correction_corners(approx)
                print(f'\napprox sorted{type(approx)}: \n{approx}\n')
                return approx
        print('\nno approx?\n')

    def sort_correction_corners(self, corners) -> list:
        """
        Make sure corners are in the same order before perspective transform.
        Returns  corners of square, clockwise, with top left corner first. 
        """
        sorted_horizontal = sorted(corners, key=lambda x: x[0][0])
        top_left = sorted(sorted_horizontal[:2], key=lambda x: x[0][1])[0]
        bottom_left = sorted(sorted_horizontal[:2], key=lambda x: x[0][1])[1]
        top_right = sorted(sorted_horizontal[2:], key=lambda x: x[0][1])[0]
        bottom_right = sorted(sorted_horizontal[2:], key=lambda x: x[0][1])[1]
        return np.array([top_left, bottom_left, top_right, bottom_right])

    def enhance_class_names(self) -> List[str]:
        """
        Enhances class names by specifying prompt details.
        Returns updated list.
        """
        return [
            f"all full {class_name}s"
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

    def prepare_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Use methods to enhance, crop and create mask for input
        frame, return input frame and mask for frame.
        """
        pass

    def return_frame_mask(self, frame: np.ndarray) -> np.ndarray:
        """
        Return mask for frame
        """
        pass

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
