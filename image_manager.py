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
        #detections.class_id
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
            masks, scores, _ = self.sam_predictor.predict(
                box=box, multimask_output=False)
            index = np.argmax(scores)
            result_masks.append(masks[index])
        return np.array(result_masks)

    def distort_perspective(self, frame: np.ndarray,
                            mask: np.ndarray) -> np.ndarray:
        """
        Uses mask to detect ROI and return ROI with perspective corrected.
        """
        _, mask_binary = cv2.threshold(mask, mask.mean(), mask.max(),
                                       cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(mask_binary, cv2.RETR_LIST,
                                       cv2.CHAIN_APPROX_SIMPLE)
        cont_max = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(cont_max)

        frame = frame[y-int(0.05*h):y+h+int(0.05*h),
                      x-int(0.05*w):x+w+int(0.05*w)]
        cv2.imwrite('frame_c.png', frame)
        mask = mask_binary[y-int(0.05*h):y+h+int(0.05*h),
                           x-int(0.05*w):x+w+int(0.05*w)]
        contours, _ = cv2.findContours(mask, cv2.RETR_LIST,
                                       cv2.CHAIN_APPROX_SIMPLE)
        cont_max = max(contours, key=cv2.contourArea)

        points_1 = self.get_approx_corners(cont_max)
        points_2 = self.get_corr_corners(cont_max)

        if all(isinstance(p, np.ndarray) for p in [points_1, points_2]):
            homography, _ = cv2.findHomography(points_1, points_2)
            homography = homography.astype(np.float64)
            frame = cv2.warpPerspective(frame, homography,
                                        dsize=cv2.boundingRect(cont_max)[2:])
            cv2.imwrite('frame_w.png', frame)
        return frame

    def get_corr_corners(self, contour: np.ndarray) -> np.ndarray:
        """
        Get corners to warp ROI to.
        """
        x, y, w, h = cv2.boundingRect(contour)
        print(x, y, w, h)
        corners = self.sort_correction_corners(np.array(
            [[[0, y]], [[w, y]], [[w, h]], [[0, h]]]), contour)
        return corners

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
                print(eps_val)
                approx = self.sort_correction_corners(approx, contour)
                return approx

    def get_corr_max_min(self, contour: np.ndarray) -> np.ndarray:
        """
        Return pairs where height and width is the largest and smallest.
        """
        max_x = max(contour, key=lambda x: x[0][0])
        min_x = min(contour, key=lambda x: x[0][0])
        max_y = max(contour, key=lambda x: x[0][1])
        min_y = min(contour, key=lambda x: x[0][1])
        print(max_x, max_y, min_x, min_y)
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
        max_x, max_y, min_x, min_y = self.get_corr_max_min(contour)
        return np.array([top_left, max_x, top_right, max_y, bottom_right, min_x, bottom_left, min_y])

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
