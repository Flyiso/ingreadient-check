"""
File that collect all image enhancement, manipulation, et cetera.
contains methods for segmentation of ROI, corrections of perspective,
correction of lightning conditions and more
"""
from image_flattening import FlattenImage
from groundingdino.util.inference import Model
from segment_anything import sam_model_registry, SamPredictor
from typing import List
import pytesseract as pt
import supervision as sv
import numpy as np
import torch
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

            de = FlattenImage(img_to_depth)
            img = de.frame
        if isinstance(img, bool):
            img = self.cylindrical_unwrap(frame, roi_mask)
        if isinstance(img, np.ndarray):
            img = self.enhance_frame(img)
            return img
        return False

    def segment_label(self, frame, xyxy: np.ndarray) -> np.ndarray:
        """
        Use GroundedSAM to detect and get mask for text/label area
        of image.
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
        print('SET MANAGER VALUES (this not run?)')
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

    def get_most_different(self, frames: list,
                           num: int, patience: int = 5) -> list:
        """
        Method for processing of multiple frames. Returns list of enhanced
        and preprocessed frames, of length n
        """
        if num >= len(frames):
            print('Number of frames less than/equal to requested return len')
            return frames
        interval = len(frames)//num
        selected_frames = []
        self.set_threshold_values(frames[0], self.find_text(frames[0]))
        for frame_id, frame in enumerate(frames):
            if frame_id % interval == 0:
                frame = self.find_label(frame)
                if isinstance(frame, np.ndarray):
                    selected_frames.append([frame, frame_id])
                    continue
                for id_diff, (frame_a, frame_b) in enumerate(
                    zip(
                        frames[frame_id-patience:frame_id-1:-1],
                        frames[frame_id:frame_id+patience])):
                    frame = self.find_label(frame_a)
                    if frame:
                        selected_frames.append[frame, frame_id-id_diff]
                        break
                    frame = self.find_label(frame_b)
                    if frame:
                        selected_frames.append[frame, frame_id+id_diff]
                        break
        return [frame[0] for frame in selected_frames]

    def is_blurry(self, frame: np.ndarray,
                  threshold: float = 300) -> bool:
        """
        Uses cv2 Laplacian to sort out images where
        not enough edges are detected.
        Returns True if image blur meet threshold.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        return laplacian_var < threshold
