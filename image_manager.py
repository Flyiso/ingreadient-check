"""
image manipulation, such as
threshold, cropping, enhancement.
"""
from groundingdino.util.inference import load_model, load_image, predict, annotate
import cv2
#from transformers import AutoProcessor, GroundingDinoForObjectDetection
import pytesseract as pt
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
        """
        self.pt_config = pt_config
        self.model = \
            load_model(
                ".venv/lib/python3.11/site-packages/groundingdino/config/GroundingDINO_SwinT_OGC.py",
                ".venv/lib/python3.11/site-packages/groundingdino/weights/groundingdino_swint_ogc.pth")
        self.text_promt = 'ingredient label . word'
        self.box_threshold = 0.35
        self.text_threshold = 0.25

    def add_image(self, frame):
        cv2.imwrite('frame.jpeg', frame)
        image_source, image = load_image('frame.jpeg')
        boxes, logits, phrases = predict(
            model=self.model,
            image=image,
            caption=self.text_promt,
            box_threshold=self.box_threshold,
            text_threshold=self.text_threshold)
        annotated_frame = annotate(image_source=image_source,
                                   boxes=boxes,
                                   logits=logits,
                                   phrases=phrases)
        cv2.imwrite("annotated_image.jpg", annotated_frame)

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
