"""
Manages and extracts data from
merged images.
"""
import pytesseract as pt
import numpy as np
import cv2


class ManageFrames:
    """
    Does the requested operations on frame.
    """
    def __init__(self, config: str) -> None:
        """
        Initializes the fame manager
        """
        self.config = config

    def set_manager_values(self, frame):
        """
        set values for frame manager threshold
        based on where most frame is found in
        input frame.
        """
        frame_data = self.find_text(frame)
        self.set_threshold_values(frame, frame_data)

    def process_frame(self, frame):
        """"
        Processes frame, prints information.
        """
        data = self.find_text(frame)
        print(data['text'])
        return data

    def find_text(self, frame,
                  output_type: str = 'dict',
                  lang: str = 'swe') -> dict:
        """
        Returns data dictionary of the text
        found in the frame.
        """
        data = pt.image_to_data(frame, config=self.config,
                                lang=lang,
                                output_type=output_type)
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
        self.target_area = frame
        self.hue_threshold1 = np.min(cv2.cvtColor(self.target_area,
                                                  cv2.COLOR_BGR2HSV),
                                     axis=(0, 1)).astype(int)
        self.hue_threshold2 = np.max(cv2.cvtColor(self.target_area,
                                                  cv2.COLOR_BGR2HSV),
                                     axis=(0, 1)).astype(int)
        print(self.hue_threshold1, self.hue_threshold2)
        self.hue_threshold1[0] = int(self.hue_threshold1[0]*0.33)
        self.hue_threshold2[0] = int(self.hue_threshold2[0]*1.33)
        self.hue_threshold1[1] = int(self.hue_threshold1[1]*0.33)
        self.hue_threshold2[1] = int(self.hue_threshold2[1]*1.33)
        self.hue_threshold1[2] = int(self.hue_threshold1[2]*0.33)
        self.hue_threshold2[2] = int(self.hue_threshold2[2]*1.33)
        self.gs_threshold1 = int((np.min(cv2.cvtColor(self.target_area,
                                                      cv2.COLOR_BGR2GRAY),
                                         axis=(0, 1)).astype(int))*0.33)
        self.gs_threshold2 = int(np.max(cv2.cvtColor(self.target_area,
                                                     cv2.COLOR_BGR2GRAY),
                                        axis=(0, 1)).astype(int)*1.33)
        print(self.gs_threshold1, self.gs_threshold2)

    def extract_roi(self, frame):
        """"
        extract roi from grayscale threshold and roi threshold
        """
        frame_p = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_p = cv2.GaussianBlur(frame_p, (3, 3), 0)
        ret, bin_img = cv2.threshold(frame_p,
                                     self.gs_threshold1, self.gs_threshold2,
                                     cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11))
        bin_img = cv2.morphologyEx(bin_img,
                                   cv2.MORPH_OPEN,
                                   kernel,
                                   iterations=1)
        bg_mask = cv2.dilate(bin_img, kernel, iterations=8)
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        cl_mask = cv2.inRange(hsv_frame, self.hue_threshold1,
                              self.hue_threshold2)
        mask_full = cv2.bitwise_and(cl_mask, bg_mask)
        result = cv2.bitwise_and(frame, frame, mask=mask_full)
        return result

    def wrap_img(self, frames: list):
        """
        tries to wrap the images to stitch them together.
        # TODO: detect corners of frame to wrap image/correct perspective.
        # TODO: remove usage of Pytessreact / text detection method.
        # TODO: Return images with perspective wrapped correctly.
        """
        frames2 = []
        for frame in frames:
            frame_data = self.find_text(frame)
            """matrix = cv2.getPerspectiveTransform(input_points,
                                                 output_points)
            warped_image = cv2.warpPerspective(frame,matrix,
                                               (max_width, max_height),
                                               flags = cv2.INTER_LINEAR)"""
            for i in range(len(frame_data['text'])):
                # Extract the bounding box coordinates
                x, y, w, h = frame_data['left'][i], frame_data['top'][i], frame_data['width'][i], frame_data['height'][i]
                # Draw a rectangle around the text region
                frame = cv2.rectangle(frame, (x, y), (x + w, y + h),
                                      (0, 255, 0), 2)
            frames2.append(frame)
        return frames2
