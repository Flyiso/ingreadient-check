"""
Manages and extracts data from
merged images.
"""
import statistics
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
        extract roi from grayscale threshold and hsv threshold
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

    def warp_img(self, frames: list) -> list:
        """
        tries to wrap the images to stitch them together.
        # TODO: detect corners of frame to wrap image/correct perspective.
        # TODO: Return images with perspective wrapped correctly.
        """
        frames2 = []
        for frame in frames:
            gs_f = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gs_f = cv2.GaussianBlur(gs_f, (9, 9), 0)
            ret, thresh1 = cv2.threshold(gs_f, gs_f.mean(),
                                         gs_f.max(),
                                         cv2.THRESH_BINARY)
            contours, hierarchy = cv2.findContours(thresh1,
                                                   cv2.RETR_LIST,
                                                   cv2.CHAIN_APPROX_SIMPLE)
            contour_max = max(contours, key=cv2.contourArea)
            approx = self.get_approx_corners(contour_max)

            # perspective transform stuff
            points_1 = [list(val[0]) for val in approx]
            width = max(point[0] for point in points_1)
            height = max(point[1] for point in points_1)

            points_1, points_2 = self.get_correction_matrix_values(points_1)
            print(bool(points_1), bool(points_2))
            if bool(points_1) is not False:
                matrix = cv2.getPerspectiveTransform(np.float32(points_1),
                                                     np.float32(points_2))

                corrected = cv2.warpPerspective(frame, matrix,
                                                (width, height),
                                                flags=cv2.INTER_LINEAR)
                frames2.append(corrected)
        return frames2

    def get_correction_matrix_values(self, points_1: list) -> list:
        """
        attempts to create a matrix to match the length
        and width/height of input matrix maximum values.
        """
        # top right, bottom right, bottom_left, top_left
        points_2 = [[max(point[0] for point in points_1),
                    min(point[1] for point in points_1)],
                    [min(point[0] for point in points_1),
                    min(point[1] for point in points_1)],
                    [min(point[0] for point in points_1),
                    max(point[1] for point in points_1)],
                    [max(point[0] for point in points_1),
                    max(point[1] for point in points_1)]]
        bottom_left, top_left, bottom_right, top_right = False, False, False, False
        for pair in points_1:
            if pair[0] >= statistics.median([val[0] for val in points_1]):
                if pair[1] >= statistics.median([val[1] for val in points_1]):
                    bottom_right = pair
                else:
                    top_right = pair
            else:
                if pair[1] <= statistics.median([val[1] for val in points_1]):
                    bottom_left = pair
                else:
                    top_left = pair
        points_1 = [top_right, bottom_left, top_left, bottom_right]
        if all(points_1):
            return points_1, points_2
        return False, False

    def get_approx_corners(self, contour):
        """
        Finds 4 corners in image.
        tries different values until 4 corners are
        found.
        """
        eps_vals = [0.9, 0.5, 0.4, 0.3, 0.1,
                    0.09, 0.08, 0.07, 0.06,
                    0.05, 0.04, 0.025, 0.030,
                    0.02, 0.015, 0.01, 0.009, 0.005]
        for eps_val in eps_vals:
            epsilon = eps_val * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            if len(approx) == 4:
                print(eps_val)
                return approx
