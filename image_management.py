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
        separates roi by hue and binary threshold
        Return extracted roi
        """
        frame_p = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_p = cv2.GaussianBlur(frame_p, (3, 3), 0)
        ret, bin_img = cv2.threshold(frame_p,
                                     self.gs_threshold1, self.gs_threshold2,
                                     cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (8, 8)) #11 11
        bin_img = cv2.morphologyEx(bin_img,
                                   cv2.MORPH_OPEN,
                                   kernel,
                                   iterations=1)
        bg_mask = cv2.dilate(bin_img, kernel, iterations=9)#8
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        cl_mask = cv2.inRange(hsv_frame, self.hue_threshold1,
                              self.hue_threshold2)
        mask_full = cv2.bitwise_and(cl_mask, bg_mask)
        result = cv2.bitwise_and(frame, frame, mask=mask_full)
        inverted_mask = np.invert(mask_full)
        contours, hierarchy = cv2.findContours(mask_full,
                                               cv2.RETR_EXTERNAL,
                                               cv2.CHAIN_APPROX_SIMPLE)
        contour_max = max(contours, key=cv2.contourArea)
        contour_rect = cv2.minAreaRect(contour_max)
        x, y, width, height = cv2.boundingRect(contour_max)
        box = cv2.boxPoints(contour_rect)
        box = np.int0(box)
        result = cv2.drawContours(result, [box], 0, (255, 2, 222), 2)
        result = cv2.rectangle(result, (x, y), (x+width, y+height), (0, 255, 255), 2)
        return result

    def warp_img(self, frames: list) -> list:
        """
        tries to wrap the images to stitch them together.
        parameters:
            frames(lst): list of frames to warp.
        output:
            lift of frames transformed to match perspective
        """
        print('wrap images..?')
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
            if isinstance(approx, np.ndarray):
                points_1 = [list(val[0]) for val in approx]
                width = max(point[0] for point in points_1)
                height = max(point[1] for point in points_1)
                points_1, points_2 = self.get_correction_matrix_values(
                    points_1)
            else:
                points_1 = False

            if isinstance(approx, np.ndarray) and bool(points_1) is True:
                matrix = cv2.getPerspectiveTransform(np.float32(points_1),
                                                     np.float32(points_2))

                corrected = cv2.warpPerspective(frame, matrix,
                                                (width, height),
                                                flags=cv2.INTER_LINEAR)
                for point in points_1:
                    corrected = cv2.circle(frame, point, 5, (255, 22, 255), -1)
                
                frames2.append(corrected)
            else:
                frames2.append(frame)
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
        bottom_left, top_left, bottom_right, top_right = [False, False,
                                                          False, False]
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

    def enhance_text_lightness(self, frame):
        """
        Enhance text by perceptual lightness
        """
        # CLAHE:
        frame2 = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        clahe = cv2.createCLAHE(clipLimit=1.7, tileGridSize=(5, 5))
        frame_planes = list(cv2.split(frame2))
        frame_planes[0] = clahe.apply(frame_planes[0])
        frame2 = cv2.merge(frame_planes)
        frame2 = cv2.cvtColor(frame2, cv2.COLOR_LAB2BGR)
        return frame2


# NOTES/ SAVE METHODS
# BILATERAL FILTER (parameter values):
# 1 pixel neighborhood size
# 2 sigmaColor(how similar colors need to be to mix together)
# 3 distance of pixels mixed together(within sigma values)
"""bf_gray = cv2.bilateralFilter(cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY),
                                5, 9, 77)
"""
# THRESHOLDS- ADAPTIVE
# 1 max value (up to 255)
# 2 adaptive method:
#   a) cv.ADAPTIVE_THRESH_MEAN_C:
#      The threshold value is the mean of the
#      neighborhood area minus the constant C.
#   b) cv.ADAPTIVE_THRESH_GAUSSIAN_C
#      The threshold value is a gaussian-weighted sum
#      of the neighborhood values minus the constant C
# 3 threshold type
# 4 block_size(odd int?)/neighborhood size
# 5 C- a constant subtracted from mean or weighted sum of neigh. px.

# THRESHOLDS- OTSUS BINARIZATION:
# 1 thresh
# 2 maxval
# 3 type
# 4 dist
"""thresh, img = cv2.threshold(bf_gray,
                            self.gs_threshold1, 255,
                            cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)"""
# CONTOURS
# 1 mode:
#   a) cv2.RETR_EXTERNAL-
#       retrieve extreme outer contours
#   b) cv2.RETR_LIST-
#       retrieves all contours without hierarchy
#   c) cv2.RETR_CCOMP-
#       retrieves all contours. organizes in 2-level hierarchy.
#           1- external boundaries, contours inside 2.
#           2- contours inside external
#   d) cv2.RETR_TREE:
#       all contours, in full hierarchy of nested contours.
# 2 Method:
#   a) cv2.CHAIN_APPROX_NONE-
#       store all boundary points.
#   b) cv2.CHAIN_APPROX_SIMPLE-
#       store ends of boundary points representing a line. saves memory
"""contours, hir = cv2.findContours(img, mode=cv2.RETR_TREE,
                                    method=cv2.CHAIN_APPROX_SIMPLE)"""
# DRAW ON FRAME
"""for num,  contour in enumerate(contours):
    if hir[0][num][0] != -1 | np.max(hir):
        frame = cv2.drawContours(frame2, contours=[contour],
                                    contourIdx=0,
                                    color=(255, 255, 0), thickness=2)"""
