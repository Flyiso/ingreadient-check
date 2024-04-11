"""
Manages and extracts data from
merged images.

modified version of ManageFrames in image_management.py
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
        Calls methods to get text on image and 
        method to set threshold values for image
        enhancement
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

    def get_roi_mask(self, frame):
        """
        extracts roi from grayscale threshold and hsv threshold
        separates roi by hue and binary threshold
        return mask of ROI.
        """
        frame_p = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_p = cv2.GaussianBlur(frame_p, (3, 3), 0)
        ret, bin_img = cv2.threshold(frame_p,
                                     self.gs_threshold1, self.gs_threshold2,
                                     cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))  # 11 11
        bin_img = cv2.morphologyEx(bin_img,
                                   cv2.MORPH_OPEN,
                                   kernel,
                                   iterations=1)
        bg_mask = cv2.dilate(bin_img, kernel, iterations=3)
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        cl_mask = cv2.inRange(hsv_frame, self.hue_threshold1,
                              self.hue_threshold2)
        mask_full = cv2.bitwise_and(cl_mask, bg_mask)
        contours, hierarchy = cv2.findContours(mask_full,
                                               cv2.RETR_EXTERNAL,
                                               cv2.CHAIN_APPROX_SIMPLE)
        contour_max = max(contours, key=cv2.contourArea)
        contour_rect = cv2.minAreaRect(contour_max)
        box = cv2.boxPoints(contour_rect)
        box = np.int0(box)
        mask = np.zeros_like(frame_p)
        cv2.drawContours(mask, [box], 0, (255), -1)
        mask_full = cv2.bitwise_and(mask_full, mask)
        return mask_full

    def extract_roi(self, frame):
        """"
        uses mask from self.get_roi_mask to return
        an image with roi extracted.
        """
        mask = self.get_roi_mask(frame)
        result = cv2.bitwise_and(frame, frame, mask=mask)
        return result

    def warp_img(self, frame):
        """
        tries to wrap the images to stitch them together.
        parameters:
            frames(lst): list of frames to warp.
        output:
            frame transformed to match perspective
        """
        gs_f = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
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

            frame = cv2.warpPerspective(frame, matrix,
                                        (width, height),
                                        flags=cv2.INTER_LINEAR)
            for point in points_1:
                frame = cv2.circle(frame, point, 5, (255, 22, 255), -1)
        return frame

    def _get_lines(self, frame, thresh: int = 50,
                   line_len: int = 30, gap: int = 5):
        """
        ties to find the direction of the text.
        """
        frame_g = cv2.GaussianBlur(frame, (9, 9), 0)
        frame_g = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(frame_g,
                          self.gs_threshold1, self.gs_threshold2,
                          None, 3)

        lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi/180,
                                threshold=thresh,
                                minLineLength=line_len, maxLineGap=gap)
        if lines is None:
            return None
        return lines

    def rotate_by_lines(self, frame):
        """
        calculates the median slope and rotates image accordingly.
        """
        lines = self._get_lines(frame)
        if lines is None or len(lines) == 0:
            return frame
        slopes = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if (x1-x2) != 0 and abs((y1 - y2) / (x1 - x2)) < 0.25:
                slope = (y1 - y2) / (x1 - x2)
                slopes.append(slope)
        if slopes == []:
            return frame
        angle = np.degrees(np.mean(slopes))
        height, width = frame.shape[:2]
        rotation_matrix = cv2.getRotationMatrix2D((width // 2, height // 2),
                                                  angle, 1)
        rotated_frame = cv2.warpAffine(frame, rotation_matrix, (width, height),
                                       cv2.INTER_LINEAR)
        return rotated_frame

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
                return approx

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

    def crop_to_four_corners(self, frame):
        """
        crops out ROI, limits it to a square-shape
        to manage image perspective transformation.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 30, 250, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)

        contour_max = max(contours, key=cv2.contourArea)
        approx = []
        for epsilon in range(1, 100):
            epsilon = (epsilon / 100) * cv2.arcLength(contour_max, True)
            approx_c = cv2.approxPolyDP(contour_max, epsilon, True)
            if len(approx_c) == 4:
                if cv2.contourArea(approx_c) > 0:
                    approx = [approx_c]
                    break
            approx.append(approx_c)
        largest_approximation = max(approx, key=cv2.contourArea)
        frame = self.crop_out_contour(frame, largest_approximation)
        return frame, largest_approximation

    def crop_out_contour(self, frame, largest_approximation):
        contour_mask = np.zeros_like(cv2.cvtColor(frame,  cv2.COLOR_RGB2GRAY))
        cv2.drawContours(contour_mask, [largest_approximation], -1,
                         (255), thickness=cv2.FILLED)
        image = cv2.bitwise_and(frame, frame,  mask=contour_mask)
        return image

    def crop_image(self, frame, approximation):
        """
        crop out what is outside roi
        """
        x, y, w, h = cv2.boundingRect(approximation)
        frame_roi = frame[y:y+h, x:x+w]
        return frame_roi

    def stretch_image(self, frame, contour, shape):
        """
        make image a rectangle by stretching corners together.
        """
        x_max = sorted([point[0][0] for point in contour], reverse=True)[:2]
        y_max = sorted([point[0][1] for point in contour], reverse=True)[:2]
        x, y, w, h = cv2.boundingRect(contour)
        # cv2.drawContours(frame, [contour], -1, (0, 255, 0), 2)
        fin_rect = (x, y), (x + w, y + h)
        top_l = [x, y]
        top_r = [x + w, y]
        bot_l = [x, y + h]
        bot_r = [x + w, y + h]
        points_b = np.array([[top_l], [top_r], [bot_l], [bot_r]])
        for point in contour:
            x, y = point[0]
            if x in x_max and y in y_max:
                bot_r = point
            elif x in x_max:
                top_r = point
            elif y in y_max:
                bot_l = point
            else:
                top_l = point
        points_a = np.array([top_l, top_r, bot_l, bot_r])
        frame = self.warp_frame(frame, points_a, points_b, shape)
        # cv2.rectangle(frame, fin_rect[0], fin_rect[1], (255, 0, 255), 2)
        return frame

    def warp_frame(self, frame, points_a, points_b, shape):
        he = shape[0]
        wi = shape[1]
        matrix = cv2.getPerspectiveTransform(np.float32(points_a),
                                             np.float32(points_b))

        warped = cv2.warpPerspective(frame, matrix,
                                     (he, wi),
                                     flags=cv2.INTER_NEAREST,
                                     borderMode=cv2.BORDER_REPLICATE)
        return warped

    def draw_detect_keypoints(self, frame):
        """
        Draw detected keyponts on img.
        """
        frame_gs = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        orb = cv2.ORB_create()
        key_points = orb.detect(frame_gs, None)
        key_points, _ = orb.compute(frame_gs, key_points)
        frame_gs = cv2.drawKeypoints(frame, key_points,
                                     frame,
                                     flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        return frame_gs

    def hugh_lines_mask(self, frame):
        """
        finds horizontal-ich lines in frame and draws them on frame.
        """
        lines = self._get_lines(frame, 50, 20, 40)
        if lines is None:
            return frame
        mask = np.zeros_like(frame[:,:,0], dtype=np.uint8)
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if (x1-x2) != 0 and abs((y1 - y2) / (x1 - x2)) < 0.15:
                cv2.line(mask, (x1, y1), (x2, y2),
                         (255, 255, 255), 6)
        return mask
