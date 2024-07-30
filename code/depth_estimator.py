"""
File for test if depth estimation could help with
correcting the frames.

diff of max & min in label area = save, to not run this all the time
run this once or a few times? when?
TODO: make this take cyliners/circles into concideration
"""
import numpy as np
import cv2
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error


class DepthCorrection:
    """
    Class that finds depth and
    correct the image's perspective/flatten it.
    init sets min/max diff
    method just correct by values?
    """
    def __init__(self, frame: np.ndarray) -> None:
        depth = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_bgra = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)
        self.correct_image(frame=frame_bgra, depth_mask=depth)

    def flatten_label_by_contour(self, depth_img):
        """
        use contours to get area of label.
        """
        edge_points = []
        for pixel_row in depth_img:
            roi = [idx_nr for idx_nr, pix in enumerate(pixel_row) if pix > 1]
            if len(roi) < 1:
                edge_points.append(edge_points[-1])
            else:
                edge_points.append((max(roi), min(roi)))

        pixels_start = [edge_point[0] for edge_point in edge_points]
        pixels_end = [edge_point[1] for edge_point in edge_points]
        pixels_start = self.normalize_values(pixels_start)
        pixels_end = self.normalize_values(pixels_end)

        pixel_map = []
        for row_idx, (start, stop) in enumerate(zip(pixels_start, pixels_end)):
            pixel_map.append(np.linspace(start, stop, len(depth_img[0])))
            self.distribute_by_depth_value(start, stop,
                                           len(depth_img[0]),
                                           depth_img[row_idx])
            depth_img = cv2.circle(depth_img, (row_idx, int(start)),
                                   1, (255), 1)
            depth_img = cv2.circle(depth_img, (row_idx, int(stop)),
                                   1, (0), 1)
        cv2.imwrite('points.png', depth_img)
        input('wait?')
        return np.array(pixel_map)

    def distribute_by_depth_value(self, start_value: int, stop_value: int,
                                  length: int, d_map_row: np.ndarray) -> list:
        """
        create a list of values from start_value to stop_value,
        of length length where each value is evenly distributed
        from each other while depth values influence the as well.
        """
        print(start_value)
        print(stop_value)
        print(length)
        print(d_map_row)
        print('......')

    def normalize_values(self, values):
        values = np.array(values)
        mean = np.mean(values)
        std = np.std(values)*0.33

        values[values < mean-std] = mean-std
        values[values > mean+std] = mean+std
        values = self.choose_best_fit(values)

        return values.tolist()

    def correct_image(self, frame: np.ndarray,
                      depth_mask: np.ndarray) -> np.ndarray:
        """
        use estimated size to correct
        the images shape and perspective
        frame: BGRA image
        depth_mask: GRAYSCALE image
        """
        map_a = cv2.flip(
            self.flatten_label_by_contour(depth_mask), 1).astype(np.float32)
        map_b = cv2.rotate(self.flatten_label_by_contour(
            cv2.rotate(depth_mask, cv2.ROTATE_90_CLOCKWISE)),
            cv2.ROTATE_90_COUNTERCLOCKWISE).astype(np.float32)
        cv2.imwrite('map_b.png', map_b)
        cv2.imwrite('map_a.png', map_a)
        flattened_image = cv2.remap(frame, map_a, map_b,
                                    interpolation=cv2.INTER_NEAREST,
                                    borderMode=cv2.BORDER_WRAP)

        gray = cv2.cvtColor(flattened_image, cv2.COLOR_BGRA2GRAY)
        flattened_image = cv2.cvtColor(flattened_image, cv2.COLOR_BGRA2BGR)
        _, mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY_INV)
        flattened_image = self.inpaint_img(flattened_image, mask)
        cv2.imwrite('flat_img.png', flattened_image)
        self.frame = flattened_image

    def inpaint_img(self, img, mask):
        img = cv2.inpaint(img, mask, 3, cv2.INPAINT_TELEA)
        return img

    def fit_to_line(self, y):
        x = np.arange(len(y)).reshape(-1, 1)
        y = np.array(y)

        model = LinearRegression()
        model.fit(x, y)

        y_fit = model.predict(x)
        return y_fit

    def fit_to_quadratic(self, y):
        x = np.arange(len(y)).reshape(-1, 1)
        y = np.array(y)

        poly = PolynomialFeatures(degree=2)
        x_poly = poly.fit_transform(x)

        model = LinearRegression()
        model.fit(x_poly, y)

        y_fit = model.predict(x_poly)
        return y_fit

    def choose_best_fit(self, y):
        y_fit_line = self.fit_to_line(y)
        y_fit_quad = self.fit_to_quadratic(y)

        # Calculate RMSE for both models
        rmse_line = np.sqrt(mean_squared_error(y, y_fit_line))
        rmse_quad = np.sqrt(mean_squared_error(y, y_fit_quad))

        if rmse_line < rmse_quad:
            return y_fit_line
        else:
            return y_fit_quad
