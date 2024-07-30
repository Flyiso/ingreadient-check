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
        masked_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_bgra = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)
        self.correct_image(frame=frame_bgra, masked=masked_img)

    def get_flattening_maps(self, masked):
        """
        use contours to get area of label.
        TODO:use difference between min and max
            and use the difference to estimate distance.
        TODO:(?) update this method(or code in general) to create the 2 maps
            at the same time. differences  in min/max distance to allow pixel
            distribution that consider depth when flattening.
        TODO: Update with separate method to call for each map making.
        """
        # Create bases for map making info
        map_base_a = masked
        map_base_b = cv2.rotate(masked, cv2.ROTATE_90_CLOCKWISE)
        edge_points_a = []
        edge_points_b = []

        # Find min and max for each row(map_a) and column(map_b)
        for pixel_row in map_base_a:
            roi = [idx_nr for idx_nr, pix in enumerate(pixel_row) if pix > 0]
            if len(roi) < 1:
                edge_points_a.append(edge_points_a[-1])
            else:
                edge_points_a.append((max(roi), min(roi)))
        for pixel_row in map_base_b:
            roi = [idx_nr for idx_nr, pix in enumerate(pixel_row) if pix > 0]
            if len(roi) < 1:
                edge_points_b.append(edge_points_b[-1])
            else:
                edge_points_b.append((max(roi), min(roi)))

        # Get lists of min\max indexes,
        # and adjust them to fit to first or second grade equations
        # A
        pixels_a_start = [edge_point[0] for edge_point in edge_points_a]
        pixels_a_end = [edge_point[1] for edge_point in edge_points_a]
        pixels_a_start = self.normalize_values(pixels_a_start)
        pixels_a_end = self.normalize_values(pixels_a_end)
        # B
        pixels_b_start = [edge_point[0] for edge_point in edge_points_b]
        pixels_b_end = [edge_point[1] for edge_point in edge_points_b]
        pixels_b_start = self.normalize_values(pixels_b_start)
        pixels_b_end = self.normalize_values(pixels_b_end)

        # TODO: make distribution of points consider est. depth(min\max diff)
        masked = cv2.cvtColor(masked, cv2.COLOR_GRAY2BGR)

        # TODO: move code from above to separate method.
        # Create empty maps for img correction
        pixel_map_a = []
        pixel_map_b = []

        # Fill Maps and print estimated edge lines on test image.
        for row_idx, (start, stop) in enumerate(zip(pixels_a_start,
                                                    pixels_a_end)):
            pixel_map_a.append(np.linspace(start, stop,
                                           len(map_base_a[0])))
            masked = cv2.circle(masked, (int(start), row_idx),
                                1, (255, 0, 255), 1)
            masked = cv2.circle(masked, (int(stop), row_idx),
                                1, (0, 255, 0), 1)
        for row_idx, (start, stop) in enumerate(zip(pixels_b_start,
                                                    pixels_b_end)):
            pixel_map_b.append(np.linspace(start, stop,
                                           len(map_base_b[0])))
            masked = cv2.circle(masked, (row_idx, int(start)),
                                1, (0, 0, 255), 1)
            masked = cv2.circle(masked, (row_idx, int(stop)),
                                1, (255, 255, 0), 1)

        cv2.imwrite('points.png', masked)

        # Transform maps for remapping and return them.
        pixel_map_a = cv2.flip(np.array(pixel_map_a),
                               1).astype(np.float32)
        pixel_map_b = cv2.rotate(np.array(pixel_map_b),
                                 cv2.ROTATE_90_COUNTERCLOCKWISE
                                 ).astype(np.float32)

        return pixel_map_a, pixel_map_b

    def distribute_by_depth_value(self, start_value: int, stop_value: int,
                                  length: int, d_map_row: np.ndarray) -> list:
        """
        create a list of values from start_value to stop_value,
        of length length where each value is evenly distributed
        from each other while depth values influence the as well.
        """
        return
        print(start_value)
        print(stop_value)
        print(length)
        print(d_map_row)
        print('......')

    def normalize_values(self, values: list) -> list:
        values = np.array(values)
        mean = np.mean(values)
        std = np.std(values)

        values[values < mean-std] = mean-std
        values[values > mean+std] = mean+std
        values = self.choose_best_fit(values)

        return values.tolist()

    def correct_image(self, frame: np.ndarray,
                      masked: np.ndarray) -> np.ndarray:
        """
        use estimated size to correct
        the images shape and perspective
        frame: BGRA image
        depth_mask: GRAYSCALE image
        """
        map_a, map_b = self.get_flattening_maps(masked)
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
        """
        TODO: update this to compare rmse sum  opposite 'lines' and choose
            best fit by comparing them to connected lines
        """
        y_fit_line = self.fit_to_line(y)
        y_fit_quad = self.fit_to_quadratic(y)

        # Calculate RMSE for both models
        rmse_line = np.sqrt(mean_squared_error(y, y_fit_line))
        rmse_quad = np.sqrt(mean_squared_error(y, y_fit_quad))

        if abs(rmse_line-rmse_quad) <= 5:
            print(f'LINEAR,  {rmse_line}')
            return y_fit_line
        else:
            print(f'SECOND GRADE, {rmse_quad}')
            return y_fit_quad
