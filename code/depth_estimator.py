"""
File for test if depth estimation could help with
correcting the frames.

diff of max & min in label area = save, to not run this all the time
run this once or a few times? when?
TODO: make this take cylinders/circles into consideration
"""
import numpy as np
import cv2
from sklearn.linear_model import RANSACRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures


class DepthCorrection:
    """
    Class to correct the image's perspective/flatten it.
    init sets min/max diff
    method just correct by values?
    """
    def __init__(self, frame: np.ndarray) -> None:
        """
        Attempt to match left and right edges of ROI to ml model.

        :param frame: numpy array of the image to create map from.
        :param evaluation_class: class for model evaluation. optional.
        """
        self.models = []  # store all models here to evaluate
        masked_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_bgra = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)
        self.correct_image(frame=frame_bgra, masked=masked_img)

    def correct_image(self, frame: np.ndarray,
                      masked: np.ndarray) -> np.ndarray:
        """
        use estimated size to correct the images shape and perspective

        :param frame: BGRA image
        :param masked: GRAYSCALE image
        """
        self.map_a, self.map_b = self.get_flattening_maps(masked)
        cv2.imwrite('map_b_vertical.png', self.map_b)
        cv2.imwrite('map_a_horizontal.png', self.map_a)
        flattened_image = cv2.remap(frame, self.map_a, self.map_b,
                                    interpolation=cv2.INTER_NEAREST,
                                    borderMode=cv2.BORDER_WRAP)

        gray = cv2.cvtColor(flattened_image, cv2.COLOR_BGRA2GRAY)
        flattened_image = cv2.cvtColor(flattened_image, cv2.COLOR_BGRA2BGR)
        _, mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY_INV)
        flattened_image = self.inpaint_img(flattened_image, mask)
        cv2.imwrite('flat_img.png', flattened_image)
        self.frame = flattened_image

    def get_flattening_maps(self, masked: np.ndarray):
        """
        Method to create 2 maps for flattening/remapping.

        :param masked: 2 dimensional numpy array where all except ROI is 0.
        :return: 2 arrays of same shape as masked, describing re-map values.
        """
        map_base = masked
        map_base_rotated = cv2.rotate(masked, cv2.ROTATE_90_COUNTERCLOCKWISE)

        edge_points = self.get_edge_points(map_base)
        edge_points = self.manage_outliers(edge_points)
        edge_points_rotated = self.get_edge_points(map_base_rotated)
        edge_points_rotated = self.manage_outliers(edge_points_rotated)

        pixel_map_a, masked = self.get_maps(True,
                                            edge_points,
                                            len(map_base[0]), masked,
                                            (255, 255, 0), (0, 0, 255))
        pixel_map_b, masked = self.get_maps(True,
                                            edge_points_rotated,
                                            len(map_base_rotated[0]), masked,
                                            (255, 0, 255), (0, 255, 0))
        cv2.imwrite('points.png', masked)

        pixel_map_a = np.array(pixel_map_a).astype(np.float32)
        pixel_map_b = cv2.rotate(np.array(pixel_map_b),
                                 cv2.ROTATE_90_CLOCKWISE).astype(np.float32)
        return pixel_map_a, pixel_map_b

    def get_edge_points(self, map_base: np.ndarray,
                        reverse: bool = False) -> list:
        """
        returns [list of (min/start, max/end) for each row.]
        indexes of where ROI of each row of the map start and end.
        Return as list of pairs for each row, starting with lowest value.
        """
        edge_points = []
        for pixel_row in map_base:
            roi = [idx_nr for idx_nr, pix in enumerate(pixel_row) if pix > 0]
            if len(roi) < 1:
                edge_points.append(edge_points[-1])
            else:
                edge_points.append((min(roi), max(roi)))
        if reverse:
            edge_points = [pair[::-1] for pair in edge_points]
        return edge_points

    @staticmethod
    def manage_outliers(edge_points: list):
        """
        Replace outlier points with RANSAC regressor and np.linspace.

        :param  edge_points: List of where  edges where detected
        :output: List where outlier edges have been replaced.
        """
        edges = [[edge_point[0] for edge_point in edge_points],
                 [edge_point[1] for edge_point in edge_points]]
        for point_index, points in enumerate(edges):
            first = points[:len(points)//3]
            second = points[len(points)//3:(len(points)//3)*2]
            third = points[(len(points)//3)*2:]
            x = np.linspace(0, len(points), len(points))
            X = x[:, np.newaxis]
            if np.median(first) < np.median(second) > np.median(third):
                model = make_pipeline(PolynomialFeatures(2), RANSACRegressor())
                model.fit(X, points)
                points = model.predict(X)
            elif np.median(first) > np.median(second) < np.median(third):
                model = make_pipeline(PolynomialFeatures(2), RANSACRegressor())
                model.fit(X, points)
                points = model.predict(X)
            else:
                points = np.linspace(np.mean(first),
                                     np.mean(third), len(points))
            edges[point_index] = points

        filtered = []
        for point_1, point_2 in zip(edges[0], edges[1]):
            filtered.append((point_1, point_2))
        return filtered

    def get_maps(self, first_value: bool,
                 pixel_pairs: list,
                 len_active: int, image: np.ndarray,
                 color_1: tuple, color_2: tuple):
        """
        generate the actual maps.
        TODO: make this take width/difference between numbers into account.
              for roi length index at the opposite direction.
        """
        pixel_map = []
        for row_idx, (start, stop) in enumerate(pixel_pairs):
            pixel_map.append((np.linspace(start, stop, len_active)))
            if first_value:
                location_start = (int(start), row_idx)
                location_stop = (int(stop), row_idx)
            else:
                location_start = (row_idx, int(start))
                location_stop = (row_idx, int(stop))

            masked = cv2.circle(image, location_start,
                                1, color_1, 1)
            masked = cv2.circle(masked, location_stop,
                                1, color_2, 1)
        return pixel_map, masked

    def inpaint_img(self, img, mask):
        img = cv2.inpaint(img, mask, 3, cv2.INPAINT_TELEA)
        return img
