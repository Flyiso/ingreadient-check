"""
Class that evaluate and create an image for easy model evaluation.
"""
import numpy as np
import cv2
from sklearn.pipeline import Pipeline


class EvaluationImages:
    """
    Temporary class for evaluation purposes.
    Class to collect and display data on frame map creation.
    For each image used in remapping, the class collect information
    on maps used, ROI img, img and the models used. That information
    is then displayed to simplify model evaluation.
    """
    def __init__(self) -> None:
        self.images = []
        self.top = (255, 0, 0)
        self.bottom = (255, 153, 51)
        self.left = (153, 51, 255)
        self.right = (152, 255, 51)

    def add_image(self, masked_img: np.ndarray,
                  pre_remapping: np.ndarray, post_remapping: np.ndarray,
                  horizontal_map: np.ndarray, vertical_map: np.ndarray,
                  start_model_horizontal: Pipeline,
                  end_model_horizontal: Pipeline,
                  start_model_vertical: Pipeline,
                  end_model_vertical: Pipeline,
                  model_points: list):
        """
        Use the re-mapping information to create evaluation image.

        :param masked_img: Masked version of the image being remapped,
        where non roi sections are black.
        :param pre_remapping: original image
        :param post_remapping: Image after remapping.
        :param horizontal_map: Image visualizing the remapping
        in horizontal direction. Grayscale.
        :param vertical_map: Image visualizing the remapping
        in vertical direction. Grayscale.
        :param start_model_horizontal: The model used when
        evaluating where ROI starts.
        :param end_model_horizontal: The model used when
        evaluating where ROI ends.
        :param start_model_vertical: The model used when
        evaluating where ROI starts. (this is for the image rotated)
        :param end_model_vertical: The model used when
        evaluating where ROI ends. (this is for the image rotated)
        :param model_points: List of list, where all predictions are.
        """
        img_w_lines = self.draw_evaluations(pre_remapping, model_points)
        collage = self.create_collage(masked_img, img_w_lines,
                                      pre_remapping, post_remapping,
                                      horizontal_map, vertical_map,
                                      start_model_horizontal,
                                      end_model_horizontal,
                                      start_model_vertical,
                                      end_model_vertical)
        self.images.append(collage)

    def draw_evaluations(self, img: np.ndarray,
                         model_points: list) -> np.ndarray:
        """
        Predict and mark predictions on an image.

        :param img:
        Numpy array to draw the detected boundaries on.
        :param model_points: List of list, where all predictions are.
        :return: The input image, with predicted edges drawn.
        """
        length_vertical = []
        length_horizontal = []
        for predictions in model_points:
            if len(predictions) == len(img):
                length_horizontal.append(predictions)
            elif len(predictions) == len(img[0]):
                length_vertical.append(predictions)
            else:
                print('no length match for')
                print(f'n_ predictions: {len(predictions)}')
                print(f'target_size: {img.shape}\n')

        for h, point_pair in enumerate(zip(length_vertical[0],
                                           length_vertical[1])):
            # green
            img = cv2.circle(img, (h, max(point_pair)), 2, self.right, -1)
            img = cv2.circle(img, (h, min(point_pair)), 2, self.left, -1)
        for w, point_pair in enumerate(zip(length_horizontal[0],
                                           length_horizontal[1])):
            img = cv2.circle(img, (max(point_pair), w), 2, self.bottom, -1)
            img = cv2.circle(img, (min(point_pair), w), 2, self.top, -1)
        return img

    def create_collage(self, masked_img: np.ndarray, img_lines: np.ndarray,
                       pre_remapping: np.ndarray, post_remapping: np.ndarray,
                       map_horizontal: np.ndarray, map_vertical: np.ndarray,
                       start_model_horizontal: Pipeline,
                       end_model_horizontal: Pipeline,
                       start_model_vertical: Pipeline,
                       end_model_vertical: Pipeline) -> np.ndarray:
        """
        Create image displaying all relevant information.

        :param masked_img: Masked version of the image being remapped,
        where non roi sections are black.
        :param img_lines: Image where predicted edges are drawn
        :param pre_remapping: Original image
        :param post_remapping: Image after remapping.
        :param horizontal_map: Image visualizing
        the remapping in horizontal direction. Grayscale.
        :param vertical_map: Image visualizing
        the remapping in vertical direction. Grayscale.
        :param start_model_horizontal: The model used
        when evaluating where ROI starts.
        :param end_model_horizontal: The model used
        when evaluating where ROI ends.
        :param start_model_vertical: The model used
        when evaluating where ROI starts. (this is for the image rotated)
        :param end_model_vertical: The model used
        when evaluating where ROI ends.(this is for the image rotated)
        :return: image of input information displayed.
        """

        col_0 = cv2.cvtColor(np.vstack([map_horizontal,
                                        map_vertical]), cv2.COLOR_GRAY2BGR)    
        col_1 = np.vstack([pre_remapping, post_remapping])
        col_2 = np.vstack([masked_img, img_lines])
        col_3 = np.zeros_like(np.hstack(col_0, col_0))
        collage = np.hstack([col_0, col_1, col_2, col_3])
        for idx, (model, color) in enumerate(zip(
            [start_model_horizontal, end_model_horizontal,
             start_model_vertical, end_model_vertical],
                [self.left, self.right, self.top, self.bottom]), 1):
            text = f'Model for edges on the left:\n{model}'
            collage = cv2.putText(collage,
                                  text, (collage.shape[1] - 
                                         sum(col_0.shape[1], col_0.shape[1]),
                                         collage.shape[0]/idx),
                                  cv2.FONT_HERSHEY_COMPLEX, 3.0, color, 3)
        return collage

    def save_images(self):
        """
        Save the created collages in specific folder
        """
        for index, image in enumerate(self.images, 1):
            cv2.imwrite(f'progress_images/evaluation_img_{index}.png', image)
