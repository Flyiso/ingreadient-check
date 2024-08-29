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
        self.top = (0, 0, 255)
        self.bottom = (51, 153, 255)
        self.left = (255, 51, 153)
        self.right = (51, 255, 152)

    def add_image(self, masked_img: np.ndarray,
                  pre_remapping: np.ndarray, post_remapping: np.ndarray,
                  horizontal_map: np.ndarray, vertical_map: np.ndarray,
                  start_model_horizontal: Pipeline,
                  end_model_horizontal: Pipeline,
                  start_model_vertical: Pipeline,
                  end_model_vertical: Pipeline,
                  model_points: list, verbose: bool = True):
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
        if verbose:
            print('verbose=True\ndisplay results:')
            collage = cv2.resize(collage, (collage.shape[0]//3,
                                           collage.shape[1]//3))
            cv2.imshow('Prediction summary', collage)

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
        print('order of the detections are:')
        for predictions in model_points:
            if len(predictions) == len(img):
                print('vertical')
                length_horizontal.append(predictions[::-1])
            elif len(predictions) == len(img[0]):
                print('horizontal')
                length_vertical.append(predictions[::-1])
            else:
                print('no length match for')
                print(f'n_ predictions: {len(predictions)}')
                print(f'target_size: {img.shape}\n')

        img_lines = img.copy()
        for h, point_pair in enumerate(zip(length_vertical[0],
                                           length_vertical[1])):
            img_lines = cv2.circle(img_lines, (h, int(max(point_pair))),
                                   2, self.right, -1)
            img_lines = cv2.circle(img_lines, (h, int(min(point_pair))),
                                   2, self.left, -1)
        for w, point_pair in enumerate(zip(length_horizontal[0],
                                           length_horizontal[1])):
            img_lines = cv2.circle(img_lines, (int(max(point_pair)), w),
                                   2, self.bottom, -1)
            img_lines = cv2.circle(img_lines, (int(min(point_pair)), w),
                                   2, self.top, -1)
        return img_lines

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
        col_1 = np.vstack([pre_remapping, img_lines])
        col_2 = np.vstack([post_remapping, np.zeros_like(post_remapping)])
        col_3 = np.zeros_like(np.hstack([col_0, col_0]))
        collage = np.hstack([col_0, col_1, col_2, col_3])
        for idx, (model, color) in enumerate(zip(
            [start_model_horizontal, end_model_horizontal,
             start_model_vertical, end_model_vertical],
                [self.top, self.bottom, self.left, self.right]), 1):
            step_text = ([f'{s_name}, {step}' for
                          s_name, step in model.steps])
            text = ['Pipeline(steps=', *step_text]

            line_height = cv2.getTextSize("Text", cv2.FONT_HERSHEY_COMPLEX,
                                          1, 1)[0][1] + 10
            y = int(collage.shape[0] -
                    (((collage.shape[0]/2)//4)*idx)+line_height)
            x = int(collage.shape[1] - (col_0.shape[1]*3)+5)
            for i, line in enumerate(text):
                y_offset = y + i * line_height
                cv2.putText(collage, line, (x, y_offset),
                            cv2.FONT_HERSHEY_COMPLEX, 1,
                            color, 2, cv2.LINE_AA)
        return collage

    def save_images(self):
        """
        Save the created collages in specific folder
        """
        for index, image in enumerate(self.images, 1):
            cv2.imwrite(f'progress_images/evaluation_img_{index}.png', image)
