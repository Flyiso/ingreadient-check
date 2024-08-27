"""
Class that evaluate and create an image for easy model evaluation.
"""
import numpy as np
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

    def add_image(self, masked_img: np.ndarray,
                  pre_remapping: np.ndarray, post_remapping: np.ndarray,
                  horizontal_map: np.ndarray, vertical_map: np.ndarray,
                  start_model_horizontal: Pipeline,
                  end_model_horizontal: Pipeline,
                  start_model_vertical: Pipeline,
                  end_model_vertical: Pipeline):
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
        """
        img_w_lines = self.draw_evaluations(pre_remapping,
                                            start_model_horizontal,
                                            end_model_horizontal,
                                            start_model_vertical,
                                            end_model_vertical)
        collage = self.create_collage(masked_img, img_w_lines,
                                      pre_remapping, post_remapping,
                                      horizontal_map, vertical_map,
                                      start_model_horizontal,
                                      end_model_horizontal,
                                      start_model_vertical,
                                      end_model_vertical)
        self.images.append(collage)

    @staticmethod
    def draw_evaluations(img: np.ndarray,
                         start_model_horizontal: Pipeline,
                         end_model_horizontal: Pipeline,
                         start_model_vertical: Pipeline,
                         end_model_vertical: Pipeline) -> np.ndarray:
        """
        Predict and mark predictions on an image.

        :param img:
        Numpy array to draw the detected boundaries on.
        :param start_model_horizontal:
        The model used when evaluating where ROI starts.
        :param end_model_horizontal:
        The model used when evaluating where ROI ends.
        :param start_model_vertical:
        The model used when evaluating where ROI starts.
        (this is for the image rotated)
        :param end_model_vertical:
        The model used when evaluating where ROI ends.
        (this is for the image rotated)
        :return:
        the input image, with predicted edges drawn.
        The model for start_horizontal is Orange.
        The model for end_horizontal is Red.
        The model for start_vertical is Purple.
        The model for end_vertical is Green
        """
        pass

    @staticmethod
    def create_collage(masked_img: np.ndarray, img_lines: np.ndarray,
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
        pass
