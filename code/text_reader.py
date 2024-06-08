"""
Classes and functions responsible for extracting the
correct text out of an image.
"""
import numpy as np
import pytesseract as pt
import cv2


class ReadText:
    """
    Uses (currently) pyTesseract to get the text
    out of the image passed to it. This will later on
    be updated to select the text we are interested in,
    (ingredient list of target language).
    """
    def __init__(self, image: np.ndarray,
                 pytesseract_config: str) -> None:
        self.image = image
        self.config = pytesseract_config
        self.text_data = self._detect_text()

    def _detect_text(self):
        data = pt.image_to_data(image=self.image,
                                config=self.config,
                                output_type='dict')
        return data

    def draw_detected_text(self):
        for i in range(len(self.text_data['text'])):
            if int(self.text_data['conf'][i]) > 0:
                (x, y, w, h) = (self.text_data['left'][i],
                                self.text_data['top'][i],
                                self.text_data['width'][i],
                                self.text_data['height'][i])
                text = self.text_data['text'][i]
                cv2.rectangle(self.image, (x, y), (x + w, y + h),
                              (0, 255, 0), 1)
                cv2.putText(self.image, text, (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.4, (0, 255, 0), 1)
        return self.image
