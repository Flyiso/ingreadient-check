"""
Responsible for creating a readable panorama image
from content saved by label reader.
"""
import cv2
import pytesseract as pt


class CreatePanorama:
    def __init__(self) -> None:
        self.stitcher = cv2.Stitcher.create()
        self.results = []

    def add_images(self, panorama_images: list):
        image_stitched = self.stitch(panorama_images)
        self.results.append(image_stitched)
        return image_stitched

    def stitch(self, panorama_images):
        print(len(panorama_images))
        status, result = self.stitcher.stitch(panorama_images)
        if status != cv2.STITCHER_OK:
            print('Stitcher failed')
            return
        result = self.draw_on_image(result)
        return result

    def draw_on_image(self, img):
        """
        uses tessreact and cv2 boxes to find words.
        """
        data = pt.image_to_data(img, config='--psm 12',
                                output_type='data.frame',
                                lang='swe')
        for index, row in data.iterrows():
            x, y, w, h = row['left'], row['top'], row['width'], row['height']
            color = (255, 0, 0)  # default
            thickness = 1
            cv2.rectangle(img, (x, y), (x + w, y + h),
                          color, thickness)
            cv2.putText(img, str(index), (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (255, 0, 0), 2)
        return img
