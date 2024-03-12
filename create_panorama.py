"""
Responsible for creating a readable panorama image
from content saved by label reader.
"""
import cv2


class CreatePanorama:
    def __init__(self, pan_group: int = 3) -> None:
        self.stitcher = cv2.Stitcher.create()
        self.waiting = []
        self.stitched = []
        self.all = []

    def add_image(self, panorama_image):
        """
        activates merge for images.
        """
        self.waiting.append(panorama_image)
        self.all.append(panorama_image)
        print(len(self.waiting))
        if len(self.waiting) >= 3:
            self.stitch()

    def final_merge(self, final_img=None):
        """
        merges final frames.
        """
        print(len(self.stitched))
        if final_img is not None:
            self.waiting.append(final_img)

        if len(self.waiting) > 1:
            self.stitch()

    def stitch(self):
        """
        stitches 2 images together and adds result to
        self.stitched
        """
        status, result = self.stitcher.stitch(self.waiting)
        if status != cv2.STITCHER_OK:
            print('Stitcher failed')
            self.success = False
            print('fail...')
        self.stitched.append(result)
        print('stitcher success!')
        self.success = True
        self.waiting = []
        print(len(self.waiting))
