"""
Responsible for creating a readable panorama image
from content saved by label reader.
"""
import cv2


class ManagePanorama:
    """
    class responsible for building panorama image of
    output frames.
    """
    def __init__(self, pan_group: int = 3) -> None:
        self.merge_size = pan_group
        self.stitcher = cv2.Stitcher.create()
        self.waiting = []
        self.stitched = []
        self.all = []

    def add_image(self, panorama_image):
        """
        activates merge for images.
        """
        self.success = False
        self.waiting.append(panorama_image)
        self.all.append(panorama_image)
        print(len(self.waiting))
        if len(self.waiting) >= self.merge_size:
            self.stitch()
        if self.success is True:
            return self.stitched[-1]
        return panorama_image

    def final_merge(self, final_img=None):
        """
        merges final frames
        forces merge if  frames unmerged
        """
        self.success = False
        if final_img is not None:
            self.waiting.append(final_img)
        if len(self.waiting) > 1:
            self.stitch()
        if self.success:
            return self.stitched[-1]
        return final_img

    def stitch(self):
        """
        stitches 2 images together and adds result to
        self.stitched
        """
        status, result = self.stitcher.stitch(self.waiting)
        if status != cv2.STITCHER_OK:
            print('Stitcher failed')
            self.success = False
            return
        self.stitched.append(result)
        print('stitcher success!')
        self.success = True
        self.waiting = []
