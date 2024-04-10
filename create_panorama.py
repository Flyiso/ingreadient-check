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
        self.stitcher = cv2.Stitcher_create(mode=1)

        #self.stitcher.setWaveCorrection
        self.masks = []
        self.waiting = []
        self.stitched = []

    def add_image(self, panorama_image,
                  mask=None):
        """
        activates merge for images.
        add (optional)masks to specify roi.
        """
        self.success = False
        self.masks.append(mask)
        self.waiting.append(panorama_image)
        print(f'n-images merging: {len(self.waiting)}')
        if len(self.waiting) >= self.merge_size:
            self.stitch()
        if self.success is True:
            return self.stitched[-1]
        return panorama_image

    def add_images(self, frames: list,
                   masks: list = None):
        """
        Merge together all input frames.
        """
        mem = self.waiting
        self.waiting = [frame for frame in frames
                        if len(frame.shape) == 3]
        panorama = self.stitch()
        self.waiting = mem
        if self.success:
            panorama = self.stitched[-1]
            self.stitched.pop(-1)
            return panorama

    def final_merge(self, final_img=None) -> list:
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
            return self.stitched
        return final_img

    def stitch(self):
        """
        stitches 2 images together and adds result to
        self.stitched
        """
        for val in self.waiting:
            print(val.shape)
        status, result = self.stitcher.stitch(self.waiting, self.masks)
        if status != cv2.STITCHER_OK:
            print('Stitcher failed\n\n')
            return
        self.stitched.append(result)
        print(result.shape)
        print('stitcher success!\n\n')
        self.success = True
        self.waiting = []
        self.masks = []
