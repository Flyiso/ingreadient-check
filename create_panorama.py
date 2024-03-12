"""
Responsible for creating a readable panorama image
from content saved by label reader.
"""
import cv2


class CreatePanorama:
    def __init__(self) -> None:
        self.stitcher = cv2.Stitcher.create()
        self.waiting = []
        self.stitched = []
        self.final_stitch_queue = []

    def add_images(self, panorama_image):
        """
        activates merge for images in pairs.
        """
        self.waiting.append(panorama_image)
        if len(self.waiting) >= 2:
            self.stitch()

    def stitch(self):
        """
        stitches 2 images together and adds result to
        self.stitched
        """
        status, result = self.stitcher.stitch(self.waiting)
        if status != cv2.STITCHER_OK:
            print('Stitcher failed')
        self.stitched.append(result)
        self.waiting = []

    def final_merge(self, frame=None):
        """
        Merge the final images together to one image.
        """
        if frame is not None:
            self.add_final(frame)

        while len(self.final_stitch_queue) != 0 and len(self.stitched) != 1:
            self.waiting = []
            self.final_stitch_queue = self.stitched
            self.stitched = []
            while self.final_stitch_queue > 1:
                self.waiting.append(self.final_stitch_queue.pop(0))
                self.add_images(self.final_stitch_queue.pop(0))
            if len(self.final_stitch_queue) == 1:
                self.stitched.append(self.final_stitch_queue.pop(-1))

            print(f'final_queue: {len(self.final_stitch_queue)}')
            print(f'stitched: {len(self.stitched)}')
            print(f'waiting: {len(self.waiting)}')

    def add_final(self, frame):
        """
        Adds the final frame to the merge lists
        """
        if self.waiting != []:
            self.add_images(frame)
        elif len(self.stitched) % 2 == 0:
            self.waiting.append(self.stitched.pop(-1))
            self.add_images(frame)
        else:
            self.stitched.append(frame)
