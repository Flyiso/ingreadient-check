"""
uses images to try to create a panorama of all.
"""
import cv2


class ManagePanorama:
    """
    class responsible for building panorama image of
    output frames.
    """
    def __init__(self,
                 interval: int = 10,
                 merge_size: int = 4,) -> None:
        """
        parameters:
            interval(int): distance in frames between every merge default: 10 
            merge_size(int): How many frames to merge to panorama img.
        """
        self.interval = interval
        self.merge_size = merge_size
        self.image_management = None
        self.frames = []
        self.base = None
        self.merge_counter = 0
        self.fail_counter = 0
        self.stitcher = cv2.Stitcher_create(mode=1)

    def add_frame(self, frame) -> bool:
        """
        Adds a new frame to the mergers frames.
        Tries self.interval frames before if unsuccessful.
        Returns True for successful merges, returns false else.
        """
        self.frames.append(frame)
        if len(self.frames) % self.interval == 0 or self.base is None:
            for x in range(1, self.interval+1):
                if self.stitch_frames(self.frames[-x]):
                    return True
                if len(self.frames) == 2:
                    self.base = self.frames[1]
                    self.frames.pop(0)
                    return True
                self.frames.pop(-1)
        return False

    def stitch_frames(self, frame) -> bool:
        """
        Attempts to merge frame to panorama.
        returns True if successful,  returns False if not.
        """
        # make sure there is a frame base
        if self.base is None:
            self.base = frame
            return True
        # stitch:
        status, result = self.stitcher.stitch([self.base, frame])
        if status == cv2.STITCHER_OK:
            self.base = result
            self.merge_counter += 1
            return True
        self.fail_counter += 1
        return False
