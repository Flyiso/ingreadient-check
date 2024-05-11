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
                 frame_manager,
                 interval: int = 10) -> None:
        """
        parameters:
            interval(int): distance in frames between every merge default: 10
        """
        self.interval = interval
        self.image_management = None
        self.frames = []
        self.base = False
        self.merge_counter = 0
        self.fail_counter = 0
        self.stitcher = cv2.Stitcher.create(1)
        self.frame_manager = frame_manager
        self.max_merge = 10  # CONNECTED TO TEST OF NEW MERGE FLOW

    def add_frame(self, frame) -> bool:
        """
        Adds a new frame to the mergers frames.
        Tries self.interval frames before if unsuccessful.
        Returns True for successful merges, returns false else.
        """
        self.frames.append(frame)
        if len(self.frames) % self.interval == 0 or self.base is False:
            return self.add_more_frames(frame)
        return False

    def add_more_frames(self, frame) -> bool:
        """
        Modified version of frame stitching to test if
        more frames solve stitcher error 1 problems
        """
        if self.base is False:
            self.base = self.frame_manager.find_label(frame)
            print('New Merge: First frame')
            self.frame_manager.set_manager_values(self.base)
            return True
        to_stitch = [self.base]
        status = None
        for frame in self.frames[::-1]:
            frame = self.frame_manager.find_label(frame)
            if frame is False:
                continue
            to_stitch.append(frame)
            status, result = self.stitcher.stitch(to_stitch)
            if status == cv2.Stitcher_OK:
                print('New Merge: Success')
                self.base = result
                self.merge_counter += 1
                cv2.imwrite('merged.png', result)
                return True
            print('New Merge: Failed')
            if len(to_stitch) > self.max_merge:
                print('maximum merge reached.')
                return False
