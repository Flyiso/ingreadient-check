"""
File to test different ways to make merging of images work better.
"""
import cv2
from image_management import ManageFrames
from create_panorama import ManagePanorama
import numpy as np


class FromVideo:
    def __init__(self, video_path: str | int = 0,
                 interval: int = 15,
                 merge_size: int = 2,
                 adjust_h: float = 1,
                 adjust_w: float = 1,
                 config: str = '--oem 3 --psm 6') -> None:
        """
        Initialise video capture. Manage video feed and
        frame management.

        parameters:
        video_path(str|int): path to the video. Default:0-current camera.
        interval(int): How often to process frame default: every 50 frame.
        merge_size(int): How many frames to merge to panorama img.
        adjust_h(float): size adjustment for frame height.
        adjust_w(float): size adjustment for frame width.
        """
        self.interval = interval
        self.merge_size = merge_size
        self.adjust_h = adjust_h
        self.adjust_w = adjust_w
        self.video_path = video_path
        self.config = config
        self.start_video()

    def start_video(self):
        """
        loops through images and
        defines how to preprocess them
        """
        self.frame_n = 0
        last_frame = None
        self.next_interval = self.interval
        self.panoramas = []
        self.final_frame = False
        self.capture = cv2.VideoCapture(self.video_path)
        while self.capture.isOpened():
            ret, frame = self.capture.read()

            if not ret:
                self.merge_loops = 0
                self.end_video()
                break

            if self.frame_n == 0:
                self.set_video_values(frame)

            frame = self.process_image(frame)
            self.frame_n += 1

            cv2.imshow('frame', frame)
            last_frame = frame
            if cv2.waitKey(25) & 0xFF == 27:
                break

        self.capture.release()
        cv2.destroyAllWindows()
        if isinstance(self.final_frame, np.ndarray):
            data = self.frame_manager.find_text(self.final_frame)
            print(data['text'])

    def process_image(self, frame):
        """
        Apply correction, enhancement
        and detection methods on frame
        """
        frame = cv2.resize(frame, (self.width, self.height))
        frame = self.frame_manager.enhance_text_lightness(frame)
        frame = self.frame_manager.rotate_by_lines(frame)
        frame = self.frame_manager.extract_roi(frame)
        frame, contour = self.frame_manager.crop_to_four_corners(frame)
        frame = self.frame_manager.stretch_image(frame, contour,
                                                 (self.width,
                                                  self.height))
        if self.next_interval != 0:
            self.next_interval -= 1
            return frame
        frame_stitched = self.panorama_manager.add_image(frame)
        if self.panorama_manager.success:
            self.save_image(f'stitched_{self.frame_n}',
                            frame_stitched)
            self.panoramas.append(frame_stitched)
            self.next_interval = self.interval
        return frame

    def end_video(self):
        """
        merges all currently merged images together
        """
        merged = []
        merged_n = 0
        new_merge_size = self.calculate_merge_size()
        self.panorama_manager = ManagePanorama(new_merge_size)
        for frame in self.panoramas:
            frame = self.frame_manager.rotate_by_lines(frame)
            frame = self.frame_manager.extract_roi(frame)
            frame, contour = self.frame_manager.crop_to_four_corners(frame)
            frame = self.frame_manager.stretch_image(frame, contour,
                                                     (self.width,
                                                      self.height))
            frame_merged = self.panorama_manager.add_image(frame)
            if self.panorama_manager.success:
                merged_n += 1
                self.save_image(
                    f'double_merged_{merged_n}_{self.merge_loops}_m_size-{new_merge_size}',
                                frame)
                merged.append(frame_merged)
        self.panoramas = merged
        if len(self.panoramas) >= 2:
            self.merge_loops += 1
            self.end_video()

    def calculate_merge_size(self):
        """
        increases or decreases limit for merge size
        depending on number of saved merged frames.
        """
        dividers = [n if (len(self.panoramas) % n == 0) else None
                    for n in range(1, len(self.panoramas)+1)]
        divider = min([divider for divider in dividers if divider is not None],
                      key=lambda x: abs(x - self.merge_size))
        return divider

    def set_video_values(self, frame):
        """
        get data from image to uniform future
        processing methods.
        """
        self.height = int(frame.shape[0]*self.adjust_h)
        self.width = int(frame.shape[1]*self.adjust_w)
        frame = cv2.resize(frame, (self.width, self.height))
        self.frame_manager = ManageFrames(self.config)
        self.panorama_manager = ManagePanorama(self.merge_size)
        self.frame_manager.set_manager_values(frame)
        return 1

    def save_image(self, filename: str, frame):
        cv2.imwrite(
            f'outputs/from_v/{self.merge_size}-{self.interval}_{filename}.png',
            frame)
        print(f'frame- saved shape: {frame.shape}')
