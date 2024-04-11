"""
File to test different ways to make merging of images work better.
"""
import cv2
from image_manipulation import ManageFrames
from create_panorama import ManagePanorama
import numpy as np


class FromVideo:
    def __init__(self, video_path: str | int = 0,
                 interval: int = 10,
                 frame_space: int = 5,
                 merge_size: int = 4,
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
        self.frame_space = frame_space
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

            if len(self.panoramas) > 2:
                self.panoramas = self.merge_merged()

            cv2.imshow('frame', frame)
            last_frame = frame
            if cv2.waitKey(25) & 0xFF == 27:
                break

        self.capture.release()
        cv2.destroyAllWindows()
        if isinstance(self.final_frame, np.ndarray):
            data = self.frame_manager.find_text(self.final_frame)
            print(data['text'])
        print(self.frame_n)

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

        if self.frame_n % self.frame_space != 0:
            return frame
        if self.next_interval != 0:
            self.next_interval -= 1
            return frame
        mask = self.frame_manager.get_roi_mask(frame)
        frame_stitched = self.panorama_manager.add_image(frame, mask)

        if self.panorama_manager.success:
            frame_stitched = self.process_output(frame_stitched)
            self.save_image(f'stitched_{self.frame_n}',
                            frame_stitched)
            self.panoramas.append(frame_stitched)
            self.next_interval = self.interval
            self.get_text_mask(frame_stitched)
        return frame

    def process_output(self, merged_frame):
        """
        attempts to extract the ROI from merged panorama and
        tries to correct rotation.
        """
        merged_txt = self.frame_manager.find_text(merged_frame)
        for i in range(len(merged_txt['text'])):
            if int(merged_txt['conf'][i]) > 20:
                x, y, w, h = merged_txt['left'][i], merged_txt['top'][i], merged_txt['width'][i], merged_txt['height'][i]
                cv2.rectangle(merged_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        merged_frame = self.frame_manager.rotate_by_lines(merged_frame)
        merged_frame = self.frame_manager.extract_roi(merged_frame)
        merged_frame, contour = \
            self.frame_manager.crop_to_four_corners(merged_frame)
        merged_frame = \
            self.frame_manager.stretch_image(merged_frame,
                                             contour,
                                             (merged_frame.shape[1],
                                              merged_frame.shape[0]))
        return merged_frame

    def merge_merged(self) -> list:
        """
        merges 2 or more frames together.
        """
        merge_obj = ManagePanorama(len(self.panoramas))
        for panorama in self.panoramas:
            mask = self.frame_manager.hugh_lines_mask(panorama)
            panorama = merge_obj.add_image(panorama, mask)
            if merge_obj.success:
                print('merged!')
                self.save_image(
                    f'pan_{self.frame_n}_{len(self.panoramas)}',
                    panorama)
                self.panoramas = [panorama]
        return self.panoramas

    def end_video(self):
        """
        merges all currently merged images together
        """
        merged = []
        merged_n = 0
        print(f'\nframes to merge: {len(self.panoramas)}')
        new_merge_size = self.calculate_merge_size()
        self.panorama_manager = ManagePanorama(new_merge_size)
        for frame_idx, frame in enumerate(self.panoramas):
            frame = self.frame_manager.extract_roi(frame)
            frame, contour = self.frame_manager.crop_to_four_corners(frame)
            frame = self.frame_manager.rotate_by_lines(frame)
            frame = self.frame_manager.stretch_image(frame, contour,
                                                     (self.width,
                                                      self.height))
            frame = self.frame_manager.crop_image(frame, contour)
            mask = self.frame_manager.get_roi_mask(frame)
            frame_merged = self.panorama_manager.add_image(frame, mask)
            if self.panorama_manager.success:
                merged_n += 1
                self.save_image(
                    f'double_merged_{merged_n}_{self.merge_loops}_m_size-{new_merge_size}',
                    frame)
                merged.append(frame_merged)
            #frame = self.frame_manager.draw_detect_keypoints(frame)
            self.save_image(f'keypoints_{frame_idx}', frame)

        self.panoramas = merged
        if len(self.panoramas) >= 2:
            self.merge_loops += 1
            print(f'\n\nNEW MERGE LOOP: {self.merge_loops}\n\n')
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
        # frame = cv2.resize(frame, (self.width, self.height))
        self.frame_manager = ManageFrames(self.config)
        self.panorama_manager = ManagePanorama(self.merge_size)
        self.frame_manager.set_manager_values(frame)
        return 1

    def get_text_mask(self, frame):
        """
        Draws lines, keypoints, and other attributes on
        frame to help figure out how to best manage merged frames.
        This method is to be deleted later on.
        """
        # do stuff here
        mask = self.frame_manager.hugh_lines_mask(frame)
        #self.save_image(f'text_mask_{self.frame_n}',
        #                mask)
        return mask

    def save_image(self, filename: str, frame):
        cv2.imwrite(
            f'outputs/from_v/{self.merge_size}-{self.interval}_{filename}.png',
            frame)
        print(f'frame- saved shape: {frame.shape}\nsaved name: {filename}\n')
