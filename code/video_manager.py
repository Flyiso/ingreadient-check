"""
File to test different ways to make merging of images work better.
"""
import cv2
import numpy as np
from image_manager import ManageFrames
from panorama_manager import ManagePanorama


class RecordLabel:
    def __init__(self, video_path: str | int = 0,
                 adjust_h: float = 1,
                 adjust_w: float = 1,
                 pt_config: str = '--oem 3 --psm 6',
                 img_dir: str = 'default',
                 display_current: bool = True) -> None:
        """
        Initialise video capture. Manage video feed and
        frame management.

        parameters:
        video_path(str|int): path to the video. Default:0-current camera.
        interval(int): How often to process frame default: every 50 frame.
        adjust_h(float): size adjustment for frame height.
        adjust_w(float): size adjustment for frame width.
        config(str): configuration str for pytesseract text recognition.
        img_dir(str): directory to store images in.
        display_current(bool): if the panorama manager should merge and return
                               result every time the interval is met.
        """
        self.pt_config = pt_config
        self.frame_manager = ManageFrames(self.pt_config)
        self.panorama_manager = ManagePanorama(self.frame_manager,
                                               display_current=display_current)
        self.adjust_h = adjust_h
        self.adjust_w = adjust_w
        self.video_path = video_path
        self.img_dir = f'outputs/{img_dir}/'
        self.display_current = display_current
        self.current_merge = False
        self.start_video()

    def start_video(self):
        """
        loops through images and
        defines how to preprocess them
        """
        last_frame = None
        self.frame_n = 0
        self.capture = cv2.VideoCapture(self.video_path)
        while self.capture.isOpened():
            ret, frame = self.capture.read()

            if not ret:
                self.merge_loops = 0
                self.end_video(last_frame)
                break

            if self.frame_n == 0:
                self.set_video_values(frame)

            frame, is_blurry, merged = self.process_image(frame)

            cv2.imshow('frame', frame)
            if isinstance(merged, np.ndarray):
                cv2.imshow('merged', merged)

            if not is_blurry:
                last_frame = frame
                self.frame_n += 1
            if cv2.waitKey(25) & 0xFF == 27:
                break

        self.capture.release()
        cv2.destroyAllWindows()
        self.save_image('merge_result', self.panorama_manager.base)
        self.final_image = self.panorama_manager.base

    def process_image(self, frame: np.ndarray) -> tuple[np.ndarray, bool,
                                                        np.ndarray | bool]:
        """
        Apply correction, enhancement
        and detection methods on frame
        sorts out frames that are blurry and does not send them to the
        panorama manager
        """
        frame = cv2.resize(frame, (self.width, self.height))
        if not self.frame_manager.is_blurry(frame):
            self.current_merge = self.panorama_manager.add_frame(frame)
        else:
            print('blurry frame removed')
            self.save_image('blurry_frame', frame)
        return frame, self.frame_manager.is_blurry(frame), self.current_merge

    def end_video(self, last_frame: bool | np.ndarray):
        """
        final cleanup.
        """
        if isinstance(last_frame, np.ndarray):
            self.panorama_manager.add_frame(last_frame, last_frame=True)
        final_image, merge_status = self.panorama_manager.get_final_panorama()
        if isinstance(final_image, np.ndarray):
            self.save_image('Merged_result', final_image)
            cv2.imshow('current_merge', final_image)
            input('EXIT?')
        if merge_status:
            print('Merging succeeded.')

    def set_video_values(self, frame: np.ndarray):
        """
        get data from image to uniform future
        processing methods.
        """
        self.height = int(frame.shape[0]*self.adjust_h)
        self.width = int(frame.shape[1]*self.adjust_w)

    def save_image(self, filename: str, frame: np.ndarray):
        cv2.imwrite(
            f'{self.img_dir}{filename}.png', frame)
