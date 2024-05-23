"""
File to test different ways to make merging of images work better.
"""
import cv2
import numpy as np
from image_manager import ManageFrames
from panorama_manager import ManagePanorama


class RecordLabel:
    def __init__(self, video_path: str | int = 0,
                 interval: int = 10,
                 adjust_h: float = 1,
                 adjust_w: float = 1,
                 pt_config: str = '--oem 3 --psm 6',
                 img_dir: str = 'default') -> None:
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
        """
        self.pt_config = pt_config
        panorama_manager = ManageFrames(self.pt_config)
        self.panorama_manager = ManagePanorama(panorama_manager, interval)
        self.adjust_h = adjust_h
        self.adjust_w = adjust_w
        self.video_path = video_path
        self.img_dir = f'outputs/{img_dir}/'
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

            frame, is_blurry = self.process_image(frame)
            

            cv2.imshow('frame', frame)
            if not is_blurry:
                last_frame = frame
                self.frame_n += 1
            if cv2.waitKey(25) & 0xFF == 27:
                break

        self.capture.release()
        cv2.destroyAllWindows()
        self.save_image('merge_result', self.panorama_manager.base)

    def process_image(self, frame: np.ndarray):
        """
        Apply correction, enhancement
        and detection methods on frame
        sorts out frames that are blurry and does not send them to the
        panorama manager
        """
        frame = cv2.resize(frame, (self.width, self.height))
        if not self.is_blurry(frame):
            self.panorama_manager.add_frame(frame)
            self.frame_n = len(self.panorama_manager.frames)
        else:
            print('blurry frame removed')
            self.save_image('blurry_frame', frame)
        return frame, self.is_blurry(frame)

    def is_blurry(self, frame: np.ndarray,
                  threshold: float = 250.00) -> bool:
        """
        Uses cv2 Laplacian to sort out images where
        not enough edges are detected.
        Returns True if image blur meet threshold.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        return laplacian_var < threshold

    def end_video(self, last_frame: bool | np.ndarray):
        """
        final cleanup.
        """
        if last_frame is not False:
            self.panorama_manager.add_more_frames(last_frame)
        print('DONE!')
        print(f'merged: {self.panorama_manager.merge_counter}\
              failed: {self.panorama_manager.fail_counter}')
        print(f'total frames: {len(self.panorama_manager.frames)}\
              interval: {self.panorama_manager.interval}')
        last_frame = self.panorama_manager.detect_text()
        for idx_nr, frame in enumerate(self.panorama_manager.to_stitch):
            self.save_image(f'merge_queue_{idx_nr}', frame)
        self.save_image('Merged_result', last_frame)

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
