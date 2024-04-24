"""
File to test different ways to make merging of images work better.
"""
import cv2
from image_manager import ManageFrames
from panorama_manager import ManagePanorama


class RecordLabel:
    def __init__(self, video_path: str | int = 0,
                 interval: int = 10,
                 adjust_h: float = 1,
                 adjust_w: float = 1,
                 config: str = '--oem 3 --psm 6',
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
        self.config = config
        self.panorama_manager = ManagePanorama(
            ManageFrames(self.config), interval)
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
        self.frame_n = 0
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
            if cv2.waitKey(25) & 0xFF == 27:
                break

        self.capture.release()
        cv2.destroyAllWindows()
        self.save_image('merge_result', self.panorama_manager.base)

    def process_image(self, frame):
        """
        Apply correction, enhancement
        and detection methods on frame
        """
        frame = cv2.resize(frame, (self.width, self.height))
        self.panorama_manager.add_frame(frame)
        self.frame_n = len(self.panorama_manager.frames)
        return frame

    def end_video(self):
        """
        final cleanup.
        """
        print('DONE!')
        print(f'merged: {self.panorama_manager.merge_counter}\
              failed: {self.panorama_manager.fail_counter}')
        print(f'total frames: {len(self.panorama_manager.frames)}\
              interval: {self.panorama_manager.interval}')

    def set_video_values(self, frame):
        """
        get data from image to uniform future
        processing methods.
        """
        self.height = int(frame.shape[0]*self.adjust_h)
        self.width = int(frame.shape[1]*self.adjust_w)

    def save_image(self, filename: str, frame):
        merges = self.panorama_manager.merge_counter
        cv2.imwrite(
            f'{self.img_dir}-{filename}_{merges}.png',
            frame)
