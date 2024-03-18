"""
Management of video feed
"""
import cv2
from image_management import ManageFrames


class VideoFeed:
    def __init__(self, video_path: str | int = 0,
                 interval: int = 50,
                 adjust_h: float = 1,
                 adjust_w: float = 1) -> None:
        """
        Initialise video capture. Manage video feed and
        frame management.

        parameters:
        video_path(str|int): path to the video. Default:0-current camera.
        interval(int): How often to process frame default: every 50 frame.
        adjust_h(float): size adjustment for frame height.
        adjust_w(float): size adjustment for frame width.
        """
        self.interval = interval
        self.adjust_h = adjust_h
        self.adjust_w = adjust_w
        self.video_path = video_path
        self.start_video()

    def start_video(self):
        frame_n = 0
        print('run...')
        self.capture = cv2.VideoCapture(self.video_path)
        while self.capture.isOpened():
            ret, frame = self.capture.read()
            if not ret:
                break

            elif frame_n == 0:
                frame_n = self.set_video_values(frame)
                frame = cv2.resize(frame, (self.width, self.height))
            else:
                frame = cv2.resize(frame, (self.width, self.height))
                frame = self.frame_manager.extract_roi(frame)
            cv2.imshow('frame', frame)
            if cv2.waitKey(25) & 0xFF == 27:
                break
        self.capture.release()
        cv2.destroyAllWindows()

    def set_video_values(self, frame):
        """
        calls methods and classes responsible for
        defining threshold values, frame sizes, for
        the video feed.
        returns 1 if successful
        returns 0 if unsuccessful.
        parameters:
        frame: the current frame of the feed.
        """
        self.height = int(frame.shape[0]*self.adjust_h)
        self.width = int(frame.shape[1]*self.adjust_w)
        frame = cv2.resize(frame, (self.width, self.height))
        self.frame_manager = ManageFrames()
        self.frame_manager.set_manager_values(frame)
        return 1
