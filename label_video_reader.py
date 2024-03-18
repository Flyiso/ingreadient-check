"""
Management of video feed
"""
import cv2
from image_management import ManageFrames
from create_panorama import ManagePanorama


class VideoFeed:
    def __init__(self, video_path: str | int = 0,
                 interval: int = 20,
                 merge_size: int = 5,
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
        frame_n = 0
        last_frame = None
        self.capture = cv2.VideoCapture(self.video_path)
        while self.capture.isOpened():
            ret, frame = self.capture.read()
            if not ret:
                if last_frame is not None:
                    merged = self.panorama_manager.final_merge(last_frame)
                    last_merge = merged[-1]
                    self.save_image('merged', last_merge)
                    frames_boxed = self.frame_manager.wrap_img(merged) # Update when wrap method is done
                    for id, box in enumerate(frames_boxed):
                        self.save_image(f'boxed_img_{id}', box)
                break

            elif frame_n == 0:
                frame_n = self.set_video_values(frame)
                frame = cv2.resize(frame, (self.width, self.height))
            else:
                frame = cv2.resize(frame, (self.width, self.height))
                frame = self.frame_manager.extract_roi(frame)
            if frame_n % self.interval == 0:
                frame = self.panorama_manager.add_image(frame)
                if self.panorama_manager.success:
                    self.save_image(f'stitched_panorama_{frame_n}', frame)
            cv2.imshow('frame', frame)
            frame_n += 1
            last_frame = frame
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
        self.frame_manager = ManageFrames(self.config)
        self.panorama_manager = ManagePanorama(self.merge_size)
        self.frame_manager.set_manager_values(frame)
        return 1

    def save_image(self, filename: str, frame):
        cv2.imwrite(
            f'outputs/img/{self.merge_size}-{self.interval}_{filename}.png',
            frame)
