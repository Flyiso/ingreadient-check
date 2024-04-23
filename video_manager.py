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
        self.panorama_manager = ManagePanorama(ManageFrames(config), interval)
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

    def track_motion(self, frame1, frame2):
        """
        return img w motion track
        """
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

        # Compute optical flow using Lucas-Kanade method
        lk_params = dict(winSize=(15, 15),
                         maxLevel=2,
                         criteria=(
                             cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                             10, 0.03))
        prev_pts = cv2.goodFeaturesToTrack(gray1, maxCorners=100,
                                           qualityLevel=0.3, minDistance=7,
                                           blockSize=7)
        next_pts, status, err = cv2.calcOpticalFlowPyrLK(gray1, gray2,
                                                         prev_pts, None,
                                                         **lk_params)

        # Select good points
        good_new = next_pts[status == 1]
        good_old = prev_pts[status == 1]

        # Draw motion vectors
        for i, (new, old) in enumerate(zip(good_new, good_old)):
            a, b = new.ravel().astype(int)
            c, d = old.ravel().astype(int)
            frame2 = cv2.line(frame2, (a, b), (c, d), (0, 255, 0), 2)
            frame2 = cv2.circle(frame2, (a, b), 5, (0, 0, 255), -1)

        # Display frames with motion vectors
        cv2.imshow('Optical Flow', frame2)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
