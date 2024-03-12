import pytesseract as pt
import cv2
from create_panorama import CreatePanorama


class ReadLabelVideo:
    """
    class responsible for managing the video feed
    """
    def __init__(self, video_path: str | int = 0) -> None:
        """
        Get video capture and call image processing class
        Streams from camera if no video input.
        """
        self.video_path = video_path
        self.start_video()
        self.panorama_data = self.frame_manager.panorama_frames
        for data in self.panorama_data:
            print(data.shape)
        for panorama1, panorama2 in zip(self.panorama_data[::2],
                                        self.panorama_data[1::2]):
            print('.......')
            print(panorama1.shape)
            print(panorama2.shape)
        if len(self.panorama_data) % 2 == 1:
            print(f'manage panorama_data[-1]\n{self.panorama_data[-1]}')

        for num, panorama in enumerate(self.panorama_data):
            cv2.imwrite(f'outputs/panorama_{num}.png', panorama)
        print(f'min text regions found in allowed image: {self.frame_manager.min}') # to check remove later

    def start_video(self):
        """
        Loops through the frames and decides how to manage them
        """
        frame_n = 0
        self.capture = cv2.VideoCapture(self.video_path)
        while self.capture.isOpened():
            ret, frame = self.capture.read()

            # end video, add last frame if no frame recently added.
            if not ret:
                self.frame_manager.check_frame(self.last_frame, True)
                break

            # set parameters for size and initialize Collect frames
            elif frame_n == 0:
                frame_n = 1
                self.initialize_video_parameters(frame)

            # get new frame for panorama build in interval of 25
            elif frame_n % 20 == 0:
                frame = cv2.resize(frame, (self.width, self.height))
                if self.frame_manager.check_frame(frame):
                    frame_n += 1
            else:
                frame = cv2.resize(frame, (self.width, self.height))
                frame_n += 1

            # Display the current frame.
            cv2.imshow('frame', frame)
            self.last_frame = frame
            if cv2.waitKey(25) & 0xFF == 27:
                break

        self.capture.release()
        cv2.destroyAllWindows()

    def initialize_video_parameters(self, first_frame):
        """
        sets video feed parameters and initializes CollectFrames
        """
        self.height, self.width = int(
            first_frame.shape[:2][0]/2), int(first_frame.shape[:2][1]/2)
        self.frame_manager = CollectFrames(cv2.resize(
            first_frame, (self.width, self.height)))


class CollectFrames:
    """
    Class responsible for building of panorama image.
    collects images with detected text as self.panorama_frames
    """
    def __init__(self, first_frame) -> None:
        """
        initialize building of panorama image
        """
        self.n = 0
        self.min = 5000  # to check min detection- remove later
        self.panorama_frames = []
        self.check_frame(first_frame)
        self.panorama_stitcher = CreatePanorama()

    def check_frame(self, frame, last_frame: bool = False):
        """
        Concludes if frame quality is suitable
        for text recognition and panorama build.
        Adds frame and returns True if image ok.
        """
        print('Frame_check...')
        data = False
        data = self.find_text(frame)
        if last_frame is True:
            self.panorama_stitcher.final_merge(self.frame
                                               if data is True else None)
            return True

        if bool(data) is True:
            self.panorama_stitcher.add_image(self.frame)
            return True
        return False

    def find_text(self, frame):
        """
        concludes if text can be found in frame
        """
        data = pt.image_to_data(frame, config='--oem 3 --psm 6',
                                output_type='dict',
                                lang='swe')
        if len(data['text']) >= 10:
            if len(data['text']) < self.min:  # remove
                self.min = len(data['text'])  # remove
                print('new low')              # remove
            self.find_text_region(data, frame)
            return True
        return False

    def find_text_region(self, data, frame):
        """
        Crop out the section of the image where text is found.
        """
        self.n += 1
        self.frame = frame[min(data['left'][:1]):
                           min(data['left'][:1])+max(data['width'][:1]),
                           min(data['top'][:1]):
                           min(data['top'][1:])+max(data['height'][1:])]
        cv2.imwrite(f'outputs/cropped_{self.n}.png', self.frame)

    def merge_panoramas(self):
        """
        tries to merge all panoramas together.
        """
        print(f'final merge of {len(self.panorama_frames)} panoramas')
        failure_frames = []
        while len(self.panorama_frames) > 1:
            print(f'frames: {len(self.panorama_frames)}  goal: 1')
            for frame_1, frame_2 in zip(self.panorama_frames[::2],
                                        self.panorama_frames[1::1]):
                try:
                    merged = self.panorama_stitcher.add_images([frame_1,
                                                                frame_2])
                    self.panorama_frames.append(merged)
                    print(f'merged frames!- current no frames {len(self.panorama_frames)}')
                except TypeError:
                    print('failure...')
                    failure_frames.append(frame_1)
                    failure_frames.append(frame_2)
                    self.panorama_frames.pop(0)
                    if len(self.panorama_frames) > 1:
                        self.panorama_frames.pop(1)
        for idx, img in enumerate(self.panorama_frames):
            cv2.imwrite(f'outputs/merged_{idx}.png', img)
        for idx, img in enumerate(failure_frames):
            cv2.imwrite(f'outputs/fail_{idx}.png', img)
