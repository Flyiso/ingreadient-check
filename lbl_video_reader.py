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
        self.panorama_data = self.frame_manager.generated_panoramas
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
                if frame_n % 20 < 5:
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
        self.panorama_frames = []
        self.check_frame(first_frame)
        self.generated_panoramas = []
        self.panorama_stitcher = CreatePanorama()

    def check_frame(self, frame, last=False):
        """
        Concludes if frame quality is suitable
        for text recognition and panorama build.
        Adds frame and returns True if image ok.
        """
        print('Frame_check...')
        data = False
        data = self.find_text(frame)
        if bool(data) is True:
            self.panorama_frames.append(self.frame)
            if last is True and len(self.panorama_frames) == 1:
                self.generated_panoramas.append(
                    [self.panorama_frames[-1], self.frame])
            if len(self.panorama_frames) == 2 or last is True:
                self.generated_panoramas.append(
                    self.panorama_stitcher.add_images(self.panorama_frames))
                self.panorama_frames = []
            return True
        return False

    def find_text(self, frame):
        """
        concludes if text can be found in frame
        """
        data = pt.image_to_data(frame, config='--oem 3 --psm 6',
                                output_type='dict',
                                lang='swe')
        if len(data['text']) >= 2:
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
