import pytesseract as pt
import cv2
from create_panorama import CreatePanorama
import numpy as np


class ReadLabelVideo:
    """
    class responsible for managing the video feed
    """
    def __init__(self, video_path: str | int = 0,
                 save_img: bool = False,
                 frame_l: int = 5, frame_s: int = 25,
                 pan_group: int = 3) -> None:
        """
        Get video capture and call image processing class
        Streams from camera if no video input.
        video_path(str): Path to video, uses camera if none
        save_img(bool): if to save output images
        frame_l(int): number of frames to check for best output in
        frame_s(int): interval between frame checks
        pan_group(int): how many images to merge each time.
        """
        self.pan_group = pan_group
        self.video_path = video_path
        self.start_video(frame_l, frame_s)
        #self.panorama = self.frame_manager.panorama_created
        #if save_img is True:
        #    for id, pan in enumerate(self.panorama):
        #        print(type(pan))
        #        if pan is not None:
        #            print('saves!')
        #            cv2.imwrite(f'outputs/outputs_6/PANORAMA_{id}.png', pan)

    def start_video(self, frame_l, frame_s):
        """
        Loops through the frames and decides how to manage them
        """
        frame_n = 0
        frame_limit = frame_l
        frame_spacing = frame_s
        frames = []
        self.capture = cv2.VideoCapture(self.video_path)
        while self.capture.isOpened():
            ret, frame = self.capture.read()

            # end video, add last frame if no frame recently added.
            if not ret:
                self.frame_manager.check_frame(frames, True)
                break

            # set parameters for size and initialize Collect frames
            elif frame_n == 0:
                frame_n = 1
                self.initialize_video_parameters(frame)
                frame = cv2.resize(frame, (self.width, self.height))
            elif frame_n == frame_limit:
                frame = cv2.resize(frame, (self.width, self.height))
                if self.frame_manager.check_frame(frames):
                    frame_n += 1
            # get new frame for panorama build in interval of 25
            elif frame_n % frame_spacing == 0:
                frame = cv2.resize(frame, (self.width, self.height))
                if self.frame_manager.check_frame(frames):
                    frame_n += 1
            else:
                frame = cv2.resize(frame, (self.width, self.height))
                frame_n += 1
            frames.append(frame)
            if len(frames) >= frame_limit:
                frames = frames[1:]
            # Display the current frame.
            cv2.imshow('frame', frame)
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
        self.frame_manager = CollectFrames(self.pan_group)


class CollectFrames:
    """
    Class responsible for building of panorama image.
    collects images with detected text as self.panorama_frames
    """
    def __init__(self, pan_group) -> None:
        """
        initialize building of panorama image
        """
        self.panorama_stitcher = CreatePanorama(pan_group)
        self.panorama_created = self.panorama_stitcher.stitched

    def check_frame(self, frames: list, last_frame: bool = False):
        """
        Concludes if frame quality is suitable
        for text recognition and panorama build.
        Adds frame and returns True if image ok.
        """
        data_max = 0
        data = 0
        for frame in frames:
            data = self.find_text(frame)
            if data >= data_max:
                final_frame = self.frame
                data_max = data

        if last_frame is True:
            print('final frame...')
            self.panorama_stitcher.final_merge(final_frame
                                               if data != 0 else None)
            return True
        if bool(data) != 0:
            self.panorama_stitcher.add_image(final_frame)
            return True
        return False

    def find_text(self, frame):
        """
        concludes if text can be found in frame
        """
        frame_g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_g = cv2.GaussianBlur(frame_g, (5, 5), 0)
        lines = cv2.HoughLinesP(frame_g,
                                rho = 1,
                                theta = np.pi / 180,
                                threshold = 100,
                                lines = np.array([]),
                                minLineLength = 50,
                                maxLineGap = 5)
        print(lines)
        for line in lines:
            for start_x, start_y, end_x, end_y in line:
                cv2.line(frame, (start_x, start_y), (end_x, end_y), (0, 255, 0), 2)
        self.frame = frame
        cv2.imwrite('outputs/shaped/hlines.png', frame)
        #data = pt.image_to_data(frame_g, config='--oem 3 --psm 6',
        #                        output_type='dict',
        #                        lang='swe')
        #if len(data['text']) >= 20:
        #    self.find_text_region(data, frame)
        #    return len(data['text'])
        return 1

    def find_text_region(self, data, frame):
        """
        Crop out the section of the image where text is found.
        """
        s_size = 100
        section_width = int(frame.shape[1]/s_size)
        section_height = int(frame.shape[0]/s_size)
        long = []
        lati = []
        for n in range(1, s_size+1):
            hig = [0, 0]
            btm = [0, frame.shape[1]]
            lft = [frame.shape[0], 0]
            rgt = [0, 0]
            for left, top in zip(data['left'], data['top']):
                if (section_width*n)-section_width <= left and left <= n*section_width:
                    if hig[1] < top:
                        hig = [left, top]
                    if btm[1] > top:
                        btm = [left, top]
                if (n*section_height)-section_height <= top and top <= section_height*n:
                    if rgt[0] < left:
                        rgt = [left, top]
                    if lft[0] > left:
                        lft = [left, top]
            if btm[1] == frame.shape[1]:
                btm[1] = 0
            if lft[0] == frame.shape[0]:
                lft[0] = 0
            long.append([tuple(hig), tuple(btm)])
            lati.append([tuple(lft), tuple(rgt)])
        #vertices = upper + right + lower + left_
        #vertices = set(vertices)
        #vertices = list(vertices)

        for ve in long:
            if isinstance(ve[1], tuple) and isinstance(ve[0], tuple):
                if ve[0] != (0, 0) and ve[1] != (0, 0):
                    cv2.line(frame, ve[0], ve[1], (0, 255, 0), 2)
        for ve in lati:
            if isinstance(ve[1], tuple) and isinstance(ve[0], tuple):
                if ve[0] != (0, 0) and ve[1] != (0, 0):
                    cv2.line(frame, ve[0], ve[1], (255, 0, 255), 2)
        for w, h in zip(data['left'], data['top']):
            cv2.circle(frame, (w, h), 1, (0, 0, 255), 3)
        cv2.imwrite('outputs/dotted.png', frame)


        self.frame = frame[min(data['left'][:1]):
                           max(data['left'][:1])+max(data['width'][:1]),
                           min(data['top'][:1]):
                           max(data['top'][1:])+max(data['height'][1:])]

    def create_roi_mask(self, vertices, edges):
        """
        create roi mask for interesting section of video.
        """
        mask = np.zeros_like(edges)
        mask_color = 255
        polyg = cv2.fillPoly(mask, vertices, mask_color)
        bit_mask = cv2.bitwise_and(edges, mask)
        return bit_mask
