import cv2


class RecordLabel:
    """
    Class to manage flow, selection and storage of video frames
    """
    def __init__(self, video_path: str):
        # Update path to also allow live recording
        self.video_path = video_path
        self.frame_memory = []
        self.merge_interval = 50
        self.thresh_frame_interval = True
        self.start_video()

        pass
        # find first 'good enough' frame
        # save frames in interval of x.
        # try to merge in last of saved
        # frames to panorama(or first frame)
        # backtrack OR remove/restart frame storage
        # possibly adjust frame distance.
        # from last successfull merge.
        # return final image

    def start_video(self):
        """
        loop through all frames in video.
        """
        self.capture = cv2.VideoCapture(self.video_path)
        while self.capture.isOpened():
            ret, frame = self.capture.read()
            if not ret or cv2.waitKey(25) & 0xFF == 27:
                break
            self.frame_memory.append(frame)
            if self.test_threshold:
                self.try_frames()
        self.capture.release()
        cv2.destroyAllWindows()

    def try_frames(self) -> bool:
        """
        Try to merge frame saved in self.frame_memory, looped backwards.

        :return: Bool- true or false depending on merge success.
        """
        for frame_number, frame in enumerate(self.frame_memory[::-1]):
            merged = self.merge_to_panorama(frame)
            if merged:
                self.update_parameters(self, frame_number)
                return True
        return False

    def merge_to_panorama(self, frame) -> bool:
        """
        Pre-process frame and try to merge it to the panorama.
        Evaluate results to ensure merge is useful.
        
        :return: True or False, to indicate if success.
        """
        pass

    def update_parameters(self, frame_number):
        """
        Method to update thresholds, variables for frame flow algorithm.
        """
        self.last_merged = self.last_merged+(self.interval-frame_number)
        # other stuff depending on threshold choices?

    def test_threshold(self):
        """
        Method to test if threshold is reached for self.
        This method is to allow test of different threshold methods.
        """
        if self.thresh_frame_interval:
            if len(self.frame_memory) >= self.merge_interval:
                return True
            return False
