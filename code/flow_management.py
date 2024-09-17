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
                self.try_frame()
        self.capture.release()
        cv2.destroyAllWindows()

    def try_frame(self):
        """
        Try to merge frame saved in self.frame_memory, looped backwards.
        """
        pass

    def test_threshold(self):
        """
        Method to test if threshold is reached for self.
        This method is to allow test of different threshold methods.
        """
        if self.thresh_frame_interval:
            if len(self.frame_memory) >= self.merge_interval:
                return True
            return False
