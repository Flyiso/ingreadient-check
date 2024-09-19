import cv2
from frame_evaluations import ThreshMethods, ThreshFrameNumber


class RecordLabel:
    """
    Class to manage flow, selection and storage of video frames
    """
    def __init__(self, video_path: str,
                 thresh_methods: list = [ThreshFrameNumber],
                 thresh_params: list | None = None,
                 pre_process_steps: list = [],
                 post_process_steps: list = [],
                 merge_evaluation_steps: list = []):
        """
        Start video flow and set base parameters for flow management.

        :param video_path: string. path to video location.
        :param thresh_method: List of classes that select what frames to pass.
        :param thresh_params: List of custom/individual settings, as dicts.
        :param pre-process_steps: List, method(s) for frame pre(merge)-process.
        :param post-process_steps: List, method(s) for processing panorama.
        :param merge_evaluation_seps: List methods(s) eval. satisfactory merge.
        """
        # set self.thresh_methods & get error if any non-valid values.
        self.validate_thresholds(thresh_methods)
        self.thresh_methods = thresh_methods

        # set self.pre_process_steps & get error if any non-valid values.

        # set self.post_process_steps & get error if any non-valid values.

        # set self.merge_evaluation_seps & get error if any non-valid values.

        # TODO:Add selection of how to run,backtracking and so on.
        # TODO:Add 
        # TODO:Update path to also allow live recording
        # TODO:Method for if merged_frame_n + interval < current_frame_n
        # TODO:Add methods for checking all other values.
        # TODO:Add method to run all selected operations.
        self.video_path = video_path
        self.frame_memory = []
        self.merge_interval = 50
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
        # pre-process
        # find_label, check shape/if it is similar to previous label detection
        # flattening and filling of empty spaces.
        # attempt merge
        # evaluate merge
        # -> if evaluation ok, reset panorama to newly merged
        # return result(ass merge pass or fail)
        pass

    def update_parameters(self, frame_number) -> None:
        """
        Method to update thresholds, variables for frame flow algorithm.
        """
        self.last_merged = self.last_merged+(self.interval-frame_number)
        self.frame_memory = []
        # other stuff depending on threshold choices?
        # compare to interval+adjust

    @staticmethod
    def validate_thresholds(thresh_methods: list):
        """
        Raise AttributeError if any bad arguments in thresh methods.

        :param thresh_methods: list of methods(instances of ThreshMethods)
        """
        if not all(isinstance(t_method, ThreshMethods) for
                   t_method in thresh_methods):
            print('Error- one or more threshold_method options are not valid')
            raise AttributeError(
                f''' approved threshold values are: {
                ", ".join(
                    [t_method.__name__ for t_method in
                     ThreshMethods.__subclasses__()])},\n\
                got bad values:
                {", ".joint(str(t_method) for t_method in
                 thresh_methods if not isinstance(t_method, ThreshMethods))}'''
                    )

    @staticmethod
    def validate_thresh_params(thresh_methods: list,
                               thresh_params: list | None):
        """
        Raise attribute error if any non-valid thresh_param values

        :param thresh_methods: threshold methods/classes choosen
        :param thresh_params: [{'name':MethodName, 'values':[custom settings]}]
        """
        pass
