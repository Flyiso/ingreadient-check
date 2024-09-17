import cv2


class RecordLabel:
    """
    Class to manage flow, selection and storage of video frames
    """
    def __init__(self):
        pass
        # find first 'good enough' frame
        # save frames in interval of x.
        # try to merge in last of saved
        # frames to panorama(or first frame)
        # bactrac OR remove/restart frame storage
        # possibly adjust frame distance.
        # from last successfull merge.
        # return final image
