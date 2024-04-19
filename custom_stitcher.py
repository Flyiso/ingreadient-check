import cv2


class CustomStitcher:
    def __init__(self, method: str) -> None:
        self.select_descriptor_method(method)
        self.select_match_method(method)

    def use_stitcher(self, frame_1, frame_2):
        self.keypoints_img_1, self.features_img_1 = \
            self.use_descriptor(frame_1)
        self.keypoints_img_2, self.features_img_2 = \
            self.use_descriptor(frame_2)

    def select_descriptor_method(self, method: str):
        """
        Select what descriptor to use.
        """
        assert method is not None, \
            "Define Descriptor. Accepted: 'sift', 'surf', 'orb', 'brisk'."
        if method == 'sift':
            self.descriptor = cv2.SIFT.create()
        if method == 'surf':
            self.descriptor = cv2.SURF.create()
        if method == 'orb':
            self.descriptor = cv2.ORB.create()
        if method == 'brisk':
            self.descriptor = cv2.BRISK.create()

    def select_match_method(self, method: str):
        """
        select method for keypoint matching
        """
        if method == 'sift' or method == 'surf':
            self.norm_type = cv2.NORM_L2
            self.matcher = cv2.BFMatcher
        if method == 'orb' or method == 'brisk':
            self.norm_type = cv2.NORM_HAMMING
            self.matcher = cv2.BFMatcher

    def use_descriptor(self, frame):
        (keypoints, features) = self.descriptor.detectAndCompute(frame, None)
        return (keypoints, features)

    def use_matcher(self, crosscheck: bool):
        matcher = self.matcher(self.norm_type, crossCheck=crosscheck)
        best_matches = matcher.match(self.features_img_1, self.features_img_2)
        raw_matches = sorted(best_matches, key=lambda x: x.distance)
