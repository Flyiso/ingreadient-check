"""
File collecting methods for frame selection and evaluation
"""
from abc import ABC, abstractmethod
from flow_management import RecordLabel


class ThreshMethods(ABC):
    """
    Classes that manage when to merge new frame to panorama.
    """
    @abstractmethod
    def check_threshold(self, check_class: RecordLabel) -> bool:
        """
        Return True or False depending on if threshold criteria
        is reached.
        """
        pass


class ThreshFrameNumber(ThreshMethods):
    def check_threshold(self, check_class: RecordLabel) -> bool:
        """
        Evaluate if threshold is reached by how many frames since merge.
        """
        if len(check_class.frame_memory) >= check_class.merge_interval:
            return True
        return False


class ThreshFrameDiff(ThreshMethods):
    def check_threshold(self, check_class: RecordLabel) -> bool:
        """
        Evaluate if threshold is reached by current frame difference
        since last merged frame
        """
        pass


class ThreshWordCompare(ThreshMethods):
    def check_threshold(self, check_class: RecordLabel) -> bool:
        """
        Evaluate if threshold is reached by detected word similarities.
        """
        pass


class ThreshWordAmount(ThreshMethods):
    def check_threshold(self, check_class: RecordLabel) -> bool:
        """
        Evaluate by how much text found by Pytesseract.
        """
        pass


class ThreshEstimatedSharpness(ThreshMethods):
    def check_threshold(self, check_class: RecordLabel) -> bool:
        """
        Evaluate by if frame is approved by estimation of it's sharpness.
        """
        pass
