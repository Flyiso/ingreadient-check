"""
All options for frame pre-process before merge attempt.
TODO:
#flatten image
#remove artifacs/inpaint
#enhance, increase contrasts
#enhance-with clahe
#segment/extract roi - text detected
#segment/extract roi - DINO & SAM
"""
import numpy as np
from abc import ABC, abstractmethod
from flow_management import RecordLabel


class PreProcess(ABC):
    """
    Classes for frame pre-proceesing
    """
    @abstractmethod
    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        return frame pre-proceesed
        """

