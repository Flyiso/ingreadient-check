"""
Manages, runs and imports lbl readers modules.
"""
from lbl_video_reader import ReadLabelVideo


def run_app(filepath):
    """
    Run all components of application in order-
    filepath: path to video of label
    """
    ReadLabelVideo(filepath)


run_app('test_video.mp4')
