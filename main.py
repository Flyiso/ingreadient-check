"""
Manages, runs and imports glbl readers modules.
"""
from label_video_reader import VideoFeed


def run_app(filepath):
    """
    Run all components of application in order-
    filepath: path to video of label
    """
    configs = ['--psm 1', '--psm 2', '--psm 3', '--psm 4', '--psm 5',
               '--psm 6', '--psm 7', '--psm 8', '--psm 9', '--psm 10',
               '--psm 11', '--psm 12', '--psm 13']
    VideoFeed(video_path=filepath,
              interval=20, merge_size=3,
              adjust_h=0.5, adjust_w=0.5,
              config='--oem 3 --psm 6')


run_app('test_video.mp4')
