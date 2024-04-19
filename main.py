"""
Manages, runs and imports readers modules.
"""

from video_manager import RecordLabel


def run_app(filepath):
    """
    Run all components of application in order-
    filepath: path to video of label
    """
    """configs = ['--psm 1', '--psm 2', '--psm 3', '--psm 4', '--psm 5',
               '--psm 6', '--psm 7', '--psm 8',
               '--psm 11', '--psm 12', '--psm 13']"""

    # 14, 15, 16, 17 works ok in interval <-- but keep as high as possible.
    # 2, 3, 4 works ok in merge_size
    RecordLabel(video_path=filepath,
                interval=5, merge_size=3,
                frame_space=3,
                adjust_h=0.25, adjust_w=0.25,
                config='--oem 3 --psm 6')


run_app('test_video_4.mp4')
