"""
Manages, runs and imports readers modules.
"""
import os
from video_manager import RecordLabel


def check_directory(img_dir: str, frame_interval: int):
    """
    Makes sure there is a directory with name of img_dir value.
    """
    if not os.path.exists(f'outputs/{img_dir}/'):
        # Create the directory
        os.makedirs(f'outputs/{img_dir}/{frame_interval}/')


def run_app(filepath, img_dir: str = None,
            frame_interval: int = 20):
    """
    Run all components of application in order-
    filepath: path to video of label
    """
    """configs = ['--psm 1', '--psm 2', '--psm 3', '--psm 4', '--psm 5',
               '--psm 6', '--psm 7', '--psm 8',
               '--psm 11', '--psm 12', '--psm 13']"""

    # 14, 15, 16, 17 works ok in interval <-- but keep as high as possible.
    # 2, 3, 4 works ok in merge_size
    if img_dir is not None:
        check_directory(img_dir, frame_interval)
    RecordLabel(video_path=filepath,
                interval=frame_interval,
                adjust_h=0.40, adjust_w=0.40,
                pt_config='oem-- 3 --psm 6',
                img_dir=img_dir)


run_app('videos/test_video_2.mp4', 'vid_2', frame_interval=13)
