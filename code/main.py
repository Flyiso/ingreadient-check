"""
Manages, runs and imports readers modules.
"""
import os
import shutil
from video_manager import RecordLabel


def check_directory(img_dir: str, frame_interval: int):
    """
    Makes sure there is a directory with name of img_dir value.
    removes directories if they exist to ensure results from different
    runs does not mix up.
    """
    if os.path.exists('progress_images'):
        shutil.rmtree('progress_images')
    os.makedirs('progress_images')
    if os.path.exists(f'outputs/{img_dir}/{frame_interval}'):
        shutil.rmtree(f'outputs/{img_dir}/{frame_interval}')
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
    if img_dir is not None:
        check_directory(img_dir, frame_interval)
    RecordLabel(video_path=filepath,
                interval=frame_interval,
                adjust_h=0.45, adjust_w=0.45,
                pt_config='oem-- 3 --psm 6',
                img_dir=img_dir)


run_app('videos/test_video_2.mp4', 'vid_2', frame_interval=15)
