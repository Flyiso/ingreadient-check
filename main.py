"""
Manages, runs and imports readers modules.
# TODO: enhance images. improve for better text recognition
# TODO: adjust/make adaptable to other videos.
# TODO: make frame/image edit class more consistent/easy to use.
# TODO: Update frame merging steps to better manage frame merging failures.

# TODO: decide when to enhance text.
# TODO: Save line information(lmage_managmement/ManageFrames/
        detect_text_direction)
        together with frame and use
        information when warping image for perspective.
"""
from label_video_reader import VideoFeed


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
    VideoFeed(video_path=filepath,
              interval=14, merge_size=4,
              adjust_h=0.5, adjust_w=0.5,
              config='--oem 3 --psm 6')


run_app('test_video.mp4')
