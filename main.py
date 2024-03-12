"""
Manages, runs and imports lbl readers modules.
"""
from lbl_video_reader import ReadLabelVideo
from image_management import FindText


def run_app(filepath):
    """
    Run all components of application in order-
    filepath: path to video of label
    """
    frame_l = 5
    frame_s = 15
    pan_group = 3
    save_img = False

    configs = ['--psm 1', '--psm 2', '--psm 3', '--psm 4', '--psm 5',
               '--psm 6', '--psm 7', '--psm 8', '--psm 9', '--psm 10',
               '--psm 11', '--psm 12', '--psm 13']
    label_data = ReadLabelVideo(filepath, save_img,
                                frame_l, frame_s,
                                pan_group)
    for config in configs:
        print('')
        print(config)
        print('')
        try:
            text_data = FindText(frames=label_data.panorama, config=config)
            for item in text_data.text:
                print(item)
        except:
            print('DID NOT WORK')
        print('')


run_app('test_video.mp4')
