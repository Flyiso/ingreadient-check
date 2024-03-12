"""
Manages, runs and imports lbl readers modules.
"""
from label_video_reader import ReadLabelVideo
from image_management import FindText
import pytesseract as pt


def run_app(filepath):
    """
    Run all components of application in order-
    filepath: path to video of label
    """
    frame_l = 10
    frame_s = 30
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
                print('...')
        except FileNotFoundError:
            print('DID NOT WORK')
        except pt.TesseractError:
            print('DID NOT WORK FOR ANOTHER REASON')
        print('')


run_app('test_video.mp4')
