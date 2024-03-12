"""
Manages and extracts data from
merged images.
"""
import pytesseract as pt


class FindText:
    def __init__(self, frames: list, config: str = '--psm 12') -> None:
        self.text = []
        for frame in frames:
            self.text.append(self.find_text(frame, config))

    def find_text(self, frame, config):
        """
        concludes if text can be found in frame
        """
        data = pt.image_to_data(frame, config=config,
                                output_type='dict',
                                lang='swe')
        return ' '.join(data['text']).replace('  ', ' ')
