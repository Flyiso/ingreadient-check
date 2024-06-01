"""
uses images to try to create a panorama of all.
"""
import cv2
import numpy as np


class ManagePanorama:
    """
    class responsible for building panorama image of
    output frames.
    """
    def __init__(self,
                 frame_manager,
                 interval: int = 10) -> None:
        """
        parameters:
            interval(int): distance in frames between every merge default: 10
        """
        self.interval = interval
        self.frames = []
        self.to_stitch = []
        self.stitched = []
        self.selected = []
        self.base = False
        self.panorama_merged = False
        self.merge_counter = 1
        self.fail_counter = 0
        self.stitcher = cv2.Stitcher.create(1)
        self.stitcher.setWaveCorrection(cv2.WARP_POLAR_LINEAR)
        self.stitcher.setCompositingResol(-2)
        self.stitcher.setInterpolationFlags(cv2.INTER_LANCZOS4)
        self.stitcher.setPanoConfidenceThresh(1)
        self.stitcher.setRegistrationResol(-1.5)
        self.stitcher.setSeamEstimationResol(-5)  # fails/interval: 0
        self.frame_manager = frame_manager

    def add_frame(self, frame, last_frame: bool = False) -> bool:
        """
        Adds a new frame to the mergers frames.
        Tries self.interval frames before if unsuccessful.
        Returns True False when first merge is successfull
        Returns True True when also second
        merge(of already merged frames) succeeds.
        returns False False else.
        Last return value tells if any merge was attempted.
        Parameters:
            frame(np.ndarray):  frame to attempt to merge in to panorama.
            last_frame(bool):   Defaults to False. When true, stitcher tries to
                                do a final merge with the new image even if
                                minimum number of frames to
                                do a merge is not reached.
        """
        self.frames.append(frame)
        if last_frame:
            bool_b = self.stitch_stitched_frames(last_frame)
            return False, bool_b, True
        if isinstance(self.base, np.ndarray):
            print(f'frames: {len(self.frames)}')
            return False, False, False

        if all([bool(len(self.frames) % self.interval != 0),
                isinstance(self.base, np.ndarray),
                last_frame is False]):
            print(f'frames: {len(self.frames)}')
            return False, False, False

        bool_a = False
        if len(self.frames) % self.interval == 0:
            bool_a = self.add_more_frames(frame, last_frame)

        bool_b = self.stitch_stitched_frames(last_frame) if any(
            [last_frame, bool_a]) else False

        cv2.imwrite('progress_images/base.png', self.base)
        return bool_a, bool_b, True

    def add_more_frames(self, frame: np.ndarray,
                        last_frame: bool) -> bool:
        """
        Modified version of frame stitching to test if
        more frames solve stitcher error 1 problems
        """
        if not isinstance(frame, np.ndarray):
            return False
        if self.base is False:
            base = self.frame_manager.find_label(frame)
            if isinstance(base, bool):
                return False
            self.base = base
            self.to_stitch.append(self.base)
            self.frame_manager.set_manager_values(self.base)
            return True
        status = None
        # Find last added frame in self.frames where label is existing
        n_frames = len(self.frames)
        for frame_id, frame in enumerate(self.frames[::-1], 1):
            frame = self.frame_manager.find_label(frame)

            if frame is False:
                # remove bad frames and try next one.
                self.frames.pop(n_frames-frame_id)
                continue

            self.to_stitch.append(frame)
            self.selected.append(frame)
            if \
                len(self.to_stitch) >= 3 or\
                    all([last_frame, len(self.to_stitch) >= 2]):

                #status, result = \
                    #self.stitcher.stitch(self.to_stitch, list(
                #        map(lambda m1, m2: cv2.bitwise_and(m1, m2),
                #            self.frame_manager.get_masks(self.to_stitch),
                #            self.frame_manager.get_text_masks(self.to_stitch,
                #                                              20)
                #            )))
                result = ''

                if status == cv2.Stitcher_OK:
                    cv2.imwrite('progress_images/merged.png', result)
                    if not self.panorama_merged:
                        self.base = result
                    self.stitched.append(result)
                    self.to_stitch = []
                    return True
            # Keep add frames to force merge of last frames of video.
            if not last_frame:
                return False
        return False

    def stitch_stitched_frames(self, last_frame: bool):
        """
        Attempts to stitch together already stitched frames.
        """
        print(f'is current last?: {last_frame}')
        if last_frame:
            print('LAST FRAMES, SAVE IMAGES...')
            self.stitched = self.frame_manager.get_most_different(
                self.frames, 16)

            for f_id, f_obj in enumerate(self.stitched):
                print(f'{f_id+1}/{len(self.stitched)}')
                cv2.imwrite(f'progress_images/p_{f_id}.png', f_obj)

            stitcher = cv2.Stitcher.create(0)
            #stitcher.setInterpolationFlags(cv2.INTER_LANCZOS4)
            stitcher.setWaveCorrection(cv2.WARP_POLAR_LINEAR)
            #stitcher.setWaveCorrection(cv2.WARP_POLAR_LOG)
            #stitcher.setRegistrationResol(0.1)
            #stitcher.setCompositingResol(0.1)
            #stitcher.setSeamEstimationResol(1)
            #stitcher.setPanoConfidenceThresh(0.9)

            status1, result1 = stitcher.stitch(
                self.stitched[:int(len(self.stitched)*0.75):],
                list(map(lambda m1, m2: cv2.bitwise_and(m1, m2),
                         self.frame_manager.get_masks(
                                        self.stitched[:int(
                                            len(self.stitched)*0.75):]),
                         self.frame_manager.get_text_masks(
                                        self.stitched[:int(
                                            len(self.stitched)*0.75):],
                                        33))))
            status2, result2 = stitcher.stitch(
                self.stitched[(len(self.stitched)//4)::],
                list(map(lambda m1, m2: cv2.bitwise_and(m1, m2),
                         self.frame_manager.get_masks(
                                        self.stitched[
                                            (len(self.stitched)//4)::]),
                         self.frame_manager.get_text_masks(
                                        self.stitched[
                                            (len(self.stitched)//4)::],
                                        33))))
            status = False
            if status1 == cv2.Stitcher_OK and status2 == cv2.Stitcher_OK:
                cv2.imwrite('progress_images/m_1.png', result1)
                cv2.imwrite('progress_images/m_2.png', result2)
                status, result = stitcher.stitch([result1,
                                                  self.stitched[
                                                      len(self.stitched)//2],
                                                  result2])
                if status == cv2.Stitcher_OK:
                    self.base = result
                    print('success!')
                    cv2.imwrite('progress_images/final_base.png', result)
                    return True
            return False

    def detect_text(self):
        data = self.frame_manager.find_text(self.base)
        print(
            ' '.join(data['text']))
        return self.base
