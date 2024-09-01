"""
Classes directly related to stitching of images
ManagePanorama class that control the flow of images
through pre-processing and stitching.
StitcherSet class that control stitcher object parameters and initialization
NOTE: many images=better(and slower) result.
"""
import cv2
import numpy as np
from stitching import Stitcher, stitching_error


class ManagePanorama:
    """
    class responsible for building panorama image of
    output frames.
    """
    def __init__(self,
                 frame_manager, interval: int = 55,  # was 23
                 display_current: bool = False) -> None:
        """
        parameters:
            frame_manager(class):
                The class responsible for pre-processing of the frames in
                the video feed.
            interval(int), default=23:
                Every n frame to use in panorama. Declares how often the
                frame manager should be called to pre-process a frame
            display_current(bool), default=False:
                Bool to declare if the merger should be called every n frame
                to display continuous merger results.
        """
        self.frames = []
        self.to_stitch = []
        self.selected = []
        self.pre_processed = []
        self.base = False
        self.panorama = False
        self.interval = interval
        self.frame_manager = frame_manager
        if display_current:
            self.stitcher = StitcherSet(matcher_type='homography',
                                        warper_type='transverseMercator',
                                        compensator='channel_blocks')
            self.stitcher = self.stitcher.get_stitcher()
            self.last_merge = False
            return
        self.stitcher = StitcherSet()

    def add_frame(self, frame, last_frame: bool = False) -> bool | np.ndarray:
        """
        Adds a new frame to the mergers frames.
        tries to merge in image to base frame if continuous merge is on,
        and returns result.
        else, merges all (between interval) frames when last_frame is true.
        Parameters:
            frame(np.ndarray):  frame to attempt to merge in to panorama.
            last_frame(bool):   Defaults to False. When true, stitcher tries to
                                do a final merge with the new image even if
                                minimum number of frames to
                                do a merge is not reached.
        """
        if len(self.frames) == 0:
            self.frame_manager.set_manager_values(frame)
            base_candidate = self.frame_manager.find_label(
                frame)
            if isinstance(base_candidate, np.ndarray):
                self.base = base_candidate
                self.first_frame = self.base
                self.frames.append(frame)

        else:
            self.frames.append(frame)

        if isinstance(self.stitcher, Stitcher):
            print('STITCHER?')
            if len(self.frames) % self.interval == 0:
                self.stitch_current(last_frame)
            if last_frame is True:
                self.panorama = self.base
            return self.base

        elif last_frame:
            print('LAST FRAME?')
            if self.stitch_frames():
                cv2.imwrite('progress_images/base.png', self.base)
                return self.base
        return False

    def stitch_current(self, patience: int = 5) -> bool | np.ndarray:
        """
        Attempts To Stitch the base with the current to-add frames,
        and sets the result as new base.
        """
        for frame in self.frames[:-patience:-1]:
            frame = self.frame_manager.find_label(frame)
            if isinstance(frame, np.ndarray):
                self.pre_processed.append(frame)
                to_stitch = self.pre_processed[
                    -1::(len(self.pre_processed)//3)+1]
                to_stitch.append(self.first_frame)
                to_stitch.append(self.base)
                pan = self.stitcher.stitch(to_stitch[::-1])
                if isinstance(pan, np.ndarray):
                    self.base = cv2.addWeighted(pan, 0.5, pan, 0.5, 0)
                    return self.base

    def stitch_frames(self) -> bool:
        """
        Attempts to stitch together already stitched frames.
        Returns True on success.
        """
        self.to_stitch = self.frame_manager.get_most_different(
            self.frames, len(self.frames)//self.interval)
        print('frames collected...')
        stitcher = self.stitcher.get_stitcher()
        print(type(stitcher))
        print(len(self.to_stitch))
        try:
            panorama = stitcher.stitch(self.to_stitch)
        except stitching_error.StitchingError:
            print('stitching unsuccessful...')
            panorama = False
        if isinstance(panorama, np.ndarray):
            self.panorama = cv2.addWeighted(panorama, 0.5, panorama, 0.5, 0)
            cv2.imwrite('progress_images/FinalPanorama.png', self.panorama)
            return True
        return False

    def get_final_panorama(self) -> tuple[np.ndarray | bool, bool]:
        """
        Get final merge result if any. Get first approved frame from
        video else.
        returns:
            frame(np.ndarray | False):
                Returns panorama image merged from video frames.
                Returns False if no label at all was detected,
                returns the first approved frame if
                merging of panorama failed.
            success(bool):
                returns True when the first of this methods return values is
                the merged panorama,
                returns False if no image or the base image is returned.
        """
        if isinstance(self.panorama, np.ndarray):
            return self.panorama, True
        elif isinstance(self.base, np.ndarray):
            return self.base, False
        return False, False


class StitcherSet:
    """
    Class contain the stitcher used to do frame stitching.
    """
    def __init__(self, try_use_gpu: bool = True,
                 blend_strength: int = 5, block_size: int = 1,  # blend-s: 5, b-size:5
                 nr_feeds: int = 5, match_conf: float = 0.5,  # nfeed: 1, conf: 0.5
                 blender_type: str = 'multiband',  # multiband
                 compensator: str = 'gain',
                 detector: str = 'brisk',  # brisk
                 finder: str = 'dp_color',  # dp_color
                 matcher_type: str = 'affine',  # affine
                 warper_type: str = 'cylindrical',  # was 'cylindrical'
                 wave_correct_kind: str = 'no',  # no
                 ) -> None:
        """
        Control of the stitcher object.
        parameters:
            try_use_gpu(bool), default = True.
                If the stitching should be attempted to be done with gpu.
            blend_strength(int), default = 5
            block_size(int), default = 5
            nr_feeds(int), default = 1
            match_conf(float), default = 0.45
            blender_type(str), default = multiband
                options:[multiband, feather, no]
            compensator(str), default = gain_blocks
                options:[gain_blocks, gain, channel, channel_blocks, no]
            detector(str), default = sift
                options:[sift, orb, brisk, akaze]
            finder(str), default = dp_color
                options:[dp_color, dp_colorgrad, gc_color, gc_colorgrad, no]
            matcher_type(str), default = affine
                options:[affine, homography]
            warper_type(str), default = cylindrical
                options:[spherical, plane, affine, cylindrical, fisheye,
                         stereographic, compressedPlaneA2B1,
                         compressedPlaneA1.5B1, compressedPlanePortraitA2B1,
                         compressedPlanePortraitA1.5B1, paniniA2B1,
                         paniniA1.5B1, paniniPortraitA2B1,
                         paniniPortraitA1.5B1, mercator, transverseMercator]
            wave_correct_kind(str), default = no
                options:[horiz, vert, auto, no]
        """
        self.try_use_gpu = try_use_gpu
        self.blend_strength = blend_strength
        self.block_size = block_size
        self.nr_feeds = nr_feeds
        self.match_conf = match_conf
        self.blender_type = blender_type
        self.compensator = compensator
        self.detector = detector
        self.finder = finder
        self.matcher_type = matcher_type
        self.warper_type = warper_type
        self.wave_correct_kind = wave_correct_kind

    def get_stitcher(self) -> Stitcher:
        self.stitcher = Stitcher(detector=self.detector,
                                 finder=self.finder,
                                 blender_type=self.blender_type,
                                 blend_strength=self.blend_strength,
                                 matcher_type=self.matcher_type,
                                 wave_correct_kind=self.wave_correct_kind,
                                 compensator=self.compensator,
                                 nr_feeds=self.nr_feeds,
                                 block_size=self.block_size,
                                 warper_type=self.warper_type,
                                 try_use_gpu=self.try_use_gpu,
                                 match_conf=self.match_conf)
        return self.stitcher
