"""
This is temporary file to test/find the best way to run the
program, using the sections of the original code with the
best performance. This will be deleted at a later stage, but
might be used as guidelines when designing the final version
of the label reader.
"""
import cv2
import numpy as np
# unique from main:
#import os
#import shutil
# unique from image_flattening:
from sklearn.linear_model import RANSACRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
# unique frm image_manager and text_reader:
from groundingdino.util.inference import Model
from segment_anything import sam_model_registry, SamPredictor
#from typing import List
import pytesseract as pt
import supervision as sv
import torch
# unique from panorama_manager:
#from stitching import Stitcher, stitching_error


class VideoFlow:
    """
    This class manages the flow of the frames and decides what frames
    to create panorama from.
    """
    def __init__(self, video_path: str):
        self.video_path = video_path
        self.previous_frame = None
        self.saved_count = 0
        self.panorama_manager = PanoramaManager()
        self.panorama = None
        self.memory = None
        self.pandiff = False
        self.pandiffs = []
        self.failed_diffs = []
        self.start_video()

    def start_video(self):
        self.interval = 30
        self.frame_n = 0
        self.last_saved_frame_n = 30
        self.diff_threshold = 75000000
        self.capture = cv2.VideoCapture(self.video_path)
        while self.capture.isOpened():
            print(f'frame nr: {self.frame_n}, interval: {30}')
            ret, frame = self.capture.read()
            if not ret:
                break
            if cv2.waitKey(25) & 0xFF == 27:
                break
            if self.frame_n - self.last_saved_frame_n >= 30:
                print(self.frame_n, self.last_saved_frame_n)
                self.check_image(frame)
            if isinstance(self.panorama, np.ndarray):
                to_show = cv2.resize(self.panorama,
                                     (self.panorama.shape[0]//2,
                                      self.panorama.shape[1]//2))
                cv2.imshow('frame', to_show)
                #cv2.imwrite('FinalPanorama.png', self.panorama)
            self.frame_n += 1
            print('')
            former_frame = frame
        print('test_final_print...')
        self.diff_threshold = 0
        self.check_image(former_frame)
        self.capture.release()
        print(self.pandiffs)
        print('failed:')
        print(f'mean: {np.mean(self.failed_diffs)}\nmedian: {np.median(self.failed_diffs)}\nmax: {max(self.failed_diffs)}\nmin: {min(self.failed_diffs)}')
        cv2.destroyAllWindows()

    def check_image(self, frame: np.ndarray):
        """
        check frame and try to add it to panorama.

        :param frame: input frame- numpy array- 3 channel.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if not self.check_blur(gray):
            print('too blurry')
            return False
        print('blur check ok')
        if not self.check_difference(gray):
            #print('not different enough')
            self.reset_to_previous()
            return False
        print('difference high enough')
        frame_enhanced = self.enhance_frame(frame)
        panorama_success = self.panorama_manager.add_image(frame_enhanced)
        print(type(panorama_success))
        if not panorama_success:
            print('panorama fail.')
            self.reset_to_previous()
            self.diff_threshold -= self.diff_threshold//20
            print(f'diff: {self.pandiff}')
            print(type(self.pandiff))
            if isinstance(self.pandiff, np.uint64):
                self.failed_diffs.append(int(self.pandiff))
            return False
        # modify diff threshold
        if isinstance(panorama_success, (int, float)):
            # remove later.
            self.pandiffs.append(self.pandiff)
            print(f'threshold modification-{panorama_success}')
            modifier = round((panorama_success-0.5)*20)  # int val -10 to 10
            max_change = round(self.diff_threshold*0.5)  # change of <= 50/100
            # do more drastic thresh changes closer to -100 and 100.
            # temporary modification:
            # TODO: update/modify this with accelerating solution.
            print(f'change: {round(max_change*((modifier*abs(modifier))*0.01))}/{max_change}')
            print(f'{self.diff_threshold}->{self.diff_threshold+round(max_change*((modifier*abs(modifier)*0.01)))}')
            self.diff_threshold = round(max_change*((modifier*abs(modifier))*0.01))
        else:
            if isinstance (self.pandiff, (int, float)):
                print('add to failed')
                input('run???')
                self.failed_diffs.append(self.pandiff)
            self.diff_threshold = round(self.diff_threshold*0.66)

        print('panorama succeeded')
        self.previous_frame = frame
        self.last_saved_frame_n = self.frame_n
        self.panorama = self.panorama_manager.panorama
        return True

    def check_difference(self, frame: np.ndarray) -> bool:
        """
        Check if image difference threshold is reached, and
        save image if so.

        :param frame: np.ndarray- grayscale image to compare.
        :output: True if image is different enough.
        """

        threshold = self.diff_threshold
        upper_threshold = round(threshold + (threshold/10)*2)
        if isinstance(self.panorama, np.ndarray):
            x = frame.shape[0]/self.panorama.shape[0]
            pan_s = self.panorama.copy()
            pan_resized = cv2.resize(pan_s,
                                     (round(self.panorama.shape[1]*x),
                                      frame.shape[0]))
            pan_resized = cv2.cvtColor(pan_resized,
                                       cv2.COLOR_BGR2GRAY)
            pan_gs = pan_resized[:,
                                 pan_resized.shape[1]-frame.shape[1]:
                                 pan_resized.shape[1]]
            self.pandiff = cv2.absdiff(pan_gs, frame).sum()
            print(f'diff  to  panorama:  {self.pandiff}')
            cv2.imwrite('compare_pan.png', pan_gs)
            cv2.imwrite('compare_frame.png', frame)

        if self.previous_frame is not None:
            previous_frame_gs = cv2.cvtColor(self.previous_frame,
                                             cv2.COLOR_BGR2GRAY)
            diff = cv2.absdiff(previous_frame_gs, frame).sum()
            print(previous_frame_gs.shape)
            print(frame.shape)
            if diff > threshold:
                self.memory = self.previous_frame
                self.saved_count += 1
                self.previous_frame = frame
                print(f'diff_score: {diff}')
                if diff >= upper_threshold:
                    self.interval = round((self.interval/5)*4)
                return True
            self.interval += 3
            print(f'not different enough ({diff}/{self.diff_threshold})')
            return False

        self.previous_frame = frame
        self.saved_count += 1
        return True

    def reset_to_previous(self) -> None:
        """
        Reset images to previous state.
        """
        if self.memory is None:
            return
        self.saved_count -= 1
        self.previous_frame = self.memory

    @staticmethod
    def check_blur(frame: np.ndarray,
                   threshold: float = 400) -> bool:
        """
        Check if frame is sharp enough to stitch.

        :param frame: np.ndarray, image(grayscale) to check blur on.
        :param threshold: float, optional-defaults to 300.
        :output: True if not to blurry.
        """
        laplacian_var = cv2.Laplacian(frame, cv2.CV_64F).var()
        return laplacian_var < threshold

    @staticmethod
    def enhance_frame(frame: np.ndarray) -> np.ndarray:
        """
        Enhance frame using Clahe.

        :param frame: numpy array, 3 channel image
        :output: 3 channel numpy array, enhanced.
        """
        frame2 = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        clahe = cv2.createCLAHE(clipLimit=1.7, tileGridSize=(5, 5))
        frame_planes = list(cv2.split(frame2))
        frame_planes[0] = clahe.apply(frame_planes[0])
        frame2 = cv2.merge(frame_planes)
        frame2 = cv2.cvtColor(frame2, cv2.COLOR_LAB2BGR)
        return frame2


class PanoramaManager:
    def __init__(self):
        self.image_flattener = ExtractImage()
        self.panorama = None
        self.images = []

    def add_image(self, image) -> np.ndarray | bool:
        """
        Add new image to panorama.

        :param image: numpy array of image
        :output: true or false, depending on success.
        """
        # TODO: manage error:
        #       (-215:Assertion failed) !err.empty() in function update
        image = self.image_flattener.return_flat(image)
        if image is False:
            print('problem w flattening')
            return False
        if self.panorama is None:
            print('new panorama base')
            self.panorama = image
            return True
        return self.stitch_to_panorama(image)
        """if self.stitch_to_panorama(image):
            return True
        return False"""

    def stitch_to_panorama(self, image):
        """
        Attempt to stitch previous frame/panorama to new image.

        :param image: numpy array of 3-channel image
        :output: True or False, depending on success.
        """
        # TODO: test stitcher settings.
        masks = self.get_masks([self.panorama, image])
        stitcher = cv2.Stitcher.create(cv2.STITCHER_SCANS)
        #stitcher.setCompositingResol(0)
        stitcher.setInterpolationFlags(cv2.INTER_LANCZOS4)
        #stitcher.setRegistrationResol(0)
        #stitcher.setSeamEstimationResol(-1)
        stitcher.setWaveCorrection(cv2.detail.WAVE_CORRECT_HORIZ)
        # status, new_panorama = stitcher.stitch([self.panorama, image], masks)
        """if isinstance(new_panorama, np.ndarray):
            self.panorama = cv2.addWeighted(new_panorama, 0.5,
                                            new_panorama, 0.5, 0)
            cv2.imwrite('progress_images/FinalPanorama.png', self.panorama)
            print('Set Higher Value For difference')
            return True"""
        conf_vals = [1.0, 0.95,  0.9, 0.85, 0.8, 0.75, 0.7, 0.65,
                     0.6, 0.55, 0.5, 0.45, 0.4, 0.35, 0.3]
        for value in conf_vals:
            print(value)
            stitcher.setPanoConfidenceThresh(value)
            status, new_panorama = stitcher.stitch([self.panorama, image],
                                                   masks)
            if status == cv2.Stitcher_OK:
                # check stitching is ok:
                max_width_increase = self.panorama.shape[1]+(image.shape[1]*0.95)
                max_height_increase = self.panorama.shape[0]*1.1
                if all([new_panorama.shape[0] < max_height_increase,
                        new_panorama.shape[1] < max_width_increase]):
                    self.panorama = cv2.addWeighted(new_panorama, 0.5,
                                                    new_panorama, 0.5, 0)
                    print(f'stitcher succeeded at confidence {value}')
                    cv2.imwrite('FinalPanorama.png', self.panorama)
                    if value < 0.5:
                        value = 0.5
                    return round((value-0.5)*2, 2)
                print('stitching size problem...')
                print(f'panorama: {self.panorama.shape}\nNew panorama: {new_panorama.shape}')
        return False

    @staticmethod
    def get_masks(images: list) -> list:
        """
        Create masks for each image, highlighting the text.

        :param images: list(of np.arrays)-images to get masks for. 3 chanel.
        :output: list of black and white masks.
        """
        masks = []
        for img in images:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            boxes = pt.image_to_boxes(gray, config='--psm 5')
            mask = np.zeros_like(img, dtype=np.uint8)
            h, w, _ = img.shape
            for box in boxes.splitlines():
                b = box.split()
                x, y, x2, y2 = int(b[1]), int(b[2]), int(b[3]), int(b[4])
                y = h - y
                y2 = h - y2
                cv2.rectangle(mask, (x, y2), (x2, y), (255, 255, 255), -1)
            if len(boxes.splitlines()) == 0:
                mask = np.ones_like(img, dtype=np.unit8)
                print('no text found-? white mask.')
                input('---')
            # mask edge (where flattening & artifacts tend to worse)
            mask[:, :round(mask.shape[1]*0.075)] = (0, 0, 0)
            mask[:, -round(mask.shape[1]*0.075):] = (0, 0, 0)
            mask[:round(mask.shape[0]*0.075), :] = (0, 0, 0)
            mask[-round(mask.shape[0]*0.075):, :] = (0, 0, 0)
            masks.append(mask)
        cv2.imwrite('mask_1.png', masks[0])
        cv2.imwrite('mask_2.png', masks[1])
        print(f'm1: {masks[0].shape}\nm2: {masks[1].shape}')
        # Hide where no overlap if expected
        if masks[0].shape[1] > masks[1].shape[1]:
            black_area = round((masks[0].shape[1] - masks[1].shape[1])*0.75)
            first_mask = masks[0]
            first_mask[:, :black_area] = (0, 0, 0)
            masks[0] = first_mask
        return masks


class ExtractImage:
    """
    Use Grounding Dino and SAM to flatten the
    interesting area.
    """
    def __init__(self):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        sam_encoder_version = 'vit_h'
        sam_checkpoint_path = 'weights/sam_vit_h_4b8939.pth'
        dino_dir = '.venv/lib/python3.11/site-packages/groundingdino/'
        self.focal_length = False
        self.dino_model = Model(
            f'{dino_dir}config/GroundingDINO_SwinT_OGC.py',
            f'{dino_dir}weights/groundingdino_swint_ogc.pth')
        self.classes = ['Text or image']  # ['text or image']
        self.box_threshold = 0.25  # 0.35
        self.text_threshold = 0.25
        self.sam_model = sam_model_registry[sam_encoder_version](
            checkpoint=sam_checkpoint_path).to(device=device)
        self.sam_predictor = SamPredictor(self.sam_model)
        self.flattener = FlattenImage()

    def return_flat(self, image: np.ndarray) -> np.ndarray | bool:
        roi = self.find_label(image)
        if isinstance(roi, np.ndarray):
            flat_img = self.flattener.new_image(roi)
            return flat_img
        return False

    def find_label(self, frame):
        """
        Returns image with label extracted (everything ex ROI is 0.0.0)
        """
        detections = self.dino_model.predict_with_classes(
            image=frame,
            classes=self.enhance_class_names(),
            box_threshold=self.box_threshold,
            text_threshold=self.text_threshold)
        detections.mask = self.segment_label(frame, detections.xyxy)
        mask_annotator = sv.MaskAnnotator()
        empty_image = np.zeros((frame.shape), dtype=np.uint8)
        roi_mask = cv2.cvtColor(mask_annotator.annotate(scene=empty_image,
                                                        detections=detections,
                                                        opacity=1),
                                cv2.COLOR_BGR2GRAY)
        _, binary_mask = cv2.threshold(roi_mask, 5, 255, cv2.THRESH_BINARY)
        cv2.imwrite('mask.png', roi_mask)
        contours, _ = cv2.findContours(binary_mask, 1, 2)
        if len(contours) >= 1:
            x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
            img_to_depth = cv2.bitwise_and(frame,
                                           cv2.cvtColor(binary_mask,
                                                        cv2.COLOR_GRAY2RGB),
                                           binary_mask)
            img_to_depth = img_to_depth[y:y+h, x:x+w]
        if isinstance(img_to_depth, np.ndarray):
            return img_to_depth
        return False

    def segment_label(self, frame, xyxy: np.ndarray) -> np.ndarray:
        """
        Use GroundedSAM to detect and get mask for text/label area
        of image.
        """
        self.sam_predictor.set_image(frame, image_format='RGB')
        result_masks = []
        for box in xyxy:
            masks, scores, _ = self.sam_predictor.predict(
                box=box, multimask_output=False)
            index = np.argmax(scores)
            result_masks.append(masks[index])
        return np.array(result_masks)

    def enhance_class_names(self) -> list[str]:
        """
        Enhances class names by specifying prompt details.
        Returns updated list.
        """
        return [
            f"{class_name}"
            for class_name
            in self.classes
        ]


class FlattenImage:
    """
    Class to manage flattening of images
    TODO: functionality of sorting out/ignore 'bad' detections.
    """
    def __init__(self):
        pass

    def new_image(self, frame: np.ndarray) -> np.ndarray | bool:
        """
        Attempt to match left and right edges of ROI to ml model.

        :param frame: numpy array of the image to create map from.
        :output: frame flattened if successfull, else False.
        """
        masked_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_bgra = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)
        self.frame = self.correct_image(frame=frame_bgra, masked=masked_img)
        return self.frame

    def correct_image(self, frame: np.ndarray,
                      masked: np.ndarray) -> np.ndarray:
        """
        use estimated size to correct the images shape and perspective

        :param frame: BGRA image
        :param masked: GRAYSCALE image
        TODO error management addition.
        """
        self.map_a, self.map_b = self.get_flattening_maps(masked)
        cv2.imwrite('map_b_vertical.png', self.map_b)
        cv2.imwrite('map_a_horizontal.png', self.map_a)
        flattened_image = cv2.remap(frame, self.map_a, self.map_b,
                                    interpolation=cv2.INTER_NEAREST,
                                    borderMode=cv2.BORDER_WRAP)

        gray = cv2.cvtColor(flattened_image, cv2.COLOR_BGRA2GRAY)
        flattened_image = cv2.cvtColor(flattened_image, cv2.COLOR_BGRA2BGR)
        _, mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY_INV)
        flattened_image = self.inpaint_img(flattened_image, mask)
        cv2.imwrite('flat_img.png', flattened_image)
        self.frame = flattened_image
        return self.frame

    def get_flattening_maps(self, masked: np.ndarray):
        """
        Method to create 2 maps for flattening/remapping.

        :param masked: 2 dimensional numpy array where all except ROI is 0.
        :return: 2 arrays of same shape as masked, describing re-map values.
        """
        map_base = masked
        map_base_rotated = cv2.rotate(masked, cv2.ROTATE_90_COUNTERCLOCKWISE)

        edge_points = self.get_edge_points(map_base)
        edge_points = self.manage_outliers(edge_points)
        edge_points_rotated = self.get_edge_points(map_base_rotated)
        edge_points_rotated = self.manage_outliers(edge_points_rotated)

        pixel_map_a, masked = self.get_maps(True,
                                            edge_points,
                                            len(map_base[0]), masked,
                                            (255, 255, 0), (0, 0, 255))
        pixel_map_b, masked = self.get_maps(True,
                                            edge_points_rotated,
                                            len(map_base_rotated[0]), masked,
                                            (255, 0, 255), (0, 255, 0))
        cv2.imwrite('points.png', masked)

        pixel_map_a = np.array(pixel_map_a).astype(np.float32)
        pixel_map_b = cv2.rotate(np.array(pixel_map_b),
                                 cv2.ROTATE_90_CLOCKWISE).astype(np.float32)
        return pixel_map_a, pixel_map_b

    def get_edge_points(self, map_base: np.ndarray,
                        reverse: bool = False) -> list:
        """
        returns [list of (min/start, max/end) for each row.]
        indexes of where ROI of each row of the map start and end.
        Return as list of pairs for each row, starting with lowest value.
        """
        edge_points = []
        for pixel_row in map_base:
            roi = [idx_nr for idx_nr, pix in enumerate(pixel_row) if pix > 0]
            if len(roi) < 1:
                edge_points.append(edge_points[-1])
            else:
                edge_points.append((min(roi), max(roi)))
        if reverse:
            edge_points = [pair[::-1] for pair in edge_points]
        return edge_points

    @staticmethod
    def manage_outliers(edge_points: list):
        """
        Replace outlier points with RANSAC regressor and np.linspace.

        :param  edge_points: List of where  edges where detected
        :output: List where outlier edges have been replaced.
        """
        edges = [[edge_point[0] for edge_point in edge_points],
                 [edge_point[1] for edge_point in edge_points]]
        for point_index, points in enumerate(edges):
            first = points[:len(points)//3]
            second = points[len(points)//3:(len(points)//3)*2]
            third = points[(len(points)//3)*2:]
            x = np.linspace(0, len(points), len(points))
            X = x[:, np.newaxis]
            if np.median(first) < np.median(second) > np.median(third):
                model = make_pipeline(PolynomialFeatures(2), RANSACRegressor())
                model.fit(X, points)
                points = model.predict(X)
            elif np.median(first) > np.median(second) < np.median(third):
                model = make_pipeline(PolynomialFeatures(2), RANSACRegressor())
                model.fit(X, points)
                points = model.predict(X)
            else:
                points = np.linspace(np.mean(first),
                                     np.mean(third), len(points))
            edges[point_index] = points

        filtered = []
        for point_1, point_2 in zip(edges[0], edges[1]):
            filtered.append((point_1, point_2))
        return filtered

    def get_maps(self, first_value: bool,
                 pixel_pairs: list,
                 len_active: int, image: np.ndarray,
                 color_1: tuple, color_2: tuple):
        """
        generate the actual maps.
        TODO: make this take width/difference between numbers into account.
              for roi length index at the opposite direction.
        """
        pixel_map = []
        for row_idx, (start, stop) in enumerate(pixel_pairs):
            pixel_map.append((np.linspace(start, stop, len_active)))
            if first_value:
                location_start = (int(start), row_idx)
                location_stop = (int(stop), row_idx)
            else:
                location_start = (row_idx, int(start))
                location_stop = (row_idx, int(stop))

            masked = cv2.circle(image, location_start,
                                1, color_1, 1)
            masked = cv2.circle(masked, location_stop,
                                1, color_2, 1)
        return pixel_map, masked

    def inpaint_img(self, img, mask):
        img = cv2.inpaint(img, mask, 3, cv2.INPAINT_TELEA)
        return img


VideoFlow('videos/test_video_2.mp4')
