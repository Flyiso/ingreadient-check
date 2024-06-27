"""
File for test if depth estimation could help with
correcting the frames.

diff of max & min in label area = save, to not run this all the time
run this once or a few times? when?
"""
import numpy as np
import cv2
import heapq
from transformers import pipeline
from PIL import Image
import csv


class DepthCorrection:
    """
    Class that finds depth and
    correct the image's perspective/flatten it.
    init sets min/max diff
    method just correct by values?
    """
    def __init__(self, frame: np.ndarray) -> None:
        self.X, self.Y = (False, False)
        frame_bgra = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)
        alpha_channel = np.where(
            (frame[:, :, 0] == 0) &
            (frame[:, :, 1] == 0) &
            (frame[:, :, 2] == 0),
            0,
            255
        )
        frame_bgra[:, :, 3] = alpha_channel
        cv2.imwrite('masked.png', frame_bgra)
        frame = Image.fromarray(frame_bgra)
        pipe = pipeline(task="depth-estimation",
                        model="LiheYoung/depth-anything-large-hf")
        depth = np.array(pipe(frame)["depth"])
        depth = cv2.cvtColor(depth, cv2.COLOR_BGR2BGRA)
        depth[:, :, 3] = alpha_channel

        cv2.imwrite('depth.png', depth)

        _, _, _, a = cv2.split(depth)
        mask = a > 0
        gray = cv2.cvtColor(depth[:, :, :3],
                            cv2.COLOR_BGRA2GRAY)
        depth = np.where(mask, gray, 255)
        depth = cv2.bitwise_not(depth)
        min_val = np.min(depth[mask])
        max_val = np.max(depth[mask])
        print(min_val, max_val)
        print(np.mean(depth[mask]))
        print(np.median(depth[mask]))
        print('....')

        cv2.imwrite('depth_gs.png', depth)
        self.correct_image(frame=frame_bgra, depth_mask=depth)

    def _estimate_depth(self, frame, mask):
        pass

    def correct_image(self, frame: np.ndarray,
                      depth_mask: np.ndarray) -> np.ndarray:
        """
        use estimated size to correct
        the images shape and perspective
        frame: BGRA image
        depth_mask: GRAYSCALE image
        """
        frame = frame  # bottle_image
        depth_mask = cv2.normalize(depth_mask, None, 0, 255, cv2.NORM_MINMAX)

        assert depth_mask.shape == frame.shape[:2], '''
                Depth image and bottle image must be of the same dimensions.'''

        cv2.imwrite('d_msk.png', depth_mask)
        height, width = depth_mask.shape
        map_a = np.array(
            [self.get_map_row(depth_mask[y])
             for y in range(height)]).astype(np.float32)
        map_b = np.transpose(np.array(
            [self.get_map_row([w_vals[x]
             for w_vals in depth_mask])
             for x in range(0, width)])).astype(np.float32)
        map_a = cv2.normalize(map_a, None, 0, 255, cv2.NORM_MINMAX)
        map_b = cv2.normalize(map_b, None, 255, 0, cv2.NORM_MINMAX)
        map_a = cv2.blur(map_a, (95, 95))
        map_b = cv2.blur(map_b, (95, 95))
        #map_a = cv2.medianBlur(map_a, 5)
        #map_b = cv2.medianBlur(map_b, 5)
        #map_a = cv2.GaussianBlur(map_a, (95, 95), 100)
        #map_b = cv2.GaussianBlur(map_b, (95, 95), 100)
        #map_a = cv2.bilateralFilter(map_a, 9, 75, 75)
        #map_b = cv2.bilateralFilter(map_b, 9, 75, 75)
        #map_a = cv2.bitwise_xor(map_a)
        #map_b = cv2.bitwise_xor(map_b)
        cv2.imwrite('map_a.png', map_a)
        cv2.imwrite('map_b.png', map_b)
        map_a = cv2.normalize(map_a, None, 0, width, cv2.NORM_MINMAX)
        map_b = cv2.normalize(map_b, None, 0, height, cv2.NORM_MINMAX)

        flattened_image = cv2.remap(frame, map_a, map_b,
                                    interpolation=cv2.INTER_LANCZOS4,
                                    borderMode=cv2.BORDER_WRAP)
        cv2.imwrite('flat_img.png', flattened_image)
        self.frame = flattened_image

    def get_map_row(self, pixel_row):
        """
        uses self.distribute to get values for map.
        """
        #map_a = self.distribute(pixel_row
        #                        [:((len(pixel_row))//2)])
        #map_b = self.distribute(pixel_row[(len(pixel_row)//2)::])
        #print('......')
        #print(pixel_row)
        #print(map_a)
        #print(map_b)
        #if len(map_a+map_b) == len(pixel_row):
        #    fin_map = map_b[::-1]+map_a
        #    return fin_map
        map_row = self.distribute_pixels(pixel_row)
        if len(map_row) == len(pixel_row):
            return map_row
        else:
            print(len(pixel_row)//2)
            print(len(pixel_row) % 2)
            print(f'img_section_original: {len(pixel_row)}')
            print(f'img_section_new_size: {len(map_a + map_b)}')
            print('.......')

    def distribute_pixels(self, pixels) -> np.ndarray:
        """
        returns the pixels with their new placement.
        figures out what method to use to distribute and uses it.
        """
        non_zero = [n for n in pixels if n < 0]

        if sum(non_zero[len(non_zero)//3:
                        (len(non_zero)//3)*2]
               ) < sum(non_zero[:len(non_zero)//3]) and\
            sum(non_zero[len(non_zero)//3:
                         (len(non_zero)//3)*2]
                ) < sum(non_zero[(len(non_zero//3))*2:]):
            self.X = pixels  # Remove when plotting/test is done
            pixels = self.distribute_surface(pixels)
            self.X_NEW = pixels  # Remove when plotting/test is done

        else:
            self.Y = pixels  # Remove when plotting/test is done
            pixels = self.distribute_perspective(pixels)
            self.Y_NEW = pixels  # Remove when plotting/test is done
        pixels = np.array(pixels)
        pixels = cv2.normalize(pixels, None, 0, len(pixels))

        # Temporary code to get csv_file for comparing methods.
        if isinstance(self.X, dict) and isinstance(self.Y, dict):
            val_dict = {'X': list(self.X), 'X_new': list(self.X_NEW),
                        'Y': list(self.Y), 'Y_NEW': list(self.Y_NEW)}
            self.Y_NEW = cv2.normalize(self.Y_NEW, None, 0, 255,
                                       cv2.NORM_MINMAX)
            self.X_NEW = cv2.normalize(self.X_NEW, None, 0, 255,
                                       cv2.NORM_MINMAX)
            with open('numbers.csv', 'w') as csvfile:
                writer = csv.DictWriter(csvfile,
                                        fieldnames=['X', 'X_NEW', 'Y', 'Y_NEW'])
                writer.writeheader()
                writer.writerows(val_dict)
            input('end...?')
        # End of temporary block.
        return pixels

    def distribute_surface(self, pixels) -> list:
        """
        distribute pixels, assuming middle of pixels are closer to camera
        to lowest value.
        """
        return_map = []
        multipliers = np.linspace(0, 2, len(pixels))
        for pixel_id, (pixel, multiplier) in enumerate(zip(pixels, multipliers)):
            return_map.append(pixel_id-(pixel*multiplier))
        return return_map

    def distribute_perspective(self, pixels) -> list:
        """
        Distribute perspective when assuming one end of
        pixel row is closer to camera.
        """
        return_map = []
        for pixel_id, (pixel_value,
                       pixel_multiplier) in enumerate(zip(
                           pixels, np.linspace(0, len(pixels)))):
            return_map.append(pixel_id-(pixel_value*pixel_multiplier))
        return return_map

    def distribute(self, pixels):
        """
        pixels: depth pixels of half of the height
                or width currently working with
        """
        #pixel_a = np.array(pixels)
        pixel_a = cv2.normalize(pixels, None, 0, 255, cv2.NORM_MINMAX)
        with open('numbers.csv', 'w') as csvfile:
            writer = csv.DictWriter(csvfile,
                                    fieldnames=['X', 'Y1', 'Y2', 'Y3'])
            writer.writeheader()
            n_s = []
            x = []
            y = []
            n_dict = []
            for nu, pix in enumerate(pixel_a):
                x.append(nu)
                y.append(pix)
                n_s.append(pix)
                n_dict.append({'X': nu, 'Y1': pix[-1]})
            n_s.sort()
            dy_dx = np.zeros_like(y, dtype=float)
            for i in range(len(x)):
                if i == 0:  # Forward difference for the first point
                    dy_dx[i] = (y[i+1] - y[i]) / (x[i+1] - x[i])
                elif i == len(x) - 1:  # Backward difference for the last point
                    dy_dx[i] = (y[i] - y[i-1]) / (x[i] - x[i-1])
                else:  # Central difference for interior points
                    dy_dx[i] = (y[i+1] - y[i-1]) / (x[i+1] - x[i-1])

            dy_dx = list(dy_dx)
            dy_dx.sort()
            for n_r, d_n, der_v in zip(n_s, n_dict, dy_dx):
                d_n['Y2'] = n_r[-1]
                d_n['Y3'] = der_v[0]
            writer.writerows(n_dict)
        input('')
