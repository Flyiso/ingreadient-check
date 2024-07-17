"""
File for test if depth estimation could help with
correcting the frames.

diff of max & min in label area = save, to not run this all the time
run this once or a few times? when?
"""
import numpy as np
import cv2
from statistics import mean
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
        print(frame_bgra.shape)
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
        depth_mask = cv2.equalizeHist(depth_mask)
        depth_mask = cv2.normalize(depth_mask, None, 0, 255, cv2.NORM_MINMAX)

        assert depth_mask.shape == frame.shape[:2], '''
                Depth image and bottle image must be of the same dimensions.'''

        cv2.imwrite('d_msk.png', depth_mask)
        height, width = depth_mask.shape
        blr_h = height//5
        blr_w = width//5
        if blr_h % 2 == 0:
            blr_h += 1
        if blr_w % 2 == 0:
            blr_w += 1
        #depth_mask = cv2.medianBlur(depth_mask, 55)
        cv2.imwrite('d_msk_blr.png', depth_mask)
        depth_mask_long = cv2.GaussianBlur(depth_mask, (5, blr_h), 100)
        map_a = np.array([self.get_map_row(depth_mask_long[y], 'lat')
                          for y in range(height)]).astype(np.float32)

        # test map reverse.
        """reversed_map = np.zeros_like(map_a)
        for y in range(height):
            for x in range(width):
                # Get the value at the current pixel
                value = map_a[y, x]
                # Ensure the value is within the bounds of the map
                if 0 <= value < 256:
                    reversed_map[x, int(value)] = y
        map_a = reversed_map"""
        # end of test solution

        depth_mask_lat = cv2.GaussianBlur(depth_mask, (blr_w, 5), 200)
        map_b = np.transpose(np.array([
            self.get_map_row([w_vals[x] for w_vals in depth_mask_lat], 'long')
            for x in range(width)
            ]), (1, 0, 2)).astype(np.float32)

        #map_a = cv2.normalize(map_a, None, 0, 255, cv2.NORM_MINMAX)  # remove
        #map_b = cv2.normalize(map_b, None, 0, 255, cv2.NORM_MINMAX)  # remove
        #map_a = cv2.medianBlur(map_a, 35)
        #map_b = cv2.medianBlur(map_b, 35)
        #map_a = cv2.GaussianBlur(map_a, (5, blr_h), 100)
        #map_b = cv2.GaussianBlur(map_b, (blr_w, 5), 100)
        #map_a = cv2.bilateralFilter(map_a, 9, 75, 75)
        #map_b = cv2.bilateralFilter(map_b, 9, 75, 75)
        #map_a = cv2.bitwise_xor(map_a)
        #map_b = cv2.bitwise_xor(map_b)
        cv2.imwrite('map_a.png', map_a)
        cv2.imwrite('map_b.png', map_b)
        map_a = cv2.normalize(map_a, None, 0, width+1, cv2.NORM_MINMAX)
        map_b = cv2.normalize(map_b, None, 0, height+1, cv2.NORM_MINMAX)
        print('shapes:')
        print(map_a.shape)
        print(map_b.shape)
        print(frame.shape)

        flattened_image = cv2.remap(frame, map_a, map_b,
                                    interpolation=cv2.INTER_LINEAR,
                                    borderMode=cv2.BORDER_REPLICATE)
        cv2.imwrite('flat_img.png', flattened_image)
        self.frame = flattened_image

    def get_map_row(self, pixel_row, direction: str):
        """
        uses self.distribute to get values for map.
        """
        map_row = self.distribute_pixels(pixel_row, direction)
        if len(map_row) == len(pixel_row):
            return map_row
        else:
            print(pixel_row)
            print(map_row)

    def distribute_pixels(self, pixels, direction) -> np.ndarray:
        """
        returns the pixels with their new placement.
        figures out what method to use to distribute and uses it.
        """
        # calls wrong method sometimes?
        # send full img instead and transpose it?
        pixels = cv2.normalize(np.array(pixels), None, 0, 255, cv2.NORM_MINMAX)
        if direction == 'lat':
            pixels = self.distribute_surface(pixels)
        if direction == 'long':
            pixels = self.distribute_perspective(pixels)
        """
        non_zero = [n for n in pixels if n > 0]
        pixels = [int(pixel) for pixel in pixels]
            if sum(non_zero[len(non_zero)//3:
                        (len(non_zero)//3)*2]
               ) < sum(non_zero[:len(non_zero)//3]) and\
            sum(non_zero[len(non_zero)//3:
                         (len(non_zero)//3)*2]
                ) < sum(non_zero[(len(non_zero)//3)*2:]):
            pixels = self.distribute_surface(pixels)

        else:
            pixels = self.distribute_perspective(pixels)"""

        pixels = np.array(pixels)
        pixels = cv2.normalize(pixels, None, 0, len(pixels), cv2.NORM_MINMAX)
        return pixels

    def distribute_surface(self, pixels) -> list:
        """
        distribute pixels, assuming middle of pixels are closer to camera
        to lowest value.
        TODO:
        modify to get more space to high depth value
        This does the opposite of what wanted?
        """
        return_map = [0]
        # currently each pixel has 1 space
        pixels.mean()  # mean of pixels- one?
        pixels.min()   # give those no space?
        pixels.max()   # Give as much space as possible?
                       # pixel - mean = space modifier? 
                       # Do in 2 steps? one to get each pixel+mod
                       # and one to do the map?  

        for pixel, percentile in zip(pixels,
                                     np.linspace(-1, 1, len(pixels))):
            """return_map.append(return_map[-1]+((((pixel/len(pixels))*10) *
                                              ((pixel/len(pixels))*10)) *
                                              ((abs(percentile)*10)**3)))"""
            return_map.append(len(return_map)+(percentile*pixel))
        return return_map[1:]

    def distribute_perspective(self, pixels) -> list:
        """
        Distribute perspective when assuming one end of
        pixel row is closer to camera.
        """
        return_map = [0]
        for pixel_value, percentile in zip(
                           pixels, np.linspace(1, 2, len(pixels))):
            return_map.append(return_map[-1]+len(return_map)**pixel_value)
        return sorted(return_map[1:])

    def distribute(self, pixels):
        """
        pixels: depth pixels of half of the height
                or width currently working with
        """
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
