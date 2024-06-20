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
        #warper = cv2.PyRotationWarper("cylindrical", 1.0)
        #_, warped_image = warper.warp(src=frame_cv2,
        #                              K=depth, R=depth,
        #                              interp_mode=cv2.INTER_LINEAR,
        #                              border_mode=cv2.BORDER_REFLECT)

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
        mask: 
        """
        frame = frame  # bottle_image
        depth_mask = cv2.normalize(depth_mask, None, 0, 255, cv2.NORM_MINMAX)

        # Ensure the images are of the same size
        assert depth_mask.shape == frame.shape[:2], '''
                Depth image and bottle image must be of the same dimensions.'''
        #print(frame[:, :, 3])
        # Normalize depth image to [0, 1] range
        #depth_normalized = cv2.normalize(depth_mask, None,
        #                                 alpha=0, beta=1,
        #                                 norm_type=cv2.NORM_MINMAX,
        #                                 mask=frame[:, :, 3])
        #depth_normalized = 1 - depth_normalized
        #cv2.imwrite('depth_norm.png', depth_normalized)
        #print(depth_normalized)
        #input('')
        cv2.imwrite('d_msk.png', depth_mask)
        # Create a transformation map based on depth information
        height, width = depth_mask.shape
        map_x = np.zeros((height, width), np.float32)
        map_y = np.zeros((height, width), np.float32)
        max_displacement = width / 2
        max_displacement_h = height / 2
        x = 0
        #for x in range(width):
        #    self.get_map_row(depth_mask[x])
        for y in range(height):
            map_y = self.get_map_row([w_vals[y] for w_vals in depth_mask])
            [map_y[n] for n in range(len(width))]
        for y in range(height):
            for x in range(width):
                #print(f'x: {depth_mask[x]}')
                #print(f'y: {[w_vals[y] for w_vals in depth_mask]}')
                #lft = sum(width[:width//2]) # sum of depth.
                #rght = sum(width[width//2:])
                #(width//2-x)  # distance to center width.
                #(height//2-y)  # distance to center height
                #(width//2)  # maximum width distance
                #(height//2)  # maximum distance height
                # add tol until max 225, then spread all to width//2
                # get displacement maps from middle distance & depth_mask?
                # depth_mask increase values for push and pull?
                # skip add all 0 to closest corner? or edge? or ignore them?
                # use some distribution to get how much to adjust?
                # sum limit of values to add close to each other? lots of 0
                # does not change anything, but high values cant be close? 
                # some neighbor limit? (max 225 in 3*3 ngh? or similar?)
                # go with max then? or use mean(median better?)
                # just fill a list?
                displacement = depth_mask[y, x] * max_displacement
                displacement_h = depth_mask[y, x] * max_displacement_h
                map_x[y, x] = x - displacement * x / width
                print(f'x: {x}, y: {y} --- {map_x[y, x]}')
                map_y[y, x] = y - displacement_h * y / height
                print(f'x: {x}, y: {y} --- {map_y[y, x]}')
                print('............')
            #input('.....')
        # Apply the transformation map to the whole image
        flattened_image = cv2.remap(frame, map_x, map_y,
                                    interpolation=cv2.INTER_LANCZOS4,
                                    borderMode=cv2.BORDER_WRAP)  #,  # was cv2.INTER_LINEAR
                                    #  borderMode=cv2.BORDER_TRANSPARENT)

        cv2.imwrite('flat_img.png', flattened_image)
        self.frame = flattened_image

    def get_map_row(self, pixel_row):
        """
        uses self.distribute to get values for map.
        """
        map_a = self.distribute(pixel_row[:(len(pixel_row))//2])
        map_b = self.distribute(pixel_row[(len(pixel_row)//2)::-1])[-1]
        if len(map_a+map_b) == len(pixel_row):
            fin_map = map_a+map_b
            return fin_map
        else:
            print(f'img_section_original: {len(pixel_row)}')
            print(f'img_section_new_size: {len(map_a+map_b)}')

    def distribute(self, pixels):
        """
        pixels: depth pixels of half of the height
                or width currently working with
        """
        len(pixels)  # how many spaces to fill.

        pxl_groups = []
        curr_group = {'pixel_ids': [], 'pxl_sum': 0, 'pxl_space': 0}
        for pxl_id, pxl in enumerate(pixels):
            if curr_group['pxl_sum'] + pxl <= 225:
                curr_group['pxl_sum'] += pxl
                curr_group['pixel_ids'].append(pxl_id)
            else:
                pxl_groups.append(curr_group)
                curr_group = {'pixel_ids': pxl_id, 'pxl_sum': pxl, 'spaces': 0}

        pxl_space = (len(pixels)) // (len(pxl_groups))
        pxl_remain = (len(pixels)) % (len(pxl_groups))
        remain_ids = heapq.nlargest(pxl_remain,
                                    range(len(pxl_groups)),
                                    key=lambda i: pxl_groups[i]['pxl_sum'])
        return_map = []
        used_ids = 0
        #  each g_id = one new value in return map.
        for g_id, group in enumerate(pxl_groups):
            total = pxl_space
            if g_id in remain_ids:
                total += 1
            to_add = total/(len(group['pixel_ids']))
            for n in range(1, len(group['pixel_ids']+1)):
                return_map.append(round(used_ids+(to_add*n)))
            used_ids += total

        if len(return_map) == len(pixels):
            return return_map
        print(f'img_section_original: {len(pixels)}')
        print(f'img_section_new_size: {len(return_map)}')
