"""
File for test if depth estimation could help with
correcting the frames.

diff of max & min in label area = save, to not run this all the time
run this once or a few times? when?
"""
import numpy as np
import cv2
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
        for y in range(height):
            for x in range(width):
                print(f'x: {depth_mask[x]}')
                print(f'y: {[w_vals[y] for w_vals in depth_mask]}')
                print('.....')
                input('')
                (width//2-x)  # distance to center width.
                (height//2-y)  # distance to center height
                (width//2)  # maximum width distanc
                (height//2)  # maximum distance height
                # get displacement maps from middle distance & depth_mask?
                # depth_mask increase values for push and pull?
                # skip add all 0 to closest corner? or edge? or ignore them?
                # use some distrobution to get how much to adjust? 
                displacement = depth_mask[y, x] * max_displacement
                displacement_h = depth_mask[y, x] * max_displacement_h
                map_x[y, x] = x + displacement * x / width
                map_y[y, x] = y + displacement_h * y / height
        # Apply the transformation map to the whole image
        flattened_image = cv2.remap(frame, map_x, map_y,
                                    interpolation=cv2.INTER_LANCZOS4,
                                    borderMode=cv2.BORDER_WRAP)  #,  # was cv2.INTER_LINEAR
                                    #  borderMode=cv2.BORDER_TRANSPARENT)

        cv2.imwrite('flat_img.png', flattened_image)
        self.frame = flattened_image
