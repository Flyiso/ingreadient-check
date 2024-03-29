"""
File to store functions that display different attributes of image,
for example corners, edges, lines and more. 
File is here to make it easier to difference the methods that will be
in the final frame management from functions that are used to figure out
what goes in the final version of the frame management
"""

import numpy as np
import cv2


def crop_to_min_rectangle_with_lines(frame):
    # Read the input image

    # Convert the image to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Threshold the grayscale image to obtain a binary mask of non-black pixels
    _, mask = cv2.threshold(gray, 30, 250, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)

    contour_max = max(contours, key=cv2.contourArea)
    approx = []
    for epsilon in range(1, 100):
        # Approximate the contour to a simpler polygon
        epsilon = (epsilon / 100) * cv2.arcLength(contour_max, True)
        approx_c = cv2.approxPolyDP(contour_max, epsilon, True)
        # Draw the approximated contour on the image
        approx.append(approx_c)
    largest_approximation = max(approx, key=cv2.contourArea)
    frame = crop_outside_contour(frame, largest_approximation)
    cv2.drawContours(frame, [largest_approximation], -1, (0, 255, 0), 2)
    return frame


def crop_outside_contour(frame, largest_approximation):
    # Create a mask for the contour
    contour_mask = np.zeros_like(cv2.cvtColor(frame,  cv2.COLOR_RGB2GRAY))
    # Draw contours on the mask with a specific color (in BGR format)
    cv2.drawContours(contour_mask, [largest_approximation], -1,
                     (255), thickness=cv2.FILLED)
    image = cv2.bitwise_and(frame, frame,  mask=contour_mask)
    return image
