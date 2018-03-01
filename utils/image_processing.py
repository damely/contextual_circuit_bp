"""Useful functions for processing images."""
import cv2
import numpy as np


def crop_center(img, crop_size):
    """Center crop images."""
    im_shape = img.shape
    x, y = im_shape[:2]
    cx, cy = crop_size[:2]
    x_check = x < cx
    y_check = y < cy
    if np.any([x_check, y_check]):
        return resize(img, crop_size)
    startx = x // 2 - (cx // 2)
    starty = y // 2 - (cy // 2)
    if len(im_shape) == 2:
        return img[starty:starty + cy, startx:startx + cx]
    elif len(im_shape) == 3:
        return img[starty:starty + cy, startx:startx + cx, :]
    else:
        raise NotImplementedError(
            'Cannot handle im size of %s' % len(im_shape))


def resize(img, new_size):
    """Resize image."""
    return cv2.resize(img, tuple(new_size[:2]))


def pad_square(img):
    """Pad rectangular image to square."""
    im_shape = img.shape[:2]
    target_size = np.max(im_shape)
    h_pad = target_size - im_shape[0]
    w_pad = target_size - im_shape[1]
    t = h_pad // 2
    b = t
    l = w_pad // 2
    r = l
    return cv2.copyMakeBorder(img, t, b, l, r, cv2.BORDER_CONSTANT, 0.)
