import numpy as np
import skimage.transform as skt
import scipy.ndimage.interpolation as sci
from scipy.ndimage import zoom
import cv2
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage import map_coordinates
import random

def my_3d_random_crop(z1, z2, y1, y2, x1, x2, crop_size):
    z = z2 - z1 - crop_size
    z_start = z1 + random.randint(0, z)
    z_end = z_start + crop_size
    y = y2 - y1 - crop_size
    y_start = y1 + random.randint(0, y)
    y_end = y_start + crop_size
    x = x2 - x1 - crop_size
    x_start = x1 + random.randint(0, x)
    x_end = x_start + crop_size
    return z_start, z_end, y_start, y_end, x_start, x_end