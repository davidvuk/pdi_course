import matplotlib.pyplot as plt
import numpy as np
from skimage import filters, measure, exposure, feature, color
from skimage.filters import threshold_local, threshold_otsu, rank, sobel, gaussian
from skimage.feature import canny
from skimage.segmentation import slic
from skimage.measure import find_contours
from skimage.exposure import equalize_adapthist
from skimage.morphology import dilation, erosion, binary_erosion, binary_dilation
import cv2