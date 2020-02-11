import cv2
from scipy.misc import imread
from PIL import Image
import numpy as np

parent_dir = "Z:/Jeffrey-Ede/models/stem-random-walk-nin-20-43/"

io_pairs = [(parent_dir+"truth-450000.tif", parent_dir+"blurred_truth-450000.tif"),
            (parent_dir+"truth-475000.tif", parent_dir+"blurred_truth-475000.tif")]

for (i, o) in io_pairs:
    img = imread(i, mode='F')
    img = cv2.GaussianBlur(img,(3,3), 1.5)
    Image.fromarray(img.astype(np.float32)).save(o)