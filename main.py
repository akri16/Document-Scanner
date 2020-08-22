import cv2
import glob
import numpy as np
from pathlib import Path

IMG_PATH = "samples/1.jpeg"
img = cv2.imread(IMG_PATH)
img = cv2.resize(img, (1500, 880))
original = img.copy()




