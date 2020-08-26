import numpy as np
import cv2
import sys
from skimage.exposure import rescale_intensity

np.set_printoptions(threshold=sys.maxsize)


def rectify(h):
    h = h.reshape((4, 2))
    hnew = np.zeros((4, 2), dtype=np.float32)
    add = h.sum(1)
    hnew[0] = h[np.argmin(add)]
    hnew[2] = h[np.argmax(add)]

    diff = np.diff(h, axis=1)
    hnew[1] = h[np.argmin(diff)]
    hnew[3] = h[np.argmax(diff)]

    return hnew


def gray_thresh(img):
    thr_img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 19, 5)
    image = cv2.erode(thr_img, cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1)))
    return image


def enhance(image, contrast, brightness, sat):
    # Brightness and Contrasts
    image = np.clip(contrast * image + brightness, 0, 255).astype(np.uint8)

    #Intensity Rescaling
    channels = cv2.split(image)
    for i in range(3):
        channels[i] = rescale_intensity(channels[i], out_range=(0, 255))

    image = cv2.merge(channels).astype(np.uint8)

    # Saturation
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.uint16)
    image[:, :, 1] += sat
    image = image.astype(np.uint8)
    image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)

    # Sharpening
    k = np.array([[0, -1, 0],
                  [-1, 5, -1],
                  [0, -1, 0]])
    image = cv2.filter2D(image, -1, k)

    return image


def resize(img, h=None, w=None):
    if h is None and w is None:
        return img

    asp = img.shape[1] / img.shape[0]
    if h is None:
        h = w / asp

    elif w is None:
        w = h * asp

    return cv2.resize(img, (round(w), round(h)))

def open_close(img, ksize):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, ksize)
    morph = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    morph = cv2.morphologyEx(morph, cv2.MORPH_OPEN, kernel)
    return morph
