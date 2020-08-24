import cv2
import glob
import numpy as np
from pathlib import Path

IMG_PATH = "samples/2.jpeg"
img = cv2.imread(IMG_PATH)
original = img.copy()

# Preprocessing
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blurred_img = cv2.GaussianBlur(gray_img, (5, 5), 0)

#Edged
edged_img = cv2.Canny(blurred_img, 10, 120)
original_edged = edged_img.copy()

#dialate
img_dial = cv2.dilate(edged_img, np.ones((5, 5), np.uint8), iterations=1)

# Find contours
contours, _ = cv2.findContours(img_dial, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
contours = sorted(contours, key=cv2.contourArea, reverse=True)


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


for c in contours:
    p = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.02 * p, True)

    if len(approx) == 4:
        target = approx
        break

print(contours)

approx = rectify(target)
pts2 = np.float32([[0, 0], [800, 0], [800, 800], [0, 800]])
M = cv2.getPerspectiveTransform(approx, pts2)
final_image = cv2.warpPerspective(original, M, (800, 800))

cv2.drawContours(img, contours, -1, (0, 255, 0), 5)
final_image = cv2.cvtColor(final_image, cv2.COLOR_BGR2GRAY)

#Thresholding
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
morph = cv2.morphologyEx(final_image, cv2.MORPH_CLOSE, kernel)
morph = cv2.morphologyEx(morph, cv2.MORPH_OPEN, kernel)
thr_img = cv2.adaptiveThreshold(morph, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 5)

cv2.imshow("Final image", final_image)
cv2.imshow("edged", edged_img)
cv2.imshow("thr", thr_img)
cv2.imshow("Original image", img)
cv2.waitKey(0)
