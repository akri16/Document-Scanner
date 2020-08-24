import cv2
import glob
import numpy as np
from pathlib import Path

IMG_PATH = "samples/1.jpeg"
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
#thr_img = cv2.dilate(final_image, np.ones((2, 2), np.uint8))
thr_img = cv2.adaptiveThreshold(final_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 19, 8)

cv2.imshow("Final image", final_image)
cv2.imshow("edged", edged_img)
cv2.imshow("thr", thr_img)
cv2.imshow("Original image", img)
cv2.waitKey(0)
