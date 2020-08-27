import cv2
import utils
import numpy as np
from PIL import Image, ImageEnhance

IMG_PATH = "samples/4.jpeg"
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

for c in contours:
    p = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.02 * p, True)

    if len(approx) == 4:
        target = approx
        break

approx = utils.rectify(target)
h, w = np.abs(approx[0] - approx[2])
height = 300
width = abs(w/h * height)
pts2 = np.float32([[0, 0], [400, 0], [400, 400], [0, 400]])
M = cv2.getPerspectiveTransform(approx, pts2)
final_image = cv2.warpPerspective(original, M, (400, 400))
cv2.drawContours(img, c, -1, (0, 255, 0), 5)

final_image_noise_free = cv2.fastNlMeansDenoisingColored(cv2.fastNlMeansDenoisingColored(final_image))
final_image_gray = cv2.cvtColor(final_image_noise_free, cv2.COLOR_BGR2GRAY)

#Removing Disconnectivities
morph = utils.open_close(final_image_noise_free, (1, 1))

img = utils.resize(img, 400)
im_output = utils.enhance(morph, 1.2, 8, 10)
magic_color = utils.gray_thresh(final_image_gray)
magic_color = cv2.fastNlMeansDenoising(magic_color)

cv2.imshow("Final image", final_image)
cv2.imshow("Final image enhanced", im_output)
cv2.imshow("Original image", img)
cv2.imshow("Magic Color", magic_color)
cv2.waitKey(0)
