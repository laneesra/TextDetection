import cv2 as cv
import numpy as np

pi = 180


def filter(img, rect):
    box = cv.boxPoints(rect)
    box = np.int0(box)
    W = rect[1][0]
    H = rect[1][1]

    Xs = [i[0] for i in box]
    Ys = [i[1] for i in box]
    x1 = min(Xs)
    x2 = max(Xs)
    y1 = min(Ys)
    y2 = max(Ys)

    angle = rect[2]
    if angle < - pi / 4:
        angle += pi / 2

    center = ((x1 + x2) / 2, (y1 + y2) / 2)
    size = (x2 - x1, y2 - y1)
    M = cv.getRotationMatrix2D((size[0] / 2, size[1] / 2), angle, 1.0)
    cropped = cv.getRectSubPix(img, size, center)
    cropped = cv.warpAffine(cropped, M, size)
    cropped_W = H if H > W else W
    cropped_H = H if H < W else W
    text = cv.getRectSubPix(cropped, (int(cropped_W), int(cropped_H)), (size[0] / 2, size[1] / 2))

    gray = cv.cvtColor(text, cv.COLOR_BGR2GRAY)
    edges = cv.Canny(gray, 80, 160)
    min_line_length = 20
    max_line_gap = 5
    lines = cv.HoughLinesP(edges, 1, np.pi / 180, threshold=30, minLineLength=min_line_length, maxLineGap=max_line_gap)
    if lines is not None:
        if len(lines):
            return True
    return False