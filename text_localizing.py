from math import sqrt

import cv2 as cv
import Components_pb2 as pbcomp
import numpy as np

import components_chain as chains


def text_localizing(id):
    components = pbcomp.Components()

    f = open("/home/laneesra/CLionProjects/TextDetection/protobins/component_IMG_" + id + ".bin", "rb")
    components.ParseFromString(f.read())
    f.close()

    pairs = chains.find_pairs(components)
    print(len(pairs))
    lines = chains.merge_chains(pairs)
   # filename = '/home/laneesra/PycharmProjects/TextDetection/MSRA-TD500/test/IMG_' + id + '.JPG'
    filename = '/home/laneesra/PycharmProjects/TextDetection/MSRA-TD500/train/IMG_' + id + '.JPG'

    B, G, R = cv.split(cv.imread(filename, 1))
    img = cv.merge((R, G, B))
    print(len(lines))
    for i, line in enumerate(lines):
        if len(line.letters):
            cnt = []
            for let in line.letters:
                cnt.append((let.comp.minY, let.comp.minX))
                cnt.append((let.comp.minY, let.comp.maxX))
                cnt.append((let.comp.maxY, let.comp.maxX))
                cnt.append((let.comp.maxY, let.comp.minX))
            rect = cv.minAreaRect(np.array(cnt))
            box = cv.boxPoints(rect)
            box = np.int0(box)
            cv.drawContours(img, [box], 0, (0, 255, 0), 3)
            for let in line.letters:
                cnt = np.array([(let.comp.minY, let.comp.minX), (let.comp.minY, let.comp.maxX), (let.comp.maxY, let.comp.maxX), (let.comp.maxY, let.comp.minX)])
                rect = cv.minAreaRect(cnt)
                box = cv.boxPoints(rect)
                box = np.int0(box)
                cv.drawContours(img, [box], 0, (0, 191, 255), 2)
    cv.imwrite('detected_text' + id+ '.jpg', img)
    cv.namedWindow('chains', cv.WINDOW_NORMAL)
    cv.resizeWindow('chains', 1000, 1000)
    cv.imshow('chains', img)
    cv.waitKey()

