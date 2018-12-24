import cv2 as cv
import Components_pb2 as pbcomp
import numpy as np

import components_chain as chains
from TextDetection import chain_analysis


def text_localizing(id):
    components = pbcomp.Components()

    f = open("../protobins/component_IMG_" + id + ".bin", "rb")
    components.ParseFromString(f.read())
    f.close()

    pairs = chains.find_pairs(components)
    lines = chains.merge_chains(pairs)
    filename = '/home/laneesra/PycharmProjects/TextDetection/MSRA-TD500/train/IMG_' + id + '.JPG'

    img = cv.imread(filename, 1)
    img_copy = img
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
            if chain_analysis.filter(img_copy, rect):
                cv.drawContours(img, [box], 0, (0, 255, 0), 3)
                for let in line.letters:
                    cnt = np.array([(let.comp.minY, let.comp.minX), (let.comp.minY, let.comp.maxX), (let.comp.maxY, let.comp.maxX), (let.comp.maxY, let.comp.minX)])
                    rect = cv.minAreaRect(cnt)
                    box = cv.boxPoints(rect)
                    box = np.int0(box)
                    cv.drawContours(img, [box], 0, (0, 191, 255), 2)

    cv.imwrite('../result/detected_text' + id + '.jpg', img)
    cv.namedWindow('result', cv.WINDOW_NORMAL)
    cv.resizeWindow('result', 1000, 1000)
    cv.imshow('result', img)
    cv.waitKey()

