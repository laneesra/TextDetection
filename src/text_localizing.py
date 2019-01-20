import cv2 as cv
import Components_pb2 as pbcomp
import numpy as np

import components_chain as chains
import chain_analysis


def text_localizing():
    components = pbcomp.Components()

    f = open("../protobins/components.bin", "rb")
    components.ParseFromString(f.read())
    f.close()

    pairs = chains.find_pairs(components)
    print len(pairs)
    lines = chains.merge_chains(pairs)
    filename = components.components[0].filename
    img = cv.imread(filename, 1)
    img_copy = img
   # text = open("text.txt", "a")
   # text.write(filename + "\n")

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
            '''for point in box:
                text.write(str(point[0]) + ' ' + str(point[1]) + ' ')
            text.write("\n")'''
            for let in line.letters:
                cnt = np.array([(let.comp.minY, let.comp.minX), (let.comp.minY, let.comp.maxX),
                                (let.comp.maxY, let.comp.maxX), (let.comp.maxY, let.comp.minX)])
                rect = cv.minAreaRect(cnt)
                box = cv.boxPoints(rect)
                box = np.int0(box)
                cv.drawContours(img, [box], 0, (0, 191, 255), 2)
  #  text.close()

    cv.imwrite('../result/detected_text.jpg', img)
    cv.namedWindow('result', cv.WINDOW_NORMAL)
    cv.resizeWindow('result', 1000, 1000)
    cv.imshow('result', img)
    cv.waitKey(0)

