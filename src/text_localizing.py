import math
from math import sqrt

import cv2 as cv
from skimage import morphology, measure
from sklearn.cluster import KMeans

import Components_pb2 as pbcomp
import numpy as np

import components_chain as chains


def final_text(filename, preds, lines, is_dark_on_light):
    final_lines = []
    if preds is not None:
        print preds
        for i, pred in enumerate(preds):
            if pred > 0:
                final_lines.append(lines[i])

        draw_bounding_box(filename, final_lines, is_dark_on_light)


def text_localizing(components, is_dark_on_light):
    filename = components.components[0].filename
    img = cv.imread(filename, 1)
    pairs = chains.find_pairs(components)
    lines = chains.merge_chains(pairs)
    extract_features(lines, img)
    draw_bounding_box(filename, lines, is_dark_on_light)

    return extract_features(lines, img)


def read_component(is_dark_on_light):
    components = pbcomp.Components()

    if is_dark_on_light:
        f = open("../protobins/components_dark.bin", "rb")
    else:
        f = open("../protobins/components_light.bin", "rb")

    components.ParseFromString(f.read())
    f.close()

    return components


def extract_features(lines, img):
    '''chain features: candidates count, average probability, average direction, size variation, distance variation, 
    average axial ratio, average density, average width variation, average color self-similarity'''
    cc = []
    ap = []
    sv = []
    dv = []
    aar = []
    ad = []
    awv = []
    c = []

    for i, line in enumerate(lines):
        if len(line.letters):
            line.candidate_count = len(line.letters)
            average_probability = 0
            size_variation = 0
            distance_variation = 0
            average_axial_ratio = 0
            average_density = 0
            average_width_variation = 0
            colors = 0
            average_dist = 0
            average_size = 0

            for let in line.letters:
                q = len(let.comp.points)
                S = let.comp.minor_axis + let.comp.major_axis
                average_probability += let.comp.proba
                average_size += let.comp.minor_axis * let.comp.major_axis
                average_density += float(q)/S**2
                average_width_variation += let.comp.WV
                average_axial_ratio += let.comp.AR

                min_dist = 0
                for let1 in line.letters:
                    dist = chains.dist(let.comp.center_x, let.comp.center_y, let1.comp.center_x, let1.comp.center_y)
                    if dist:
                        min_dist = min(dist, min_dist)
                let.dist = min_dist

                average_dist += min_dist

            average_probability /= line.candidate_count
            average_size /= line.candidate_count
            average_axial_ratio /= line.candidate_count
            average_density /= line.candidate_count
            average_width_variation /= line.candidate_count
            average_dist /= line.candidate_count
            for let in line.letters:
                size_variation += (let.comp.minor_axis * let.comp.major_axis - average_size) ** 2
                distance_variation += (let.dist - average_dist) ** 2

            size_variation /= line.candidate_count
            size_variation = sqrt(size_variation)
            distance_variation /= line.candidate_count
            distance_variation = sqrt(distance_variation)

            cnt = []
            for let in line.letters:
                cnt.append((let.comp.minY, let.comp.minX))
                cnt.append((let.comp.minY, let.comp.maxX))
                cnt.append((let.comp.maxY, let.comp.maxX))
                cnt.append((let.comp.maxY, let.comp.minX))
            rect = cv.minAreaRect(np.array(cnt))
            center, size, theta = rect
            center, size = tuple(map(int, center)), tuple(map(int, size))
            M = cv.getRotationMatrix2D(center, theta, 1)
            dst = cv.warpAffine(img, M, img.shape[:2])
            out = cv.getRectSubPix(dst, size, center)

            rows, cols, bands = out.shape
            X = out.reshape(rows * cols, bands)

            kmeans = KMeans(n_clusters=5, random_state=0).fit(X)
            labels = kmeans.labels_.reshape(rows, cols)

            for i in np.unique(labels):
                blobs = np.int_(morphology.binary_opening(labels == i))
                count = len(np.unique(measure.label(blobs))) - 1
                colors += count

            cc.append(line.candidate_count)
            ap.append(average_probability)
            sv.append(size_variation)
            dv.append(distance_variation)
            aar.append(average_axial_ratio)
            ad.append(average_density)
            awv.append(average_width_variation)
            c.append(colors)

    return cc, ap, sv, dv, aar, ad, awv, c, lines


def draw_bounding_box(filename, lines, is_dark_on_light):
    img = cv.imread(filename, 1)

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
                cnt = np.array([(let.comp.minY, let.comp.minX), (let.comp.minY, let.comp.maxX),
                                (let.comp.maxY, let.comp.maxX), (let.comp.maxY, let.comp.minX)])
                rect = cv.minAreaRect(cnt)
                box = cv.boxPoints(rect)
                box = np.int0(box)
                cv.drawContours(img, [box], 0, (0, 191, 255), 2)

    if is_dark_on_light:
        filename = '../results/detected_text_dark.jpg'
    else:
        filename = '../results/detected_text_light.jpg'
    cv.imwrite(filename, img)
    cv.namedWindow('result', cv.WINDOW_NORMAL)
    cv.resizeWindow('result', 1000, 1000)
    cv.imshow('result', img)
    cv.waitKey(0)
