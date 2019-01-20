//
// Created by laneesra on 07.11.18.
//

#pragma once
#ifndef TEXTDETECTION_STROKEWIDTHTRANSFORM_H
#define TEXTDETECTION_STROKEWIDTHTRANSFORM_H

#include <vector>
#include <opencv2/opencv.hpp>
#include <cassert>
#include <cmath>
#include <iostream>
#include <cmath>
#include <ctime>
#include <utility>
#include <algorithm>
#include <vector>

using namespace std;
//using namespace cv;

struct SWTPoint {
    int y;
    int x;
    float SWT;

    SWTPoint() = default;
    SWTPoint(int x, int y) : x(x), y(y), SWT(0) {}
    SWTPoint(int x, int y, float SWT) : x(x), y(y), SWT(SWT){}
/*
    bool operator<(const SWTPoint &other) const {
        pair<int, int> p1(this->x, this->y);
        pair<int, int> p2(other.x, other.y);
        return (p2 < p1);
    }*/

};

struct Ray {
    SWTPoint p;
    SWTPoint q;
    vector<SWTPoint> points;
};


class StrokeWidthTransform {
public:
    string filename;
    int edge_threshold_low = 80;
    int edge_threshold_high = edge_threshold_low * 2;
    cv::Mat image, gray, blurred, edge, gradientX, gradientY, SWTMatrix, SWTMatrix_norm, result;
    vector<Ray> rays;

    StrokeWidthTransform(const string& filename);
    void execute(bool dark_on_light);
    void edgeDetection();
    void gradient();
    void buildSWT(bool dark_on_light);
    void showAndSaveSWT(bool dark_on_light);
    void medianFilter();
    void normalizeImage(const cv::Mat& input, cv::Mat& output);
};


#endif //TEXTDETECTION_STROKEWIDTHTRANSFORM_H
