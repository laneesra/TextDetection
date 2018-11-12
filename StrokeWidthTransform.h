//
// Created by laneesra on 07.11.18.
//

#ifndef TEXTDETECTION_STROKEWIDTHTRANSFORM_H
#define TEXTDETECTION_STROKEWIDTHTRANSFORM_H

#include <vector>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

struct SWTPoint {
    int x;
    int y;
    float SWT;

    SWTPoint(){}

    SWTPoint(int x, int y, float SWT) : x(x), y(y), SWT(SWT){}

};

struct Ray {
    SWTPoint p;
    SWTPoint q;
    vector<SWTPoint> points;
};


class StrokeWidthTransform {
public:
    int edgeThreshLow = 175;
    int edgeThreshHigh = 320;
    Mat image, gray, blurred, edge, gradientX, gradientY, SWTMatrix, SWTMatrixNorm, result;
    vector<Ray> rays;

    StrokeWidthTransform();
    void edgeDetection();
    void gradient();
    void buildSWT(bool dark_on_light);
    void showSWT();
    void medianFilter();
    void normalizeImage(Mat input, Mat output);
};


#endif //TEXTDETECTION_STROKEWIDTHTRANSFORM_H
