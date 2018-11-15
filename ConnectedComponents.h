//
// Created by laneesra on 11.11.18.
//

#ifndef TEXTDETECTION_CONNECTEDCOMPONENTS_H
#define TEXTDETECTION_CONNECTEDCOMPONENTS_H

#include "StrokeWidthTransform.h"

struct Component {
    vector<SWTPoint> points;
    int maxX, minX, maxY, minY;
    float width;
    float height;
    float mean;
    float SD; // standard deviation

    // features for filtering
    float WV; // width variation range [0, 1]
    float AR; // aspect ratio range [0.1, 1]
    float OR; // occupation ratio range [0.1, 1]

    Component() = default;

    explicit Component(vector<SWTPoint> points) : points(points){};

    bool isValid(int maxX, int minX, int maxY, int minY) {
        this->maxX = maxX;
        this->minX = minX;
        this->maxY = maxY;
        this->minY = minY;

        width = (float)maxY - minY + 1;
        height = (float)maxX - minX + 1;

        float q = (float)points.size();


        mean = 0;
        for (auto &point : points) {
            mean += point.SWT;
        }
        mean /= q;

        SD = 0;
        for (auto &point : points) {
            SD += (point.SWT - mean) * (point.SWT - mean);
        }
        SD /= q;
        SD = sqrt(SD);

        WV = SD / mean;
        AR = min(width / height, height / width);
        OR = q / (width * height);

        bool valid = 0 <= WV && WV <= 1 && 0.1 <= AR && AR <= 1 && 0.1 <= OR && OR <= 1;
        return valid;
    }
};


class ConnectedComponents {
public:
    string filename;
    vector<Component> components;
    vector<Component> valid_components;
    Mat enqueued;
    Mat num_of_component;
    Mat SWTMatrix;
    Mat connected_components;

    ConnectedComponents(string filename, Mat SWTMatrix, Mat SWTMatrixNormU);
    void execute();
    void findComponents();
    void findComponentsBoost();
    void showAndSaveComponents();
    void firstStageFilter();
};


#endif //TEXTDETECTION_CONNECTEDCOMPONENTS_H
