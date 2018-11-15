//
// Created by laneesra on 11.11.18.
//

#ifndef TEXTDETECTION_CONNECTEDCOMPONENTS_H
#define TEXTDETECTION_CONNECTEDCOMPONENTS_H

#include "StrokeWidthTransform.h"

struct Component {
    vector<Point2d> points;

    Component() = default;

    explicit Component(vector<Point2d> points) : points(points){};
};

class ConnectedComponents {
public:
    vector<Component> components;
    Mat enqueued;
    Mat numOfComponent;
    Mat SWTMatrix;
    Mat connectedComponents;

    ConnectedComponents(Mat SWTMatrix, Mat SWTMatrixNormU);
    void findComponents();
    void findComponentsBoost();
    void showComponents();
};


#endif //TEXTDETECTION_CONNECTEDCOMPONENTS_H
