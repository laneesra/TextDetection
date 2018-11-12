//
// Created by laneesra on 11.11.18.
//

#ifndef TEXTDETECTION_CONNECTEDCOMPONENTS_H
#define TEXTDETECTION_CONNECTEDCOMPONENTS_H

#include "StrokeWidthTransform.h"

struct Component {
    vector<SWTPoint> points;

    Component(){};

    Component(vector<SWTPoint> points) : points(points){};
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
    void showComponents();
};


#endif //TEXTDETECTION_CONNECTEDCOMPONENTS_H
