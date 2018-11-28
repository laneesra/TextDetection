//
// Created by laneesra on 11.11.18.
//

#pragma once
#ifndef TEXTDETECTION_CONNECTEDCOMPONENTS_H
#define TEXTDETECTION_CONNECTEDCOMPONENTS_H

#include "StrokeWidthTransform.h"
#include "Components.pb.h"

class ConnectedComponents {
public:
    string filename;
    Components components;
    Components valid_components;
   // Mat enqueued;
   // Mat num_of_component;
    Mat SWTMatrix;
    Mat connected_components;
    Mat camshift;
    Mat image;

    ConnectedComponents(string filename, Mat SWTMatrix, Mat SWTMatrixNormU, Mat image);
    void execute();
    void findComponents();
    void findComponentsBoost();
    void showAndSaveComponents();
    void firstStageFilter();
    void computeCamshiftFeatures();
    void setValidComponent(Component comp, int maxX, int minX, int maxY, int minY);
};


#endif //TEXTDETECTION_CONNECTEDCOMPONENTS_H
