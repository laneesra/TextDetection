//
// Created by laneesra on 11.11.18.
//

#pragma once
#ifndef TEXTDETECTION_CONNECTEDCOMPONENTS_H
#define TEXTDETECTION_CONNECTEDCOMPONENTS_H

#include "StrokeWidthTransform.h"
#include "Components.pb.h"
#include <opencv2/opencv.hpp>


class ConnectedComponents {
public:
    string filename;
    Components componentsDark;
    Components componentsLight;
    Components validComponents;
   // Mat enqueued;
   // Mat num_of_component;
    Mat SWTMatrixDark;
    Mat SWTMatrixLight;
    Mat connectedComponentsDark;
    Mat connectedComponentsLight;
    Mat image;

    ConnectedComponents(string filename, Mat SWTMatrixDark, Mat SWTMatrixDarkNormU, Mat SWTMatrixLight, Mat SWTMatrixLightNormU, Mat image);
    void execute();
    void findComponents();
    void findComponentsBoost(bool darkOnLight);
    void showAndSaveComponents();
    void firstStageFilter(bool darkOnLight);
    void computeCamshiftFeatures();
    void setValidComponent(Component* comp, int maxX, int minX, int maxY, int minY);
    void saveData();
    void markComponents();
    void improveComponentSWT(Component* comp, Mat morphImg, bool darkOnLight);
};


#endif //TEXTDETECTION_CONNECTEDCOMPONENTS_H
