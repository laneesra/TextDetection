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
    Components components;
    Components validComponents;
    Mat SWTMatrix;
    Mat connectedComponents;
    Mat image;
    bool isDarkOnLight;

    ConnectedComponents(string filename, Mat& SWTMatrix, Mat& SWTMatrixNormU, Mat& image, bool isDarkOnLight);
    void execute(Mat edge);
    void findComponentsBoost();
    void showAndSaveComponents();
    void firstStageFilter();
    void computeFeatures(Mat& edge);
    void setValidComponent(Component* comp, int maxX, int minX, int maxY, int minY);
    void saveData();
    void markComponents();
    void improveComponentSWT(Component* comp, Mat& morphImg);
};


#endif //TEXTDETECTION_CONNECTEDCOMPONENTS_H
