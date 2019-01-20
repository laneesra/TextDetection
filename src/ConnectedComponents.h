//
// Created by laneesra on 11.11.18.
//

#pragma once
#ifndef TEXTDETECTION_CONNECTEDCOMPONENTS_H
#define TEXTDETECTION_CONNECTEDCOMPONENTS_H

#include "StrokeWidthTransform.h"
#include "Components.pb.h"
#include <opencv2/opencv.hpp>
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/graph_traits.hpp>
#include <boost/graph/connected_components.hpp>
#include <boost/unordered_map.hpp>
#include <boost/graph/floyd_warshall_shortest.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/io.hpp>
#include <iostream>
#include <future>


class ConnectedComponents {
public:
    string filename;
    Components components;
    Components validComponents;
    cv::Mat SWTMatrix;
    cv::Mat connectedComponents;
    cv::Mat image;
    bool isDarkOnLight;

    ConnectedComponents(const string& filename, const cv::Mat& SWTMatrix, const cv::Mat& SWTMatrixNormU, const cv::Mat& image, bool isDarkOnLight);
    void execute(cv::Mat edge);
    void findComponentsBoost();
    void showAndSaveComponents();
    void firstStageFilter();
    void computeFeatures(cv::Mat& edge);
    void setValidComponent(Component* comp, int maxX, int minX, int maxY, int minY);
    void saveData();
    void markComponents();
    void improveComponentSWT(Component* comp, cv::Mat& morphImg);
};


#endif //TEXTDETECTION_CONNECTEDCOMPONENTS_H
