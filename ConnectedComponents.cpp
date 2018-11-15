//
// Created by laneesra on 11.11.18.
//
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/graph_traits.hpp>
#include <boost/graph/connected_components.hpp>
#include <boost/unordered_map.hpp>
#include <boost/graph/floyd_warshall_shortest.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/io.hpp>
#include "ConnectedComponents.h"

using namespace boost;


ConnectedComponents::ConnectedComponents(Mat SWTMatrix, Mat SWTMatrixNormU) : SWTMatrix(SWTMatrix) {
    enqueued = Mat(SWTMatrix.size[0], SWTMatrix.size[1], CV_8UC1, Scalar(0));
    numOfComponent = Mat(SWTMatrix.size[0], SWTMatrix.size[1], CV_32FC1, Scalar(-1));
    connectedComponents = Mat(SWTMatrix.size[0], SWTMatrix.size[1], CV_8UC3);
    cvtColor(SWTMatrixNormU, connectedComponents, COLOR_GRAY2BGR);
}


void ConnectedComponents::findComponents() {
    queue<Point2d> q;
    for (int row = 0; row < SWTMatrix.size[0]; row++) {
        for (int col = 0; col < SWTMatrix.size[1]; col++) {
            if (enqueued.at<uchar>(row, col) == 0) {
                q.push(Point2d(row, col));
                enqueued.at<uchar>(row, col) = 1;
                while (!q.empty()) {
                    Point2d pixel = q.back();
                    q.pop();
                    auto shade = SWTMatrix.at<float>(pixel.x, pixel.y);
                    Point2d neighbors[] = {Point2d(pixel.x, pixel.y - 1), Point2d(pixel.x - 1, pixel.y),
                                            Point2d(pixel.x, pixel.y + 1), Point2d(pixel.x + 1, pixel.y)};
                    enqueued.at<uchar>(pixel.x, pixel.y) = 1;

                    for (auto neigh : neighbors) {
                        if (0 <= neigh.x && neigh.x < SWTMatrix.size[0] && 0 <= neigh.y && neigh.y < SWTMatrix.size[1]) {
                            auto neigh_shade = SWTMatrix.at<float>(neigh.x, neigh.y);

                            if (neigh_shade > 0 && shade > 0 && (shade / neigh_shade < 3 || neigh_shade / shade < 3)) {
                                if (numOfComponent.at<float>(neigh.x, neigh.y) < 0 && numOfComponent.at<float>(pixel.x, pixel.y) < 0) {
                                    vector<Point2d> points;
                                    points.emplace_back(neigh.x, neigh.y);
                                    points.emplace_back(pixel.x, pixel.y);
                                    numOfComponent.at<float>(neigh.x, neigh.y) = components.size();
                                    numOfComponent.at<float>(pixel.x, pixel.y) = components.size();
                                    components.emplace_back(points);
                                } else if (numOfComponent.at<float>(pixel.x, pixel.y) < 0) {
                                    components[numOfComponent.at<float>(neigh.x, neigh.y)].points.emplace_back(pixel.x, pixel.y);
                                    numOfComponent.at<float>(pixel.x, pixel.y) = numOfComponent.at<float>(neigh.x, neigh.y);
                                } else {
                                    components[numOfComponent.at<float>(pixel.x, pixel.y)].points.emplace_back(neigh.x, neigh.y);
                                    numOfComponent.at<float>(neigh.x, neigh.y) = numOfComponent.at<float>(pixel.x, pixel.y);
                                }

                            }
                        }
                    }
                }
            }
        }
    }
}


void ConnectedComponents::findComponentsBoost() {
    boost::unordered_map<int, int> map;
    boost::unordered_map<int, Point2d> reverseMap;

    typedef adjacency_list<boost::vecS, boost::vecS, boost::undirectedS> Graph;
    int num_vertices = 0;
    for (int row = 0; row < SWTMatrix.rows; row++) {
        for (int col = 0; col < SWTMatrix.cols; col++) {
            if (SWTMatrix.at<float>(row, col) > 0) {
                map[row * SWTMatrix.cols + col] = num_vertices;
                Point2d p(row, col);
                reverseMap[num_vertices] = p;
                num_vertices++;
            }
        }
    }

    Graph g(num_vertices);

    for (int row = 0; row < SWTMatrix.rows; row++) {
        for (int col = 0; col < SWTMatrix.cols; col++) {
            float swt = SWTMatrix.at<float>(row, col);
            if (swt > 0) {
                int this_pixel = map[row * SWTMatrix.cols + col];
                if (col + 1 < SWTMatrix.cols) {
                    float right = SWTMatrix.at<float>(row, col + 1);
                    if (right > 0 && (swt / right <= 3.0 || right / swt <= 3.0))
                        add_edge(this_pixel, map.at(row * SWTMatrix.cols + col + 1), g);
                }
                if (col - 1 >= 0) {
                    float left = SWTMatrix.at<float>(row , col - 1);
                    if (left > 0 && (swt / left <= 3.0 || left / swt <= 3.0))
                        add_edge(this_pixel, map.at(row * SWTMatrix.cols + col - 1), g);
                }
                if (row + 1 < SWTMatrix.rows) {
                    float upper = SWTMatrix.at<float>(row + 1, col);
                    if (upper > 0 && (swt / upper <= 3.0 || upper / swt <= 3.0))
                        add_edge(this_pixel, map.at((row + 1) * SWTMatrix.cols + col), g);
                }
                if (row - 1 >= 0) {
                    float down = SWTMatrix.at<float>(row - 1, col);
                    if (down > 0 && (swt / down <= 3.0 || down / swt <= 3.0))
                        add_edge(this_pixel, map.at((row - 1) * SWTMatrix.cols + col), g);
                }
            }
        }
    }

    vector<int> c(num_vertices);

    int num_comp = connected_components(g, &c[0]);

    components.reserve(num_comp);
    for (int j = 0; j < num_comp; j++) {
        components.emplace_back();
    }
    for (int j = 0; j < num_vertices; j++) {
        Point2d p = reverseMap[j];
        (components[c[j]]).points.push_back(p);
    }

}



void ConnectedComponents::showComponents() {
    for (auto &comp : components) {
        if (comp.points.size() > 20) {
            int maxY = 0, maxX = 0;
            int minY = SWTMatrix.size[0];
            int minX = SWTMatrix.size[1];

            for (Point2d &pixel : comp.points) {
                maxY = max(maxY, (int)pixel.y);
                maxX = max(maxX, (int)pixel.x);
                minY = min(minY, (int)pixel.y);
                minX = min(minX, (int)pixel.x);
            }

            int curPixel = minX;
            while (curPixel < maxX) {
                connectedComponents.at<Vec3b>(curPixel, maxY) = Vec3b(0, 100, 255);
                connectedComponents.at<Vec3b>(curPixel, minY) = Vec3b(0, 100, 255);
                curPixel++;
            }

            curPixel = minY;
            while (curPixel < maxY) {
                connectedComponents.at<Vec3b>(minX, curPixel) = Vec3b(0, 100, 255);
                connectedComponents.at<Vec3b>(maxX, curPixel) = Vec3b(0, 100, 255);
                curPixel++;
            }
        }
    }

    imshow("Connected components", connectedComponents);
    waitKey(0);
}
