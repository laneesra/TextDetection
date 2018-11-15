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


ConnectedComponents::ConnectedComponents(string filename, Mat SWTMatrix, Mat SWTMatrix_norm_u) : SWTMatrix(SWTMatrix), filename(filename) {
    enqueued = Mat(SWTMatrix.size[0], SWTMatrix.size[1], CV_8UC1, Scalar(0));
   // num_of_component = Mat(SWTMatrix.size[0], SWTMatrix.size[1], CV_32FC1, Scalar(-1));
    connected_components = Mat(SWTMatrix.size[0], SWTMatrix.size[1], CV_8UC3);
    cvtColor(SWTMatrix_norm_u, connected_components, COLOR_GRAY2BGR);
}


void ConnectedComponents::execute() {
    //findComponents();
    findComponentsBoost();
    firstStageFilter();
    showAndSaveComponents();
}


void ConnectedComponents::findComponents() {
    queue<SWTPoint> q;
    for (int row = 0; row < SWTMatrix.size[0]; row++) {
        for (int col = 0; col < SWTMatrix.size[1]; col++) {
            if (enqueued.at<uchar>(row, col) == 0) {
                q.push(SWTPoint(row, col, num_of_component.at<float>(row, col)));
                enqueued.at<uchar>(row, col) = 1;
                while (!q.empty()) {
                    SWTPoint pixel = q.back();
                    q.pop();
                    auto swt = SWTMatrix.at<float>(pixel.x, pixel.y);
                    SWTPoint neighbors[] = {SWTPoint(pixel.x, pixel.y - 1), SWTPoint(pixel.x - 1, pixel.y),
                                            SWTPoint(pixel.x, pixel.y + 1), SWTPoint(pixel.x + 1, pixel.y)};
                    enqueued.at<uchar>(pixel.x, pixel.y) = 1;

                    for (auto &neigh : neighbors) {
                        if (0 <= neigh.x && neigh.x < SWTMatrix.size[0] && 0 <= neigh.y && neigh.y < SWTMatrix.size[1]) {
                            auto neigh_swt = SWTMatrix.at<float>(neigh.x, neigh.y);

                            if (neigh_swt > 0 && swt > 0 && (swt / neigh_swt < 3 || neigh_swt / swt < 3)) {
                                if (num_of_component.at<float>(neigh.x, neigh.y) < 0 && num_of_component.at<float>(pixel.x, pixel.y) < 0) {
                                    vector<SWTPoint> points;
                                    points.emplace_back(neigh.x, neigh.y, num_of_component.at<float>(neigh.x, neigh.y));
                                    points.emplace_back(pixel.x, pixel.y, num_of_component.at<float>(pixel.x, pixel.y));
                                    num_of_component.at<float>(neigh.x, neigh.y) = components.size();
                                    num_of_component.at<float>(pixel.x, pixel.y) = components.size();
                                    components.emplace_back(points);
                                } else if (num_of_component.at<float>(pixel.x, pixel.y) < 0) {
                                    components[(int)num_of_component.at<float>(neigh.x, neigh.y)].points.emplace_back(pixel.x, pixel.y, num_of_component.at<float>(pixel.x, pixel.y));
                                    num_of_component.at<float>(pixel.x, pixel.y) = num_of_component.at<float>(neigh.x, neigh.y);
                                } else {
                                    components[(int)num_of_component.at<float>(pixel.x, pixel.y)].points.emplace_back(neigh.x, neigh.y, num_of_component.at<float>(neigh.x, neigh.y));
                                    num_of_component.at<float>(neigh.x, neigh.y) = num_of_component.at<float>(pixel.x, pixel.y);
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
    boost::unordered_map<int, SWTPoint> reverse_map;

    typedef adjacency_list<boost::vecS, boost::vecS, boost::undirectedS> Graph;
    int num_vertices = 0;
    for (int row = 0; row < SWTMatrix.rows; row++) {
        for (int col = 0; col < SWTMatrix.cols; col++) {
            if (SWTMatrix.at<float>(row, col) > 0) {
                map[row * SWTMatrix.cols + col] = num_vertices;
                SWTPoint p(row, col, SWTMatrix.at<float>(row, col));
                reverse_map[num_vertices] = p;
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

    int num_comp = boost::connected_components(g, &c[0]);

    components.reserve(num_comp);
    for (int j = 0; j < num_comp; j++) {
        components.emplace_back();
    }
    for (int j = 0; j < num_vertices; j++) {
        components[c[j]].points.emplace_back(reverse_map[j]);
    }
}


void ConnectedComponents::firstStageFilter() {
    for (auto &comp : components) {
        int maxY = 0, maxX = 0;
        int minY = SWTMatrix.size[0];
        int minX = SWTMatrix.size[1];

        for (SWTPoint &pixel : comp.points) {
            maxY = max(maxY, pixel.y);
            maxX = max(maxX, pixel.x);
            minY = min(minY, pixel.y);
            minX = min(minX, pixel.x);
        }

        if (maxY != minY && maxX != minX && comp.isValid(maxX, minX, maxY, minY)) {
            valid_components.emplace_back(comp);
        }
    }
}


void ConnectedComponents::showAndSaveComponents() {
    for (auto &comp : components) {
        int curPixel = comp.minX;
        while (curPixel < comp.maxX) {
            connected_components.at<Vec3b>(curPixel, comp.maxY) = Vec3b(0, 100, 255);
            connected_components.at<Vec3b>(curPixel, comp.minY) = Vec3b(0, 100, 255);
            curPixel++;
        }

        curPixel = comp.minY;
        while (curPixel < comp.maxY) {
            connected_components.at<Vec3b>(comp.minX, curPixel) = Vec3b(0, 100, 255);
            connected_components.at<Vec3b>(comp.maxX, curPixel) = Vec3b(0, 100, 255);
            curPixel++;
        }
    }

    imshow("Connected components", connected_components);
    imwrite("../images/" + filename + "_connectedComponents.jpg", connected_components);
    waitKey(0);
}
