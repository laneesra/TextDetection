//
// Created by laneesra on 11.11.18.
//

#include "ConnectedComponents.h"

ConnectedComponents::ConnectedComponents(Mat SWTMatrix, Mat SWTMatrixNormU) : SWTMatrix(SWTMatrix) {
    enqueued = Mat(SWTMatrix.size[0], SWTMatrix.size[1], CV_8UC1, Scalar(0));
    numOfComponent = Mat(SWTMatrix.size[0], SWTMatrix.size[1], CV_32FC1, Scalar(-1));
    connectedComponents = Mat(SWTMatrix.size[0], SWTMatrix.size[1], CV_8UC3);
    cvtColor(SWTMatrixNormU, connectedComponents, COLOR_GRAY2BGR);
}
/*

void ConnectedComponents::findComponents() {
    queue<Point2d> q;
    int count = 0;
    for (int row = 0; row < SWTMatrix.size[0]; row++) {
        for (int col = 0; col < SWTMatrix.size[1]; col++) {
            if (enqueued.at<uchar>(row, col) == 0) {
                q.push(Point2d(row, col));
                enqueued.at<uchar>(row, col) = 1;
                components.emplace_back();

                while (!q.empty()) {
                    Point2d pixel = q.back();
                    q.pop();
                    auto shade = SWTMatrix.at<float>(pixel.x, pixel.y);
                    Point2d neighbors[] = {Point2d(pixel.x, pixel.y - 1), Point2d(pixel.x - 1, pixel.y),
                                           Point2d(pixel.x, pixel.y + 1), Point2d(pixel.x + 1, pixel.y)};

                    for (auto neigh : neighbors) {
                        if (0 <= neigh.x && neigh.x < SWTMatrix.size[0] && 0 <= neigh.y && neigh.y < SWTMatrix.size[1]) {
                            auto neigh_shade = SWTMatrix.at<float>(neigh.x, neigh.y);

                            if (neigh_shade > 0 && shade > 0 && 0.33f < neigh_shade / shade && neigh_shade / shade <= 3) {
                                if (enqueued.at<uchar>(neigh.x, neigh.y) == 0) {
                                    q.push(neigh);
                                    enqueued.at<uchar>(neigh.x, neigh.y) = 1;
                                }
                            }
                        }
                    }
                    components[count].points.push_back(SWTPoint(pixel.x, pixel.y, shade));
                }
                count++;
            }
        }
    }
}*/

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
                                    vector<SWTPoint> points;
                                    points.push_back(SWTPoint(neigh.x, neigh.y, neigh_shade));
                                    points.push_back(SWTPoint(pixel.x, pixel.y, shade));
                                    numOfComponent.at<float>(neigh.x, neigh.y) = components.size();
                                    numOfComponent.at<float>(pixel.x, pixel.y) = components.size();
                                    components.emplace_back(points);
                                } else if (numOfComponent.at<float>(pixel.x, pixel.y) < 0) {
                                    components[numOfComponent.at<float>(neigh.x, neigh.y)].points.push_back(
                                            SWTPoint(pixel.x, pixel.y, shade));
                                    numOfComponent.at<float>(pixel.x, pixel.y) = numOfComponent.at<float>(neigh.x, neigh.y);
                                } else {
                                    components[numOfComponent.at<float>(pixel.x, pixel.y)].points.push_back(
                                            SWTPoint(neigh.x, neigh.y, shade));
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


void ConnectedComponents::showComponents() {
    for (auto comp : components) {
        if (comp.points.size() > 50) {
            int maxY = 0, maxX = 0;
            int minY = SWTMatrix.size[0];
            int minX = SWTMatrix.size[1];

            for (auto pixel : comp.points) {
                maxY = max(maxY, pixel.y);
                maxX = max(maxX, pixel.x);
                minY = min(minY, pixel.y);
                minX = min(minX, pixel.x);
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
