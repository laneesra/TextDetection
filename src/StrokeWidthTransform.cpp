//
// Created by laneesra on 07.11.18.
//

#include "StrokeWidthTransform.h"

using namespace std;

StrokeWidthTransform::StrokeWidthTransform(const string& filename) : filename(filename) {
    image = cv::imread(filename);
    int width = image.size[1];
    int height = image.size[0];
    gray = cv::Mat(height, width, CV_8UC1);
    blurred = cv::Mat(height, width, CV_32FC1);
    gradientY = cv::Mat(height, width, CV_32FC1);
    gradientX = cv::Mat(height, width, CV_32FC1);
    SWTMatrix = cv::Mat(height, width, CV_32FC1, cv::Scalar(-1.));
    SWTMatrix_norm = cv::Mat(height, width, CV_32FC1);
    result = cv::Mat(height, width, CV_8UC1);
}

///executes whole process of building SWT_image
void StrokeWidthTransform::execute(bool darkOnLight) {
    edgeDetection();
    gradient();
    buildSWT(darkOnLight); // true if white text on dark background, else false
    medianFilter();
    normalizeImage(SWTMatrix, SWTMatrix_norm);
    convertScaleAbs(SWTMatrix_norm, result, 255, 0);
 //   showAndSaveSWT(darkOnLight);
}

///Canny edge detecion
void StrokeWidthTransform::edgeDetection() {
    cvtColor(image, gray, cv::COLOR_BGR2GRAY);
    blur(gray, gray, cv::Size(4, 4));
    /* Mat thresh;
    edge_threshold_high = threshold(gray, thresh, 0, 255, CV_THRESH_BINARY | CV_THRESH_OTSU);
    edge_threshold_low = edge_threshold_high * 0.5;
    cout << edge_threshold_low << " " << edge_threshold_high << endl; */

    //adaptiveThreshold(gray, edge, 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY, 11, 2);
    Canny(gray, edge, edge_threshold_low, edge_threshold_high, 3);
    filename = filename.substr(filename.size() - 12);

    /* imwrite("../images/" + filename + "_Canny.jpg", edge);
    imwrite("../images/" + filename + "_Canny.jpg", edge);
    imshow("edges", edge);
    waitKey(0); */
}

///computes gradient
void StrokeWidthTransform::gradient() {
    convertScaleAbs(gray, blurred, 1./255., 0);
    /*cuda::GpuMat grayGpu, blurredGpu, gradientXGpu, gradientYGpu;
    grayGpu.upload(gray);
    cuda::cvtColor(grayGpu, grayGpu, COLOR_GRAY2BGRA);

    Ptr<cuda::Filter> blurFilter = cuda::createGaussianFilter(grayGpu.type(), grayGpu.type(), Size(5, 5), 1);
    blurFilter->apply(grayGpu, blurredGpu);
    blurred = Mat(blurredGpu);
    Ptr<cuda::Filter> ScharrFilterX = cuda::createScharrFilter(grayGpu.type(), grayGpu.type(), 1, 0);
    Ptr<cuda::Filter> ScharrFilterY = cuda::createScharrFilter(grayGpu.type(), grayGpu.type(), 0, 1);*/

    blur(gray, blurred, cv::Size(5, 5));
    Scharr(blurred, gradientX, CV_32F, 1, 0);
    Scharr(blurred, gradientY, CV_32F, 0, 1);
    /*gradientXGpu.upload(gradientX);
    gradientYGpu.upload(gradientY);
    ScharrFilterX->apply(blurredGpu, gradientXGpu);
    ScharrFilterY->apply(blurredGpu, gradientYGpu);
    blurFilter->apply(gradientXGpu, gradientXGpu);
    blurFilter->apply(gradientYGpu, gradientYGpu);
    gradientX = Mat(gradientXGpu);
    gradientY = Mat(gradientYGpu);*/

    blur(gradientX, gradientX, cv::Size(5, 5));
    blur(gradientY, gradientY, cv::Size(5, 5));
 //   imwrite("../images/" + filename + "_gradientX.jpg", gradientX);
 //   imwrite("../images/" + filename + "_gradientY.jpg", gradientY);
 //   imshow("Gradient X : Scharr", gradientX);
 //   imshow("Gradient Y : Scharr", gradientY);
}

///methid for showing result and saving it 
void StrokeWidthTransform::showAndSaveSWT(bool darkOnLight) {
    if (darkOnLight) {
        imwrite("../images/" + filename + "_SWT" + "_dark.jpg", result);
    } else {
        imwrite("../images/" + filename + "_SWT" + "_light.jpg", result);
    }
    imshow("SWT", result);
    cv::waitKey(0);
}

///second pass for SWT-image to detect possible mistakes on corners
void StrokeWidthTransform::medianFilter() {
    for (auto &ray : rays) {
        for (auto &point : ray.points) {
            int x = point.x;
            int y = point.y;
            point.SWT = SWTMatrix.at<float>(y, x);
        }
        sort(ray.points.begin(), ray.points.end(), [](const SWTPoint &lhs, const SWTPoint &rhs) -> bool {
                return lhs.SWT < rhs.SWT;
        });
        float median = ray.points[ray.points.size() / 2].SWT;
        for (auto &point : ray.points) {
            point.SWT = min(median, point.SWT);
        }
    }
}

///main method for building SWT-image
void StrokeWidthTransform::buildSWT(bool dark_on_light) {
    float prec = .05;
    for (int row = 0; row < edge.rows; row++) {
        for (int col = 0; col < edge.cols; col++) {
            if (edge.at<uchar>(row, col) > 0) {
                Ray r;

                SWTPoint p(col, row);
                r.p = p;
                vector<SWTPoint> points;
                points.push_back(p);

                float curX = (float)col + 0.5f;
                float curY = (float)row + 0.5f;
                int curPixX = col;
                int curPixY = row;
                float G_x = gradientX.at<float>(row, col);
                float G_y = gradientY.at<float>(row, col);

                // normalize gradient
                float mag = sqrt((G_x * G_x) + (G_y * G_y));

                if (dark_on_light){
                    G_x = -G_x / mag;
                    G_y = -G_y / mag;
                } else {
                    G_x = G_x / mag;
                    G_y = G_y / mag;
                }

                while (true) {
                    curX += G_x * prec;
                    curY += G_y * prec;
                    if ((int)(floor(curX)) != curPixX || (int)(floor(curY)) != curPixY) {
                        curPixX = (int)(floor(curX));
                        curPixY = (int)(floor(curY));
                        if (curPixX < 0 || (curPixX >= edge.cols) || curPixY < 0 || (curPixY >= edge.rows)) {
                            break;
                        }

                        SWTPoint pnew(curPixX, curPixY);
                        points.push_back(pnew);

                        if (edge.at<uchar>(curPixY, curPixX) > 0) {
                            r.q = pnew;
                            float G_xt = gradientX.at<float>(curPixY, curPixX);
                            float G_yt = gradientY.at<float>(curPixY, curPixX);
                            mag = sqrt((G_xt * G_xt) + (G_yt * G_yt));
                            if (dark_on_light){
                                G_xt = -G_xt / mag;
                                G_yt = -G_yt / mag;
                            } else {
                                G_xt = G_xt / mag;
                                G_yt = G_yt / mag;
                            }
                            if (acos(G_x * -G_xt + G_y * -G_yt) <  M_PI / 2.) {
                                float length = sqrt(((float)r.q.x - (float)r.p.x)*((float)r.q.x - (float)r.p.x) + ((float)r.q.y - (float)r.p.y)*((float)r.q.y - (float)r.p.y));
                                for (auto &point : points) {
                                    int x = point.x;
                                    int y = point.y;
                                    if (SWTMatrix.at<float>(y, x) < 0) {
                                        SWTMatrix.at<float>(y, x) = length;
                                    } else {
                                        SWTMatrix.at<float>(y, x) = min(length, SWTMatrix.at<float>(y, x));
                                    }
                                }
                                r.points = points;

                                rays.push_back(r);
                            }
                            break;
                        }
                    }
                }
            }
        }
    }
}

///normalizing image for using by opencv
void StrokeWidthTransform::normalizeImage(const cv::Mat& input, cv::Mat& output) {
    float maxVal = 0;
    float minVal = 255;
    for (int row = 0; row < input.rows; row++) {
        for (int col = 0; col < input.cols; col++) {
            float pixel = input.at<float>(row, col);
            if (pixel < 0) {}
            else {
                maxVal = max(pixel, maxVal);
                minVal = min(pixel, minVal);
            }
        }
    }

    float difference = maxVal - minVal;
    for (int row = 0; row < input.rows; row++) {
        for (int col = 0; col < input.cols; col++) {
            float pixel = input.at<float>(row, col);
            if (pixel < 0) {
                output.at<float>(row, col) = 1;
            } else {
                output.at<float>(row, col) = (pixel - minVal) / difference;
            }
        }
    }
}
