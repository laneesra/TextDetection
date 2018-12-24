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


ConnectedComponents::ConnectedComponents(string filename, Mat SWTMatrix, Mat SWTMatrix_norm_u, Mat image) : SWTMatrix(std::move(SWTMatrix)), filename(std::move(filename)), image(std::move(image)) {
   // enqueued = Mat(SWTMatrix.size[0], SWTMatrix.size[1], CV_8UC1, Scalar(0));
   // num_of_component = Mat(SWTMatrix.size[0], SWTMatrix.size[1], CV_32FC1, Scalar(-1));
    connected_components = Mat(SWTMatrix.size[0], SWTMatrix.size[1], CV_8UC3);
    cvtColor(SWTMatrix_norm_u, connected_components, COLOR_GRAY2BGR);
    camshift = connected_components;
}


void ConnectedComponents::execute() {
    //findComponents();
    findComponentsBoost();
    firstStageFilter();
    showAndSaveComponents();
    computeCamshiftFeatures();
    saveData();
}

/*
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
*/

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
                        add_edge(this_pixel, map.at(row *SWTMatrix.cols + col + 1), g);
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

    for (int j = 0; j < num_comp; j++) {
        components.add_components();
    }

    for (int j = 0; j < num_vertices; j++) {
      //  components[c[j]].points.emplace_back(reverse_map[j]);
        Component* comp = components.mutable_components(c[j]);
        auto points = comp->mutable_points();
        auto point = points->Add();
        point->set_x(reverse_map[j].x);
        point->set_y(reverse_map[j].y);
        point->set_swt(reverse_map[j].SWT);
    }
}


void ConnectedComponents::firstStageFilter() {
    for (auto &comp : components.components()) {
        int maxY = 0, maxX = 0;
        int minY = SWTMatrix.size[0];
        int minX = SWTMatrix.size[1];

        for (auto &pixel : comp.points()) {
            maxY = max(maxY, pixel.y());
            maxX = max(maxX, pixel.x());
            minY = min(minY, pixel.y());
            minX = min(minX, pixel.x());
        }

        if (maxY != minY && maxX != minX) {
            setValidComponent(comp, maxX, minX, maxY, minY);
        }
    }
}


void ConnectedComponents::showAndSaveComponents() {
    for (auto &comp : valid_components.components()) {
        int curPixel = comp.minx();
        while (curPixel < comp.maxx()) {
            connected_components.at<Vec3b>(comp.maxy(), curPixel) = Vec3b(0, 100, 255);
            connected_components.at<Vec3b>(comp.miny(), curPixel) = Vec3b(0, 100, 255);
            curPixel++;
        }

        curPixel = comp.miny();
        while (curPixel < comp.maxy()) {
            connected_components.at<Vec3b>(curPixel, comp.minx()) = Vec3b(0, 100, 255);
            connected_components.at<Vec3b>(curPixel, comp.maxx()) = Vec3b(0, 100, 255);
            curPixel++;
        }
    }

    imshow("Connected components", connected_components);
    imwrite("../images/" + filename + "_connectedComponents.jpg", connected_components);
    waitKey(0);
}

void ConnectedComponents::setValidComponent(Component comp, int maxX, int minX, int maxY, int minY) {
    float height = (float)maxX - minX + 1;
    float width = (float)maxY - minY + 1;
    auto q = (float)comp.points().size();

    float mean = 0;
    for (auto &point : comp.points()) {
        mean += point.swt();
    }
    mean /= q;

    float SD = 0;
    for (auto &point : comp.points()) {
        SD += (point.swt() - mean) * (point.swt() - mean);
    }
    SD /= q;
    SD = sqrt(SD);

    float WV = SD / mean;
    float AR = min(width / height, height / width);
    float OR = q / (width * height);
    bool valid = width != 0 && height != 0 && q > 10 && width*height < image.size[0]*image.size[1]*0.8 && 0 <= WV && WV <= 1 && 0.1 <= AR && AR <= 1 && 0.1 <= OR && OR <= 1;
    if (valid) {
        Component* val_comp = valid_components.add_components();
        val_comp->set_height(height);
        val_comp->set_width(width);
        val_comp->set_wv(WV);
        val_comp->set_ar(AR);
        val_comp->set_or_(OR);
        val_comp->set_maxx(maxY);
        val_comp->set_minx(minY);
        val_comp->set_maxy(maxX);
        val_comp->set_miny(minX);
    }
}


void ConnectedComponents::computeCamshiftFeatures() {
    Mat backproj, hsv, hue, mask, hist;
    cvtColor(image, hsv, COLOR_BGR2HSV);
    int vmin = 10, vmax = 256, smin = 30;
    inRange(hsv, Scalar(0, smin, vmin), Scalar(180, 256, vmax), mask);
    int ch[] = {0, 0};
    hue.create(hsv.size(), hsv.depth());
    mixChannels(&hsv, 1, &hue, 1, ch, 1);
    int hsize = 16;
    float hranges[] = {0, 180};
    const float *phranges = hranges;

    for (int i = 0; i < valid_components.components().size(); i++) {
        auto c = valid_components.mutable_components(i);
        Rect compWindow;
        compWindow.x = c->minx();
        compWindow.y = c->miny();
        compWindow.width = (int)c->width();
        compWindow.height = (int)c->height();
        compWindow &= Rect(0, 0, camshift.cols, camshift.rows);

        Mat roi(hue, compWindow), maskroi(mask, compWindow);
        calcHist(&roi, 1, nullptr, maskroi, hist, 1, &hsize, &phranges);
        normalize(hist, hist, 0, 255, NORM_MINMAX);
        calcBackProject(&hue, 1, nullptr, hist, backproj, &phranges);
        backproj &= mask;
        RotatedRect compBox = CamShift(backproj, compWindow,
                                       TermCriteria(TermCriteria::EPS | TermCriteria::COUNT, 10, 1));

        c->set_center_x(compBox.center.x);
        c->set_center_y(compBox.center.y);
        c->set_orientation(compBox.angle);
        c->set_characteristic_scale(compBox.size.width + compBox.size.height);
       
        ellipse(camshift, compBox, Scalar(0, 0, 255), 3, LINE_AA);
    }

    imshow("Camshift", camshift);
    imwrite("../images/" + filename + "_camshift.jpg", camshift);
    waitKey(0);
}


void ConnectedComponents::saveData() {
    const string result_file = "../components.bin";
    fstream output(result_file, ios::out | ios::binary);
    if (!valid_components.SerializeToOstream(&output)) {
        cerr << "Failed to serialize data" << endl;
    }
    output.close();
    google::protobuf::ShutdownProtobufLibrary();
}