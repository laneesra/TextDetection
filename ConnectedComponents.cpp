

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
#include <iostream>

using namespace boost;
float PI = 3.14159265359;


ConnectedComponents::ConnectedComponents(string filename, Mat SWTMatrixDark, Mat SWTMatrixDarkNormU, Mat SWTMatrixLight, Mat SWTMatrixLightNormU, Mat image) : SWTMatrixDark(std::move(SWTMatrixDark)), SWTMatrixLight(std::move(SWTMatrixLight)), filename(std::move(filename)), image(std::move(image)) {
    //enqueued = Mat(SWTMatrix.size[0], SWTMatrix.size[1], CV_8UC1, Scalar(0));
    //num_of_component = Mat(SWTMatrix.size[0], SWTMatrix.size[1], CV_32FC1, Scalar(-1));
    connectedComponentsDark = Mat(image.size[0], image.size[1], CV_8UC3);
    connectedComponentsLight = Mat(image.size[0], image.size[1], CV_8UC3);

    cvtColor(SWTMatrixDarkNormU, connectedComponentsDark, COLOR_GRAY2BGR);
    cvtColor(SWTMatrixLightNormU, connectedComponentsLight, COLOR_GRAY2BGR);
}


void ConnectedComponents::execute() {
    //findComponents();
    findComponentsBoost(true);
    firstStageFilter(true);

    findComponentsBoost(false);
    firstStageFilter(false);
    //markComponents();
    showAndSaveComponents();
    computeFeatures();
    saveData();
}

/// Old and slow method for finding components
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

void ConnectedComponents::findComponentsBoost(bool darkOnLight) {
    boost::unordered_map<int, int> map;
    boost::unordered_map<int, SWTPoint> reverse_map;
    Mat SWTMatrix;
    Components* components;

    if (darkOnLight) {
        SWTMatrix = SWTMatrixDark;
        components = &componentsDark;
    } else {
        SWTMatrix = SWTMatrixLight;
        components = &componentsLight;
    }

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
        components->add_components();
    }

    for (int j = 0; j < num_vertices; j++) {
        //  components[c[j]].points.emplace_back(reverse_map[j]);
        Component* comp = components->mutable_components(c[j]);
        auto points = comp->mutable_points();
        auto point = points->Add();
        point->set_x(reverse_map[j].x);
        point->set_y(reverse_map[j].y);
        point->set_swt(reverse_map[j].SWT);
        if (darkOnLight) {
            comp->set_isdarkonlight(1);
        } else {
            comp->set_isdarkonlight(0);
        }
    }
}


void ConnectedComponents::firstStageFilter(bool darkOnLight) {
    Components* components;
    Mat morphImg, src;
    Mat element = getStructuringElement(MORPH_RECT, Size(2, 2), Point(1, 1));

    if (darkOnLight) {
        components = &componentsDark;
        src = connectedComponentsDark;
    } else {
        components = &componentsLight;
        src = connectedComponentsLight;
    }

    dilate(src, morphImg, element);
  //  imshow("Dilation", morphImg);
   // waitKey(0);

    if (darkOnLight) {
        imwrite(string("../images/Dilation") + "_dark.JPG", morphImg);
    } else {
        imwrite(string("../images/Dilation") + "_light.JPG", morphImg);
    }
    erode(morphImg, morphImg, element);
    //imshow("Erosion", morphImg);
    //waitKey(0);

    if (darkOnLight) {
        imwrite(string("../images/Erosion") + "_dark.JPG", morphImg);
    } else {
        imwrite(string("../images/Erosion") + "_light.JPG", morphImg);
    }

    for (int i = 0; i < components->mutable_components()->size(); i++) {
        improveComponentSWT(components->mutable_components(i), morphImg, darkOnLight);

        int maxY = 0, maxX = 0;
        int minY = image.size[1];
        int minX = image.size[0];

        for (auto &pixel : components->mutable_components(i)->points()) {
            maxY = max(maxY, pixel.y());
            maxX = max(maxX, pixel.x());
            minY = min(minY, pixel.y());
            minX = min(minX, pixel.x());
        }


        if (maxY != minY && maxX != minX) {
            setValidComponent(components->mutable_components(i), maxX, minX, maxY, minY);
        }
    }
}

void ConnectedComponents::markComponents() {
    for (int i = 0; i < validComponents.components().size(); i++) {
        auto comp = validComponents.mutable_components(i);
        Mat compImgToShow;
        image.copyTo(compImgToShow);
        int curPixel = comp->minx();
        while (curPixel < comp->maxx()) {
            compImgToShow.at<Vec3b>(comp->maxy(), curPixel) = Vec3b(0, 255, 0);
            compImgToShow.at<Vec3b>(comp->miny(), curPixel) = Vec3b(0, 255, 0);
            curPixel++;
        }

        curPixel = comp->miny();
        while (curPixel < comp->maxy()) {
            compImgToShow.at<Vec3b>(curPixel, comp->minx()) = Vec3b(0, 255, 0);
            compImgToShow.at<Vec3b>(curPixel, comp->maxx()) = Vec3b(0, 255, 0);
            curPixel++;
        }

        namedWindow("component" + to_string(i), WINDOW_NORMAL);
        resizeWindow("component" + to_string(i), compImgToShow.size[0]*4/5, compImgToShow.size[1]*4/5);
        imshow("component" + to_string(i), compImgToShow);

        char keyPress = (char)waitKey(0);
        cout << "Press 0 fo non-text or 1 for text to mark data" << endl;
        cout << keyPress << endl;

        if (keyPress == '1') {
            comp->set_istext(1);
            cout << i << " text" << endl;
        } else if (keyPress == '0'){
            comp->set_istext(0);
            cout << i << " non-text" << endl;
        } else {
            cout << i << " undefined key" << endl;
        }
        destroyAllWindows();
    }

}

void ConnectedComponents::showAndSaveComponents() {
    filename = filename.substr(filename.size() - 12, 8);
    int count = 0;
    for (auto comp : validComponents.components()) {
        Mat compImg = image(Rect(comp.miny(), comp.minx(), comp.height(), comp.width()));
        imwrite("../components/" + filename + "/COMP_" + to_string(count) + ".JPG", compImg);
        Mat src;
        if (comp.isdarkonlight()) {
            src = connectedComponentsDark;
        } else {
            src = connectedComponentsLight;
        }
        line(src, Point(comp.maxy(), comp.minx()), Point(comp.maxy(), comp.maxx()), Scalar(0, 191, 255), 2);
        line(src, Point(comp.miny(), comp.minx()), Point(comp.miny(), comp.maxx()), Scalar(0, 191, 255), 2);
        line(src, Point(comp.maxy(), comp.minx()), Point(comp.miny(), comp.minx()), Scalar(0, 191, 255), 2);
        line(src, Point(comp.maxy(), comp.maxx()), Point(comp.miny(), comp.maxx()), Scalar(0, 191, 255), 2);
        count++;
    }
 //   namedWindow("Connected components", WINDOW_NORMAL);
 //   resizeWindow("Connected components", connectedComponentsDark.size[0]*4/5, connectedComponentsDark.size[1]*4/5);
  //  imshow("Connected components", connectedComponentsDark);
    imwrite("../images/catboost/" + filename + "_connectedComponentsDark.jpg", connectedComponentsDark);
//    waitKey(0);

  //  imshow("Connected components", connectedComponentsLight);
    imwrite("../images/catboost/" + filename + "_connectedComponentsLight.jpg", connectedComponentsLight);
    // waitKey(0);
}


void ConnectedComponents::setValidComponent(Component* comp, int maxX, int minX, int maxY, int minY) {
    float height = (float)maxY - minY + 1;
    float width = (float)maxX - minX + 1;
    auto q = (float)comp->points().size();

    float mean = 0;
    for (auto &point : comp->points()) {
        mean += point.swt();
    }
    mean /= q;

    float SD = 0;
    for (auto &point : comp->points()) {
        SD += (point.swt() - mean) * (point.swt() - mean);
    }
    SD /= q;
    SD = sqrt(SD);

    float WV = SD / mean;
    float AR = min(width / height, height / width);
    float OR = q / (width * height);
    bool valid = width != 0 && height != 0 && q > 40 && OR >= 0.1 && AR >= 0.1 && AR <= 1; // && WV >= 0 && WV <= 1;
    if (valid) {
        Component* valComp = validComponents.add_components();
        for (int i=0; i < comp->points().size(); i++) {
            auto points = valComp->mutable_points();
            auto point = points->Add();
            point->set_x(comp->points(i).x());
            point->set_y(comp->points(i).y());
            point->set_swt(comp->points(i).swt());
        }
        valComp->set_isdarkonlight(comp->isdarkonlight());
        valComp->set_mean(mean);
        valComp->set_height(height);
        valComp->set_width(width);
        valComp->set_wv(WV);
        valComp->set_ar(AR);
        valComp->set_or_(OR);
        valComp->set_sd(SD);
        valComp->set_maxx(maxX);
        valComp->set_minx(minX);
        valComp->set_maxy(maxY);
        valComp->set_miny(minY);
    }
}

void ConnectedComponents::improveComponentSWT(Component* comp, Mat morphImg, bool darkOnLight) {
    vector<SWTPoint_buf> validPoints;
    Mat *SWT;
    if (darkOnLight) {
        SWT = &connectedComponentsDark;
    } else {
        SWT = &connectedComponentsLight;
    }
    for (auto p : comp->points()) {
        if (p.x() > 0 && p.y() > 0 && p.y() < morphImg.size[1] && p.x() < morphImg.size[0] &&
            morphImg.at<Vec3b>(p.x(), p.y()) == Vec3b(255, 255, 255)) {
            SWT->at<Vec3b>(p.x(), p.y()) = Vec3b(255, 255, 255);
        } else if (p.x() > 0 && p.y() > 0 && p.y() < morphImg.size[1] && p.x() < morphImg.size[0]) {
            validPoints.emplace_back(p);
            SWT->at<Vec3b>(p.x(), p.y()) = Vec3b(0, 0, 0);
        }
    }
    comp->clear_points();
    auto points = comp->mutable_points();
    for (auto &validPoint : validPoints) {
        auto point = points->Add();
        point->set_x(validPoint.x());
        point->set_y(validPoint.y());
        point->set_swt(validPoint.swt());
    }

}


void ConnectedComponents::computeFeatures() {
    /// find text orientation
    int edge_threshold_low = 80;
    int edge_threshold_high = edge_threshold_low * 2;
    Mat angles = Mat(image.size[0], image.size[1], CV_32F, Scalar(0));
    Mat gray, edge;
    cvtColor(image, gray, COLOR_BGR2GRAY);
    blur(gray, gray, Size(3, 3));
    Mat cdst;
    Canny(gray, edge, edge_threshold_low, edge_threshold_high, 3);

    cvtColor(edge, cdst, COLOR_GRAY2BGR);
    vector<Vec4i> lines;
    HoughLinesP(edge, lines, 1, CV_PI / 180, 50, 10, 100);
    for (size_t i = 0; i < lines.size(); i++) {
        Vec4i l = lines[i];
        Point p1, p2;
        p1 = Point(l[0], l[1]);
        p2 = Point(l[2], l[3]);
        line(cdst, p1, p2, Scalar(0, 191, 255), 3, 1);
        float angle = roundf((atan2(p1.x - p2.x, p1.y - p2.y) + PI / 2) * 100) / 100 ;
        LineIterator it(angles, p1, p2, 8);
        LineIterator it2 = it;

        for(int i = 0; i < it2.count; i++, ++it2) {
            if (angles.at<float>(it2.pos()) != 0) {
                angles.at<float>(it2.pos()) += angle;
                angles.at<float>(it2.pos()) /= 2;
                angles.at<float>(it2.pos()) = roundf(angles.at<float>(it2.pos()) * 100) / 100;
            } else {
                angles.at<float>(it2.pos()) = angle;
            }
        }
    }

    imshow("lines", cdst);
    waitKey(0);
    imwrite("../images/" + filename + "_lines.jpg", cdst);


    /// set features
    Mat rotComps;
    image.copyTo(rotComps);
    for (int i = 0; i < validComponents.components().size(); i++) {
        auto c = validComponents.mutable_components(i);
        c->set_image(stoi(filename.substr(4, 4)));
        c->set_id(i);
        vector<float> orients;
        float orientation = 0, centerY = 0, centerX = 0;
        int count = 0;
        //TODO add hashmap
        for (auto p : c->points()) {
            centerY += p.y();
            centerX += p.x();
            if (angles.at<float>(p.x(), p.y()) != 0) {
                if (find(orients.begin(), orients.end(), angles.at<float>(p.x(), p.y())) == orients.end()) {
                    count++;
                }
                orients.emplace_back(angles.at<float>(p.x(), p.y()));
            }
        }
        sort(orients.begin(), orients.end());
        int len = orients.size();
        if (len > 0) {
            orientation = orients[len / 2];
        }
        centerY /= c->points().size();
        centerX /= c->points().size();
        c->set_orientation(orientation);
        c->set_center_x(centerX);
        c->set_center_y(centerY);
        c->set_minor_axis(c->width());
        c->set_major_axis(c->height());
    }

 //   imshow("rotated components", rotComps);
//    waitKey(0);
}

void ConnectedComponents::saveData() {
    const string result_file = "../protobins/component_" + filename + ".bin";
    fstream output(result_file, ios::out | ios::binary);
    if (!validComponents.SerializeToOstream(&output)) {
        cerr << "Failed to serialize data" << endl;
    }
    output.close();
    google::protobuf::ShutdownProtobufLibrary();
}