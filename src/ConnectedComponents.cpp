//
// Created by laneesra on 11.11.18.
//
#include "ConnectedComponents.h"

using namespace std;


ConnectedComponents::ConnectedComponents(const string& filename, const cv::Mat& SWTMatrix, const cv::Mat& SWTMatrixNormU, const cv::Mat& image, bool isDarkOnLight) : SWTMatrix(SWTMatrix), filename(filename), image(image), isDarkOnLight(isDarkOnLight) {
    cvtColor(SWTMatrixNormU, connectedComponents, cv::COLOR_GRAY2BGR);
}


void ConnectedComponents::execute(cv::Mat edge) {
  //  auto func = async(launch::async, &ConnectedComponents::findComponentsBoost, this, true);
    findComponentsBoost();
    firstStageFilter();

    //markComponents();
    //showAndSaveComponents();
    computeFeatures(edge);
    saveData();
}


void ConnectedComponents::findComponentsBoost() {
    boost::unordered_map<int, int> map;
    boost::unordered_map<int, SWTPoint> reverse_map;

    typedef boost::adjacency_list<boost::vecS, boost::vecS, boost::undirectedS> Graph;
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

    for (int j = 0; j < num_comp; j++) {
        components.add_components();
    }

    for (int j = 0; j < num_vertices; j++) {
        Component* comp = components.mutable_components(c[j]);
        auto points = comp->mutable_points();
        auto point = points->Add();
        point->set_x(reverse_map[j].x);
        point->set_y(reverse_map[j].y);
        point->set_swt(reverse_map[j].SWT);
        comp->set_isdarkonlight(isDarkOnLight);
    }
}


void ConnectedComponents::firstStageFilter() {
    cv::Mat element = getStructuringElement(cv::MORPH_RECT, cv::Size(2, 2), cv::Point(1, 1));
    cv::Mat morphImg;
    cv::erode(connectedComponents, morphImg, element);
    cv::dilate(morphImg, morphImg, element);

/*    cuda::GpuMat src, morphImgGpu;
    src.upload(connectedComponents);
    cuda::cvtColor(src, src, COLOR_BGR2BGRA);

    Ptr<cuda::Filter> erode = cuda::createMorphologyFilter(MORPH_ERODE, src.type(), element);
    erode->apply(src, morphImgGpu);

    Ptr<cuda::Filter> dilate = cuda::createMorphologyFilter(MORPH_DILATE, morphImgGpu.type(), element);
    dilate->apply(morphImgGpu, morphImgGpu);

    Mat morphImg(morphImgGpu);*/
    for (int i = 0; i < components.mutable_components()->size(); i++) {
        improveComponentSWT(components.mutable_components(i), morphImg);

        int maxY = 0, maxX = 0;
        int minY = image.size[1];
        int minX = image.size[0];

        for (auto &pixel : components.mutable_components(i)->points()) {
            maxY = max(maxY, pixel.y());
            maxX = max(maxX, pixel.x());
            minY = min(minY, pixel.y());
            minX = min(minX, pixel.x());
        }

        if (maxY != minY && maxX != minX) {
            setValidComponent(components.mutable_components(i), maxX, minX, maxY, minY);
        }
    }
}


void ConnectedComponents::markComponents() {
    for (int i = 0; i < validComponents.components().size(); i++) {
        auto comp = validComponents.mutable_components(i);
        cv::Mat compImgToShow;
        image.copyTo(compImgToShow);
        int curPixel = comp->minx();
        while (curPixel < comp->maxx()) {
            compImgToShow.at<cv::Vec3b>(comp->maxy(), curPixel) = cv::Vec3b(0, 255, 0);
            compImgToShow.at<cv::Vec3b>(comp->miny(), curPixel) = cv::Vec3b(0, 255, 0);
            curPixel++;
        }

        curPixel = comp->miny();
        while (curPixel < comp->maxy()) {
            compImgToShow.at<cv::Vec3b>(curPixel, comp->minx()) = cv::Vec3b(0, 255, 0);
            compImgToShow.at<cv::Vec3b>(curPixel, comp->maxx()) = cv::Vec3b(0, 255, 0);
            curPixel++;
        }

        namedWindow("component" + to_string(i), cv::WINDOW_NORMAL);
        cv::resizeWindow("component" + to_string(i), compImgToShow.size[0]*4/5, compImgToShow.size[1]*4/5);
        imshow("component" + to_string(i), compImgToShow);

        char keyPress = (char)cv::waitKey(0);
        cout << "Press 0 fo non-text or 1 for text to mark data" << endl;
        cout << keyPress << endl;

        if (keyPress == '1') {
            comp->set_istext(1);
            comp->set_istext(1);
            cout << i << " text" << endl;
        } else if (keyPress == '0'){
            comp->set_istext(0);
            cout << i << " non-text" << endl;
        } else {
            cout << i << " undefined key" << endl;
        }
        cv::destroyAllWindows();
    }

}


void ConnectedComponents::showAndSaveComponents() {
    int count = 0;
    filename = filename.substr(filename.size() - 12, 8);

    for (const auto& comp : validComponents.components()) {
       // Mat compImg = image(Rect(comp.miny(), comp.minx(), comp.height(), comp.width()));
       // imwrite("./components/" + filename + "/COMP_" + to_string(count) + ".JPG", compImg);
        line(image, cv::Point(comp.maxy(), comp.minx()), cv::Point(comp.maxy(), comp.maxx()), cv::Scalar(0, 191, 255), 2);
        line(image, cv::Point(comp.miny(), comp.minx()), cv::Point(comp.miny(), comp.maxx()), cv::Scalar(0, 191, 255), 2);
        line(image, cv::Point(comp.maxy(), comp.minx()), cv::Point(comp.miny(), comp.minx()), cv::Scalar(0, 191, 255), 2);
        line(image, cv::Point(comp.maxy(), comp.maxx()), cv::Point(comp.miny(), comp.maxx()), cv::Scalar(0, 191, 255), 2);
        count++;
    }
    //   namedWindow("Connected components", WINDOW_NORMAL);
    //   resizeWindow("Connected components", connectedComponentsDark.size[0]*4/5, connectedComponentsDark.size[1]*4/5);
    imshow("Connected components", image);
    cv::waitKey(0);

    if (isDarkOnLight) {
  //      imwrite("./images/" + filename + "_connectedComponentsDark.jpg", connectedComponents);
    } else {
  //      imwrite("./images/" + filename + "_connectedComponentsLight.jpg", connectedComponents);
    }
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
        for (int i = 0; i < comp->points().size(); i++) {
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

void ConnectedComponents::improveComponentSWT(Component* comp, cv::Mat& morphImg) {
    vector<SWTPoint_buf> validPoints;
    for (const auto& p : comp->points()) {
        if (p.x() > 0 && p.y() > 0 && p.y() < morphImg.size[1] && p.x() < morphImg.size[0] &&
            morphImg.at<cv::Vec3b>(p.x(), p.y()) == cv::Vec3b(255, 255, 255)) {
                   connectedComponents.at<cv::Vec3b>(p.x(), p.y()) = cv::Vec3b(255, 255, 255);
        } else if (p.x() > 0 && p.y() > 0 && p.y() < morphImg.size[1] && p.x() < morphImg.size[0]) {
            validPoints.emplace_back(p);
                   connectedComponents.at<cv::Vec3b>(p.x(), p.y()) = cv::Vec3b(0, 0, 0);
        }
    }
    comp->clear_points();
    auto points = comp->mutable_points();
    for (const auto &validPoint : validPoints) {
        auto point = points->Add();
        point->set_x(validPoint.x());
        point->set_y(validPoint.y());
        point->set_swt(validPoint.swt());
    }

}


void ConnectedComponents::computeFeatures(cv::Mat& edge) {
    /// find text orientation
    cv::Mat angles = cv::Mat(image.size[0], image.size[1], CV_32F, cv::Scalar(0));
    cv::Mat cdst;
    cvtColor(edge, cdst, cv::COLOR_GRAY2BGR);

    //CPU
    vector<cv::Vec4i> lines;
    //HoughLinesP(edge, lines, 1, CV_PI / 180, 50, 10, 100);

    //GPU
    cv::cuda::GpuMat linesMat;
    cv::Ptr<cv::cuda::HoughSegmentDetector> hough = cv::cuda::createHoughSegmentDetector(1.0f, (float)CV_PI / 180, 50, 10, 100);
    hough->detect(cv::cuda::GpuMat(edge), linesMat);
    if (!linesMat.empty()) {
        lines.resize((unsigned long)linesMat.cols);
        cv::Mat h_lines(1, linesMat.cols, CV_32SC4, &lines[0]);
        linesMat.download(h_lines);
    }

    for (size_t i = 0; i < lines.size(); i++) {
        cv::Vec4i l = lines[i];
        cv::Point p1, p2;
        p1 = cv::Point(l[0], l[1]);
        p2 = cv::Point(l[2], l[3]);
        line(cdst, p1, p2, cv::Scalar(0, 191, 255), 3, 1);
        float angle = roundf((atan2(p1.x - p2.x, p1.y - p2.y) + CV_PI / 2) * 100) / 100 ;
        cv::LineIterator it(angles, p1, p2, 8);
        cv::LineIterator it2 = it;

        for (int i = 0; i < it2.count; i++, ++it2) {
            if (angles.at<float>(it2.pos()) != 0) {
                angles.at<float>(it2.pos()) += angle;
                angles.at<float>(it2.pos()) /= 2;
                angles.at<float>(it2.pos()) = roundf(angles.at<float>(it2.pos()) * 100) / 100;
            } else {
                angles.at<float>(it2.pos()) = angle;
            }
        }
    }

    /// set features
    cv::Mat rotComps;
    image.copyTo(rotComps);
    for (int i = 0; i < validComponents.components().size(); i++) {
        auto c = validComponents.mutable_components(i);

        c->set_image(0);
        c->set_filename(filename);
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
}

void ConnectedComponents::saveData() {
    string result_file;
    if (isDarkOnLight) {
        result_file = "../protobins/components_dark.bin";
    } else {
        result_file = "../protobins/components_light.bin";
    }
    fstream output(result_file, ios::out | ios::binary);
    if (!validComponents.SerializeToOstream(&output)) {
        cerr << "Failed to serialize data" << endl;
    }
    output.close();
    google::protobuf::ShutdownProtobufLibrary();
}
