#include "opencv2/core/utility.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/core/core_c.h"
#include <iostream>
#include <future>
#include "ConnectedComponents.h"

using namespace cv;
using namespace std;

void runPass(bool isDarkOnLight, string filename);


int main(int argc, const char** argv) {
    string filename = argv[1];
    string isDakrOnLight = argv[2];
    double duration;

    clock_t start;
    start = clock();
    cout << isDakrOnLight << endl;
    cout << endl << filename  << endl;
    if (isDakrOnLight == "true") {
        runPass(true, filename);
    } else {
        runPass(false, filename);
    }
    duration = (clock() - start) / (double) CLOCKS_PER_SEC;
    cout << duration << endl;
}


void runPass(bool isDarkOnLight, string filename) {
    StrokeWidthTransform swt(filename);
    swt.execute(isDarkOnLight);
    ConnectedComponents cc = ConnectedComponents(filename, swt.SWTMatrix, swt.result, swt.image, isDarkOnLight);
    cc.execute(swt.edge);
}