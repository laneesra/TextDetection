#include "opencv2/core/utility.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/core/core_c.h"
#include <iostream>
#include <future>
#include "ConnectedComponents.h"

using namespace std;

void runPass(bool isDarkOnLight, const string& filename);


int main(int argc, const char** argv) {
    string filename = argv[1];
    string isDarkOnLight = argv[2];
    double duration;

    clock_t start;
    start = clock();
    if (isDarkOnLight == "true") {
        runPass(true, filename);
    } else {
        runPass(false, filename);
    }
    duration = (clock() - start) / (double) CLOCKS_PER_SEC;
    cout << "full duration is " << duration << " seconds" << endl;
}


void runPass(bool isDarkOnLight, const string& filename) {
    double duration;
    clock_t start;
    start = clock();

    StrokeWidthTransform swt(filename);
    swt.execute(isDarkOnLight);

    duration = (clock() - start) / (double) CLOCKS_PER_SEC;
    cout << "swt duration is " << duration << " seconds" << endl;
    start = clock();

    ConnectedComponents cc = ConnectedComponents(filename, swt.SWTMatrix, swt.result, swt.image, isDarkOnLight);
    cc.execute(swt.edge);
    duration = (clock() - start) / (double) CLOCKS_PER_SEC;
    cout << "cc duration is " << duration << " seconds" << endl;

}