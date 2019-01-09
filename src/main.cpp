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

int main(int argc, const char** argv) {
    string filename;
    double duration;
    cin >> filename;
    clock_t start;
    start = clock();
    cout << endl << filename << endl;
    StrokeWidthTransform swtDark(filename);
    auto func = std::async(launch::async, &StrokeWidthTransform::execute, swtDark, true);
    duration = (clock() - start) / (double) CLOCKS_PER_SEC;
    cout << duration << endl;
    StrokeWidthTransform swtLight(filename);
    swtLight.execute(false);
    duration = (clock() - start) / (double) CLOCKS_PER_SEC;
    cout << duration << endl;
    func.get();
    ConnectedComponents cc = ConnectedComponents(filename, swtDark.SWTMatrix, swtDark.result,
                                                 swtLight.SWTMatrix,
                                                 swtLight.result, swtDark.image);
    cc.execute(swtDark.edge);

    duration = (clock() - start) / (double) CLOCKS_PER_SEC;
    cout << duration << endl;
}