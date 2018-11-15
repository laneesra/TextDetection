#include "opencv2/core/utility.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/core/core_c.h"
#include <iostream>
#include "StrokeWidthTransform.h"
#include "ConnectedComponents.h"

using namespace cv;
using namespace std;

int main(int argc, const char** argv) {
    StrokeWidthTransform swt;
    swt.edgeDetection();
    swt.gradient();
    swt.buildSWT(true);
    swt.medianFilter();
    swt.showSWT();
    ConnectedComponents cc = ConnectedComponents(swt.SWTMatrix, swt.result);
    cc.findComponentsBoost();
    cc.showComponents();

}