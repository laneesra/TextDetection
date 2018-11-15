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
    string filename = "Pict0021";
    string format = ".jpg";

    StrokeWidthTransform swt(filename, format);
    swt.execute();

    ConnectedComponents cc = ConnectedComponents(filename, swt.SWTMatrix, swt.result);
    cc.execute();
}