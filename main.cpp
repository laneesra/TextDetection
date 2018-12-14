#include "opencv2/core/utility.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/core/core_c.h"
#include <iostream>
#include "ConnectedComponents.h"
#include "ComponentsChain.h"

using namespace cv;
using namespace std;

int main(int argc, const char** argv) {
    vector<string> fn;
    glob("/home/laneesra/PycharmProjects/TextDetection/MSRA-TD500/train/catboost/IMG_0605.JPG", fn, false);
    //glob("/home/laneesra/Документы/курсач/svt1/img/IMG_1802.jpg", fn, false);
    //  image = imread("../images/original/" + filename + format);

    int count = fn.size(); //number of png files in images folder
    if (count == 0) {
        cerr << "no such file";
    }
    cout << count;
    for (int i=0; i<count; i++) {
        string filename = fn[i];
        cout << endl << filename << endl;
        StrokeWidthTransform swtDark(filename);
        swtDark.execute(true);
        StrokeWidthTransform swtLight(filename);
        swtLight.execute(false);
        ConnectedComponents cc = ConnectedComponents(filename, swtDark.SWTMatrix, swtDark.result,
                                                     swtLight.SWTMatrix,
                                                     swtLight.result, swtDark.image);
        cc.execute();
    }
}
