# TextDetection
Сomputer graphics course project for text detection. The algorithm was offered by Yao in 2014. It is based on SWT and uses 2 stage classifier.

## Build
To build project run ./build script.

## Run
To test text detection run ./detect_text and enter the path to image you want to test.

### Requirements
C++: boost
python(2.7): plot, CatBoost
OpenCV 3.3.0
CUDA Toolkit 9.0, cuDNN 7.2.1, gcc 6.0, g++ 6.0 (for cuda opencv module)
protobuf 3.6.1
