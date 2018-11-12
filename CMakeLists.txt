cmake_minimum_required(VERSION 3.12)
project(TextDetection)

find_package(OpenCV REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
set(SOURCE_FILES main.cpp ConnectedComponents.cpp ConnectedComponents.h)
set(CMAKE_INCLUDE_PATH /usr/local/lib)
add_executable(TextDetection main.cpp StrokeWidthTransform.cpp StrokeWidthTransform.h ${SOURCE_FILES})

include_directories(${OpenCV_INCLUDE_DIRS})
target_link_libraries(TextDetection ${OpenCV_LIBS} )
