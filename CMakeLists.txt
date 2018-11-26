cmake_minimum_required(VERSION 3.12)
project(TextDetection)

find_package(OpenCV REQUIRED)
find_package(Boost 1.68.0 COMPONENTS system filesystem REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
set(SOURCE_FILES main.cpp ConnectedComponents.cpp ConnectedComponents.h camshift.cpp)
set(CMAKE_INCLUDE_PATH /usr/local/lib)
add_executable(TextDetection main.cpp StrokeWidthTransform.cpp StrokeWidthTransform.h ${SOURCE_FILES})

include_directories(${OpenCV_INCLUDE_DIRS} ${Boost_INCLUDE_DIRS})
target_link_libraries(TextDetection ${OpenCV_LIBS} ${Boost_LIBRARIES})
