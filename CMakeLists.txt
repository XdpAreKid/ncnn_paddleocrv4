cmake_minimum_required(VERSION 3.27)
project(test_paddle)
find_package(OpenCV REQUIRED)
add_subdirectory(ncnn)
include_directories( ${OpenCV_INCLUDE_DIRS})
set(CMAKE_CXX_STANDARD 14)
#set(ncnn_DIR "/usr/local/lib/cmake/ncnn/ncnnConfig.cmake" CACHE PATH "Directory that contains ncnnConfig.cmake")
find_package(ncnn REQUIRED)

add_executable(test_paddle
        main.cpp
        dbNet.cpp
        clipper.cpp
        crnnNet.cpp)
target_link_libraries(test_paddle
        ncnn
        opencv_core
        opencv_imgproc
        opencv_highgui
        opencv_freetype
)
