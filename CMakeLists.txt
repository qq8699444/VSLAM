cmake_minimum_required( VERSION 2.6 )
project( VSLAM_code )

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3 -g -Wall")

set_property(GLOBAL PROPERTY USE_FOLDERS On)
include_directories( /usr/include/eigen3/)
include_directories( /usr/local/include/)
#include_directories( ${PROJECT_SOURCE_DIR}/3rd/Sophus/)
include_directories( ${PROJECT_SOURCE_DIR}/3rd/Sophus/)

find_package(OpenCV REQUIRED)
include_directories( ${OpenCV_INCLUDE_DIRS})
SUBDIRS( ch3  ch4 ch5 ch6 ch7)		
