cmake_minimum_required( VERSION 2.6 )
project( VSLAM_code )

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O0 -g -Wall")

set_property(GLOBAL PROPERTY USE_FOLDERS On)
include_directories( /usr/include/eigen3/)

 
SUBDIRS( ch3  )		
