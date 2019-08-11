#指定生成目标
add_executable(ceres_curve_fitting ceres_curve_fitting.cpp)
add_executable(g2o_curve_fitting g2o_curve_fitting.cpp)
#链接共享库
target_link_libraries(ceres_curve_fitting ceres glog ${OpenCV_LIBS})   
target_link_libraries(g2o_curve_fitting ceres glog ${OpenCV_LIBS} g2o_core g2o_stuff)       

set_target_properties(ceres_curve_fitting PROPERTIES FOLDER "chapter6") 
set_target_properties(g2o_curve_fitting PROPERTIES FOLDER "chapter6") 
