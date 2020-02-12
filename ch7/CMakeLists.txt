
list( APPEND CMAKE_MODULE_PATH /usr/local/lib/cmake/g2o/ )
find_package( g2o REQUIRED )


add_executable(feature_extraction feature_extraction.cpp)
target_link_libraries(feature_extraction   ${OpenCV_LIBS})  
set_target_properties(feature_extraction PROPERTIES FOLDER "chapter7") 

add_executable(pose_estimation_2d2d pose_estimation_2d2d.cpp)
target_link_libraries(pose_estimation_2d2d ${OpenCV_LIBS} )       
set_target_properties(pose_estimation_2d2d PROPERTIES FOLDER "chapter7") 

add_executable(triangulation triangulation.cpp)
target_link_libraries(triangulation ${OpenCV_LIBS} )       
set_target_properties(triangulation PROPERTIES FOLDER "chapter7") 


add_executable(pose_estimation_3d2d pose_estimation_3d2d.cpp)
target_link_libraries(pose_estimation_3d2d ceres glog ${OpenCV_LIBS} cxsparse g2o_core g2o_stuff g2o_types_sba g2o_csparse_extension)       
set_target_properties(pose_estimation_3d2d PROPERTIES FOLDER "chapter7") 


add_executable(pose_estimation_3d3d pose_estimation_3d3d.cpp)
target_link_libraries(pose_estimation_3d3d ceres glog ${OpenCV_LIBS} cxsparse g2o_core g2o_stuff g2o_types_sba g2o_csparse_extension)       
set_target_properties(pose_estimation_3d3d PROPERTIES FOLDER "chapter7") 