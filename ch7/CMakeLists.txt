#指定生成目标
add_executable(feature_extraction feature_extraction.cpp)
target_link_libraries(feature_extraction   ${OpenCV_LIBS})  
set_target_properties(feature_extraction PROPERTIES FOLDER "chapter7") 

#链接共享库

add_executable(pose_estimation_2d2d pose_estimation_2d2d.cpp)

target_link_libraries(pose_estimation_2d2d ${OpenCV_LIBS} )       
set_target_properties(pose_estimation_2d2d PROPERTIES FOLDER "chapter7") 
