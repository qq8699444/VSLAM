#指定生成目标
add_executable(imageBasic imageBasic.cpp)

#链接共享库
target_link_libraries(imageBasic ${OpenCV_LIBS} )                                                                             
set_target_properties(imageBasic PROPERTIES FOLDER "chapter4") 

