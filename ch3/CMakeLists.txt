#指定生成目标
add_executable(eigenMatrix eigenMatrix.cpp)

#链接共享库
#target_link_libraries(Sample3_1 Common )                                                                             
set_target_properties(eigenMatrix PROPERTIES FOLDER "chapter3") 
