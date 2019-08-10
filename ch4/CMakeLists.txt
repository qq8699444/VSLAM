#指定生成目标
add_executable(useSophus useSophus.cpp)

#链接共享库
#target_link_libraries(Sample3_1 Common )                                                                             
set_target_properties(useSophus PROPERTIES FOLDER "chapter4") 

