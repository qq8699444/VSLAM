add_executable(ceresBundle ceresBundle.cpp BALProblem.cpp flags/command_args.cpp)
target_link_libraries(ceresBundle ${OpenCV_LIBS} ceres glog)                                                                             
set_target_properties(ceresBundle PROPERTIES FOLDER "chapter10") 


