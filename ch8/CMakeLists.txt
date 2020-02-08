add_executable(useLK useLK.cpp)
target_link_libraries(useLK ${OpenCV_LIBS} )                                                                             
set_target_properties(useLK PROPERTIES FOLDER "chapter8") 


add_executable(direct_sparse direct_sparse.cpp)
target_link_libraries(direct_sparse ${OpenCV_LIBS} cxsparse g2o_core g2o_stuff g2o_types_sba g2o_csparse_extension)                                                                             
set_target_properties(direct_sparse PROPERTIES FOLDER "chapter8") 


add_executable(direct_semidense direct_semidense.cpp)
target_link_libraries(direct_semidense ${OpenCV_LIBS} cxsparse g2o_core g2o_stuff g2o_types_sba g2o_csparse_extension)                                                                             
set_target_properties(direct_semidense PROPERTIES FOLDER "chapter8") 


