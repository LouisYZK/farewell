# 只导出指定的符号
set(CMAKE_C_VISIBILITY_PRESET hidden) 
set(CMAKE_CXX_VISIBILITY_PRESET hidden)

add_library(mylib SHARED mylib.cpp)