# 只导出指定的符号
set(CMAKE_C_VISIBILITY_PRESET hidden) 
set(CMAKE_CXX_VISIBILITY_PRESET hidden)

# set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD 20)
add_library(mylib SHARED mylib.cpp)
# add_executable(main main.cpp)
# add_executable(lambda lambda_example.cpp)
# add_executable(option optional_example.cpp)
# add_executable(variant variant_example.cpp)