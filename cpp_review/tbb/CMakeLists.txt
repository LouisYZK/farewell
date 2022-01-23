cmake_minimum_required(VERSION 3.10)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE Release)

project(main LANGUAGES CXX)

add_executable(str str_para.cpp)
add_executable(map map.cpp)
add_executable(reduce reduce.cpp)
add_executable(scan scan.cpp)
add_executable(timer timer.cpp)
add_executable(bm benchmark.cpp)
add_executable(grain grain.cpp)
add_executable(concurrent_ctn concurrent_ctn.cpp)
add_executable(filter filter.cpp)
add_executable(quick_sort quick_sort.cpp)
# add_executable(pipeline pipeline.cpp)


#find_package(OpenMP REQUIRED)
#target_link_libraries(main PUBLIC OpenMP::OpenMP_CXX)

set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "Turn off the fking test!")
add_subdirectory(benchmark)
# target_link_libraries(main PUBLIC benchmark)

find_package(TBB REQUIRED)
target_link_libraries(str PUBLIC TBB::tbb)
target_link_libraries(map PUBLIC TBB::tbb)
target_link_libraries(reduce PUBLIC TBB::tbb)
target_link_libraries(scan PUBLIC TBB::tbb)
target_link_libraries(timer PUBLIC TBB::tbb)
target_link_libraries(bm PUBLIC  benchmark TBB::tbb)

target_link_libraries(grain PUBLIC TBB::tbb)
target_link_libraries(concurrent_ctn PUBLIC TBB::tbb)
target_link_libraries(filter PUBLIC TBB::tbb)
target_link_libraries(quick_sort PUBLIC TBB::tbb)
# target_link_libraries(pipeline PUBLIC TBB::tbb)