cmake_minimum_required(VERSION 3.10)

set(CMAKE_CXX_STANDARD 17)

project(cpptest LANGUAGES CXX)

add_executable(cpptest use_thread.cpp)
add_executable(async use_async.cpp)
add_executable(promise use_promise.cpp)
add_executable(mutex use_mutex.cpp)
add_executable(vec safe_vector.cpp)
add_executable(acc accessor.cpp)
add_executable(cv productor_consumer.cpp)
add_executable(atomic atomic.cpp)