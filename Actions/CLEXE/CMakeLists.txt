project(Simplest CXX)
cmake_minimum_required(VERSION 3.23)

add_executable(simplest simplest.cpp)

enable_testing()
add_test(NAME simplest COMMAND simplest.exe) 
