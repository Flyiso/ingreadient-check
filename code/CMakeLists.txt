cmake_minimum_required(VERSION 3.10)
project(stitcher)

find_package(OpenCV REQUIRED)
find_package(pybind11 REQUIRED)

add_library(stitcher MODULE stitcher.cpp)
target_link_libraries(stitcher PRIVATE ${OpenCV_LIBS} pybind11::module)
set_target_properties(stitcher PROPERTIES PREFIX "${PYTHON_MODULE_PREFIX}" SUFFIX "${PYTHON_MODULE_EXTENSION}")
