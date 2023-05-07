cmake_minimum_required(VERSION 3.0)
project(BA_PnP)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIRS})

find_package(Sophus REQUIRED)
include_directories(${Sophus_INCLUDE_DIRS})

find_package(fmt REQUIRED)
include_directories(${fmt_INCLUDE_DIRS})
link_directories(${fmt_LIBRARY_DIRS})

add_executable(ba_pnp main.cpp)
target_link_libraries(ba_pnp ${OpenCV_LIBS} ${Sophus_LIBRARIES} fmt)
