cmake_minimum_required(VERSION 3.17)
project(points_optimizer)

set(CMAKE_CXX_STANDARD 14)

find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )

add_executable(points_optimizer main.cpp)
target_link_libraries( points_optimizer ${OpenCV_LIBS})

find_package (Eigen3 3.3 REQUIRED)
target_link_libraries (points_optimizer Eigen3::Eigen)

find_package(Ceres REQUIRED)
include_directories(${CERES_INCLUDE_DIRS})
target_link_libraries(points_optimizer ${CERES_LIBRARIES})
