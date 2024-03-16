cmake_minimum_required(VERSION 3.13)
project( FrameTriangulationTest )
find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )
add_executable( FrameTriangulationTest FrameTriangulationTest.cpp )
target_link_libraries( ${PROJECT_NAME} ${OpenCV_LIBS} )
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

