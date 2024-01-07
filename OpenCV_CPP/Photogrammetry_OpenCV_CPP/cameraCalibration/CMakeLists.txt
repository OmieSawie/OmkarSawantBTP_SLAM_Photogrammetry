# set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# cmake_minimum_required(VERSION 3.5)
# project(Test)
# set(the_target "CudaFeatureExtractionTest")

# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")


# # CUDA
# SET(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/CMake/cuda" ${CMAKE_MODULE_PATH})
# # set(OpenCV_LIBS opencv_core opencv_imgproc opencv_calib3d opencv_video opencv_features2d opencv_ml opencv_highgui opencv_objdetect opencv_contrib opencv_legacy opencv_gpu)

# FIND_PACKAGE(CUDA)
# LIST(APPEND CUDA_NVCC_FLAGS "-arch=compute_30")

# # OpenCV
# find_package(OpenCV 4  REQUIRED core imgproc video highgui videostab HINTS /usr/local/opt/opencv4)

# message(STATUS "OpenCV library status:")
# message(STATUS "    version: ${OpenCV_VERSION}")
# message(STATUS "    libraries: ${OpenCV_LIBS}")
# message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")


# # Add additional cuda resources
# include_directories(${CUDA_SDK_ROOT_DIR}/common/inc)
# include_directories (CUDABOF)
# include_directories(${OpenCV_INCLUDE_DIRS})

# # These are all our source files we need to compile
# file(GLOB lib_srcs "*.cpp" "*.hpp" "*.cu")
# source_group("Src" FILES ${lib_srcs})

# CUDA_ADD_EXECUTABLE(${the_target} ${lib_srcs})
# target_link_libraries(${the_target}  ${OpenCV_LIBS})

# set(CMAKE_EXPORT_COMPILE_COMMANDS ON)







# cmake_minimum_required(VERSION 3.13)
# project( CudaFeatureExtractionTest )
# find_package( OpenCV REQUIRED )
# find_package(CUDA REQUIRED)

# # Add C++ files to the project
# set(CPP_FILES
# 	CudaFeatureExtractionTest.cpp
# )

# # Tell CMake to compile the CUDA files
# set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-O3)
# CUDA_ADD_EXECUTABLE(CudaFeatureExtractionTest  ${CPP_FILES})
# include_directories( ${OpenCV_INCLUDE_DIRS} )
# # add_executable( CudaFeatureExtractionTest CudaFeatureExtractionTest.cpp )
# target_link_libraries( CudaFeatureExtractionTest ${OpenCV_LIBS} )

# set(CMAKE_EXPORT_COMPILE_COMMANDS ON)










cmake_minimum_required(VERSION 3.13)
project( CameraCalibration )
find_package( OpenCV REQUIRED )

# Add C++ files to the project
set(CPP_FILES
	CameraCalibration.cpp
)

# Tell CMake to compile the CUDA files
include_directories( ${OpenCV_INCLUDE_DIRS} )
add_executable( CameraCalibration CameraCalibration.cpp )
target_link_libraries( CameraCalibration ${OpenCV_LIBS} )
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
