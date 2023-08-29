cmake_minimum_required(VERSION 3.13)
project( NvidiaCudaTest )
find_package( OpenCV REQUIRED )
find_package(CUDA REQUIRED)

# Add CUDA files to the project
# set(CUDA_FILES
#     vector_add.cu
# )
# Add C++ files to the project
set(CPP_FILES
	NvidiaCudaTest.cpp
)

# Tell CMake to compile the CUDA files
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-O3)
CUDA_ADD_EXECUTABLE(NvidiaCudaTest ${CUDA_FILES} ${CPP_FILES})
include_directories( ${OpenCV_INCLUDE_DIRS} )
# add_executable( NvidiaCudaTest NvidiaCudaTest.cpp )
target_link_libraries( NvidiaCudaTest ${OpenCV_LIBS} )
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
