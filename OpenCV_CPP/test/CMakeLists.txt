cmake_minimum_required(VERSION 2.8)
project( DisplayImage )
find_package( OpenCV REQUIRED )
find_package( VTK REQUIRED )
add_executable( DisplayImage FaceDetection.cpp )
target_include_directories( DisplayImage
	PUBLIC
		${OpenCV_INCLUDE_DIRS}
		${VTK_INCLUDE_DIRS}
)
target_link_libraries( DisplayImage ${OpenCV_LIBS} ${VTK_LIBRARIES} )
