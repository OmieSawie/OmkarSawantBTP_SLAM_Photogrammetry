/* // Include Libraries */
/* #include <iostream> */
/* #include <opencv2/opencv.hpp> */
/* #include <opencv2/videoio.hpp> */

/* // Namespace to nullify use of cv::function(); syntax */
/* using namespace std; */
/* using namespace cv; */

/* int main() { */
/*     // initialize a video capture object */
/*     // */
/*     String filename = "./carOnRuralRoads"; */
/*     /1* VideoCapture vid_capture("./carOnRuralRoads0.avi", cv::CAP_MSMF); *1/
 */
/*     VideoCapture vid_capture; */
/*     /1* vid_capture.open( *1/ */
/*     /1*     "v4l2src device=/dev/video0 ! videoscale ! videorate !
 * video/x-raw, " */
/*      *1/ */
/*     /1*     "width=640, height=360, framerate=30/1 ! videoconvert ! appsink",
 * *1/ */
/*     /1*     CAP_GSTREAMER); *1/ */

/*     const char *local_context = */
/*         "uridecodebin " */
/*         "uri=file:./carOnRuralRoads0.avi" */
/*         "! nvvidconv ! video/x-raw(memory:NVMM) ! nvvidconv ! " */
/*         "video/x-raw,format=BGRx ! videoconvert ! video/x-raw,format=BGR ! "
 */
/*         "appsink drop=1"; */

/*     vid_capture.open(local_context, CAP_IMAGES); */

/*     // Print error message if the stream is invalid */
/*     if (!vid_capture.isOpened()) { */
/*         cout << "Error opening video stream or file" << endl; */
/*     } */

/*     else { */
/*         // Obtain fps and frame count by get() method and print */
/*         // You can replace 5 with CAP_PROP_FPS as well, they are enumerations
 */
/*         int fps = vid_capture.get(5); */
/*         cout << "Frames per second :" << fps; */

/*         // Obtain frame_count using opencv built in frame count reading
 * method */
/*         // You can replace 7 with CAP_PROP_FRAME_COUNT as well, they are */
/*         // enumerations */
/*         int frame_count = vid_capture.get(7); */
/*         cout << "  Frame count :" << frame_count; */
/*     } */

/*     // Read the frames to the last frame */
/*     while (vid_capture.isOpened()) { */
/*         // Initialise frame matrix */
/*         Mat frame; */

/*         // Initialize a boolean to check if frames are there or not */
/*         bool isSuccess = vid_capture.read(frame); */

/*         // If frames are present, show it */
/*         if (isSuccess == true) { */
/*             // display frames */
/*             imshow("Frame", frame); */
/*         } */

/*         // If frames are not there, close it */
/*         if (isSuccess == false) { */
/*             cout << "Video camera is disconnected" << endl; */
/*             break; */
/*         } */

/*         // wait 20 ms between successive frames and break the loop if key q
 * is */
/*         // pressed */
/*         int key = waitKey(20); */
/*         if (key == 'q') { */
/*             cout << "q key is pressed by the user. Stopping the video" <<
 * endl; */
/*             break; */
/*         } */
/*     } */
/*     // Release the video capture object */
/*     vid_capture.release(); */
/*     destroyAllWindows(); */
/*     return 0; */
/* } */

// Include Libraries
#include <iostream>
#include <opencv2/opencv.hpp>

// Namespace to nullify use of cv::function(); syntax
using namespace std;
using namespace cv;

int main() {
    // initialize a video capture object
    VideoCapture vid_capture(
        "/home/omie_sawie/Code_Code/OmkarSawantBTP_SLAM_Photogrammetry/"
        "OpenCV_CPP/Photogrammetry_OpenCV_CPP/videoFeatureMatching/"
        "carOnRuralRoads.mp4");

    // Print error message if the stream is invalid
    if (!vid_capture.isOpened()) {
        cout << "Error opening video stream or file" << endl;
    }

    else {
        // Obtain fps and frame count by get() method and print
        // You can replace 5 with CAP_PROP_FPS as well, they are enumerations
        int fps = vid_capture.get(5);
        cout << "Frames per second :" << fps;

        // Obtain frame_count using opencv built in frame count reading method
        // You can replace 7 with CAP_PROP_FRAME_COUNT as well, they are
        // enumerations
        int frame_count = vid_capture.get(7);
        cout << "  Frame count :" << frame_count;
    }

    // Read the frames to the last frame
    while (vid_capture.isOpened()) {
        // Initialise frame matrix
        Mat frame;

        // Initialize a boolean to check if frames are there or not
        bool isSuccess = vid_capture.read(frame);

        // If frames are present, show it
        if (isSuccess == true) {
            // display frames
            imshow("Frame", frame);
        }

        // If frames are not there, close it
        if (isSuccess == false) {
            cout << "Video camera is disconnected" << endl;
            break;
        }

        // wait 20 ms between successive frames and break the loop if key q is
        // pressed
        int key = waitKey(20);
        if (key == 'q') {
            cout << "q key is pressed by the user. Stopping the video" << endl;
            break;
        }
    }
    // Release the video capture object
    vid_capture.release();
    destroyAllWindows();
    return 0;
}
