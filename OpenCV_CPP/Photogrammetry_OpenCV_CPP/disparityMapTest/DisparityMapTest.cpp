#include "opencv2/cudastereo.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include <iomanip>
#include <iostream>
#include <opencv2/calib3d.hpp>
#include <opencv2/core/utility.hpp>
#include <sstream>
#include <stdexcept>
#include <string>

using namespace cv;
using namespace std;

int main(int argc, char **argv) {

    bool running;
    Mat left_src, right_src;
    Mat left, right;
    cuda::GpuMat d_left, d_right;

    int ndisp = 80;

    Ptr<cuda::StereoBM> bm;

    bm = cuda::createStereoBM(ndisp, 9);

    VideoCapture vid_capture(
        "/home/omie_sawie/Code_Code/OmkarSawantBTP_SLAM_Photogrammetry/"
        "OpenCV_CPP/Photogrammetry_OpenCV_CPP/videoFeatureMatching/"
        "/resources/carOnLonelyRoads.mp4");

    // Print error message if the stream is invalid
    if (!vid_capture.isOpened()) {
        cout << "Error opening video stream or file" << endl;
    }

    while (vid_capture.isOpened()) {
        // Initialise frame matrix
        // Initialize a boolean to check if frames are there or not
        vid_capture.read(left_src);
        vid_capture.read(right_src);
        int down_width = 1900;
        int down_height = 900;

        cvtColor(left_src, left, COLOR_BGR2GRAY);
        cvtColor(right_src, right, COLOR_BGR2GRAY);

        cv::resize(left, left, cv::Size(), 0.5, 0.5);
        cv::resize(right, right, cv::Size(), 0.5, 0.5);

        // resize down

        d_left.upload(left);
        d_right.upload(right);

        imshow("left", left);
        imshow("right", right);

        // Prepare disparity map of specified type
        Mat disp(left.size(), CV_8U);
        cuda::GpuMat d_disp(left.size(), CV_8U);

        bm->compute(d_left, d_right, d_disp);

        // Show results
        d_disp.download(disp);

        imshow("disparity", (Mat_<uchar>)disp);

        cout << disp.<< endl;
        Mat disp8;
        normalize(disp, disp8, 0, 255, NORM_MINMAX, CV_8U);

        /* Q = np.array(([ 1.0, 0.0, 0.0, -160.0 ], [ 0.0, 1.0, 0.0, -120.0 ],
         */
        /*               [ 0.0, 0.0, 0.0, 350.0 ], [ 0.0, 0.0, 1.0 / 90.0, 0.0
         * ]), */
        /*              dtype = np.float32) */

        /* cv::stereoRectify(InputArray cameraMatrix1, InputArray
         * distCoeffs1,
         */
        /*                   InputArray cameraMatrix2, InputArray
         * distCoeffs2,
         */
        /*                   Size imageSize, InputArray R, InputArray T, */
        /*                   OutputArray R1, OutputArray R2, OutputArray P1,
         */
        /*                   OutputArray P2, OutputArray Q); */

        int key = waitKey(20);
        if (key == 'q') {
            cout << "q key is pressed by the user. Stopping the "
                    "video"
                 << endl;
            break;
        }
    }
    // Release the video capture object
    vid_capture.release();
    destroyAllWindows();
    return 0;
}
