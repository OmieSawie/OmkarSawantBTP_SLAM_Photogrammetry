// Include Libraries
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <unistd.h>

// Namespace to nullify use of cv::function(); syntax
using namespace std;
using namespace cv;
using namespace cv::xfeatures2d;

int main() {
    // initialize a video capture object
    VideoCapture vid_capture(
        "/home/omie_sawie/Code_Code/OmkarSawantBTP_SLAM_Photogrammetry/"
        "OpenCV_CPP/Photogrammetry_OpenCV_CPP/videoFeatureMatching/"
        "carOnLonelyRoads.mp4");

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
    Mat prevFrame;
    vid_capture.read(prevFrame);
    /* namedWindow("Frame"); */
    /* moveWindow("Frame", 0, 0); */
    /* namedWindow("PrevFrame"); */
    /* moveWindow("PrevFrame", 2000, 0); */
    namedWindow("Good Matches");
    moveWindow("Good Matches", 0, 0);

    while (vid_capture.isOpened()) {
        // Initialise frame matrix
        Mat frame;

        // Initialize a boolean to check if frames are there or not
        bool isSuccess = vid_capture.read(frame);

        // If frames are present, show it
        if (isSuccess == true) {
            // display frames
            /* imshow("Frame", frame); */
            /* imshow("PrevFrame", prevFrame); */
        }

        // If frames are not there, close it
        if (isSuccess == false) {
            cout << "Video camera is disconnected" << endl;
            break;
        }

        //-- Step 1: Detect the keypoints using SURF Detector, compute the
        // descriptors
        int minHessian = 400;
        Ptr<SURF> detector = SURF::create(minHessian);
        std::vector<KeyPoint> keypoints1, keypoints2;
        Mat descriptors1, descriptors2;
        detector->detectAndCompute(frame, noArray(), keypoints1, descriptors1);
        detector->detectAndCompute(prevFrame, noArray(), keypoints2,
                                   descriptors2);
        //-- Step 2: Matching descriptor vectors with a FLANN based matcher
        // Since SURF is a floating-point descriptor NORM_L2 is used
        Ptr<DescriptorMatcher> matcher =
            DescriptorMatcher::create(DescriptorMatcher::FLANNBASED);
        std::vector<std::vector<DMatch>> knn_matches;
        matcher->knnMatch(descriptors1, descriptors2, knn_matches, 2);
        //-- Filter matches using the Lowe's ratio test
        const float ratio_thresh = 0.2f;
        std::vector<DMatch> good_matches;
        for (size_t i = 0; i < knn_matches.size(); i++) {
            if (knn_matches[i][0].distance <
                ratio_thresh * knn_matches[i][1].distance) {
                good_matches.push_back(knn_matches[i][0]);
            }
        }
        //-- Draw matches
        Mat img_matches;
        drawMatches(frame, keypoints1, prevFrame, keypoints2, good_matches,
                    img_matches, Scalar::all(-1), Scalar::all(-1),
                    std::vector<char>(),
                    DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
        //-- Show detected matches

        int down_width = 1900;
        int down_height = 900;
        Mat resized_down;
        // resize down
        resize(img_matches, resized_down, Size(down_width, down_height),
               INTER_LINEAR);

        imshow("Good Matches", resized_down);

        sleep(0.01);

        // wait 20 ms between successive frames and break the loop if key q is
        // pressed
        int key = waitKey(20);
        prevFrame = frame;
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
