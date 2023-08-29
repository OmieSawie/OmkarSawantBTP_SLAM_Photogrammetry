#include "opencv2/highgui.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#include <cstddef>
#include <iostream>
#include <iterator>
#include <opencv2/core.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/opencv.hpp>
#include <streambuf>
#include <unistd.h>
#include <vector>

#include <cmath>
#include <ctime>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudafeatures2d.hpp>
#include <opencv2/cudaimgproc.hpp>

using namespace cv::cuda;
using namespace cv;
using namespace std;

int maxCorners = 3000;
RNG rng(12345);
const char *source_window = "Image";

Ptr<DescriptorExtractor> extractor = cv::cuda::ORB::create();
Ptr<DescriptorExtractor> prev_extractor = cv::cuda::ORB::create();
Mat src, src_gray, prev_src_gray;

class FeatureExtractor {
  public:
    FeatureExtractor(){};

    void extractFeatures_goodFeaturesToTrack(int, void *) {
        cout << " Size is:" << prev_src_gray.size() << endl;

        maxCorners = MAX(maxCorners, 1);
        vector<Point2f> corners, prev_corners;
        double qualityLevel = 0.01;
        double minDistance = 3;
        int blockSize = 3, gradientSize = 3;
        bool useHarrisDetector = false;
        double k = 0.04;
        Mat copy = src.clone();

        Mat descriptors, prev_descriptors;
        vector<KeyPoint> keypoints, prev_keypoints;

        cv::cuda::createGoodFeaturesToTrackDetector(
            src_gray, corners, maxCorners, qualityLevel, minDistance, Mat(),
            blockSize, gradientSize, useHarrisDetector, k);
        cv::cuda::createGoodFeaturesToTrackDetector(int srcType);

        goodFeaturesToTrack(src_gray, corners, maxCorners, qualityLevel,
                            minDistance, Mat(), blockSize, gradientSize,
                            useHarrisDetector, k);
        goodFeaturesToTrack(prev_src_gray, prev_corners, maxCorners,
                            qualityLevel, minDistance, Mat(), blockSize,
                            gradientSize, useHarrisDetector, k);

        /* cout << "** Number of corners detected: " << corners.size() << endl;
         */
        for (int i = 0; i < corners.size(); i++) {
            KeyPoint kp(corners[i].x, corners[i].y, 2, -1, 0, 0, -1);
            keypoints.push_back(kp);
        }
        for (int i = 0; i < prev_corners.size(); i++) {
            KeyPoint kp(prev_corners[i].x, prev_corners[i].y, 2, -1, 0, 0, -1);
            prev_keypoints.push_back(kp);
        }

        extractor->compute(src_gray, keypoints, descriptors);
        prev_extractor->compute(prev_src_gray, prev_keypoints,
                                prev_descriptors);

        drawKeypoints(copy, keypoints, copy, cv::Scalar(255, 0, 0),
                      cv::DrawMatchesFlags::DRAW_OVER_OUTIMG);

        /* int radius = 3; */
        /* for (size_t i = 0; i < corners.size(); i++) { */
        /*     circle(copy, corners[i], radius, */
        /*            Scalar(rng.uniform(0, 255), rng.uniform(0, 256), */
        /*                   rng.uniform(0, 256)), */
        /*            FILLED); */
        /* } */

        // ________________________________________________________________

        cv::BFMatcher matcher(cv::NORM_L2, true);
        std::vector<cv::DMatch> matches;
        matcher.match(prev_descriptors, descriptors, matches);

        // Sort matches by score
        std::sort(matches.begin(), matches.end());

        // Remove not so good matches
        const int numGoodMatches = matches.size() * 0.4f;
        matches.erase(matches.begin() + numGoodMatches, matches.end());

        cout << "match=";
        cout << matches.size() << endl;

        // the number of matched features between the two images
        if (matches.size() != 0) {
            cout << matches.size() << endl;
            cv::Mat imageMatches;

            std::vector<char> mask(matches.size(), 1);
            cv::drawMatches(prev_src_gray, prev_keypoints, src_gray, keypoints,
                            matches, imageMatches, cv::Scalar::all(-1),
                            cv::Scalar::all(-1), mask,
                            cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
            cv::drawMatches(prev_src_gray, prev_keypoints, src_gray, keypoints,
                            matches, imageMatches);
            cv::namedWindow("matches");
            /* cv::imshow("matches", imageMatches); */

            //
            //________________________________________________________________

            namedWindow(source_window);
            moveWindow(source_window, 0, 0);
            imshow(source_window, copy);

            prev_src_gray = src_gray;
        }
    }
};

/* int main() { */
/*     // initialize a video capture object */
/*     VideoCapture vid_capture( */
/*         "/home/omie_sawie/Code_Code/OmkarSawantBTP_SLAM_Photogrammetry/" */
/*         "OpenCV_CPP/Photogrammetry_OpenCV_CPP/videoFeatureMatching/" */
/*         "/resources/carOnLonelyRoads.mp4"); */

/*     // Print error message if the stream is invalid */
/*     if (!vid_capture.isOpened()) { */
/*         cout << "Error opening video stream or file" << endl; */
/*     } */

/*     else { */
/*         // Obtain fps and frame count by get() method and print */
/*         // You can replace 5 with CAP_PROP_FPS as well, they are */
/*         // enumerations */
/*         int fps = vid_capture.get(5); */
/*         cout << "Frames per second :" << fps; */

/*         // Obtain frame_count using opencv built in frame count reading */
/*         // method You can replace 7 with CAP_PROP_FRAME_COUNT as well, */
/*         // they are enumerations */
/*         int frame_count = vid_capture.get(7); */
/*         cout << "  Frame count :" << frame_count; */

/*         Mat frame; */
/*         // Initialize a boolean to check if frames are there or not */
/*         bool isSuccess = vid_capture.read(frame); */

/*         int down_width = 1900; */
/*         int down_height = 900; */

/*         // resize down */
/*         resize(frame, src, Size(down_width, down_height), INTER_LINEAR); */

/*         cvtColor(src, prev_src_gray, COLOR_BGR2GRAY); */
/*     } */

/*     while (vid_capture.isOpened()) { */
/*         // Initialise frame matrix */
/*         Mat frame; */
/*         // Initialize a boolean to check if frames are there or not */
/*         bool isSuccess = vid_capture.read(frame); */

/*         int down_width = 1900; */
/*         int down_height = 900; */

/*         // resize down */
/*         resize(frame, src, Size(down_width, down_height), INTER_LINEAR); */

/*         cvtColor(src, src_gray, COLOR_BGR2GRAY); */
/*         namedWindow(source_window); */
/*         cv::namedWindow("matches"); */
/*         moveWindow("matches", 0, 0); */

/*         /1* setTrackbarPos("Max corners", source_window, maxCorners); *1/ */
/*         /1* imshow(source_window, src); *1/ */

/*         FeatureExtractor feature_extractor; */

/*         feature_extractor.extractFeatures_goodFeaturesToTrack(0, 0); */

/*         // If frames are present, show it */
/*         if (isSuccess == true) { */
/*             // display frames */
/*             /1* imshow("Frame", frame); *1/ */
/*             /1* imshow("PrevFrame", prevFrame); *1/ */
/*         } */

/*         if (isSuccess == false) { */
/*             cout << "Video camera is disconnected" << endl; */
/*             break; */
/*         } */

/*         //-- Step 1: Detect the keypoints using SURF Detector, compute */
/*         // the */
/*         // descriptors */

/*         // wait 20 ms between successive frames and break the loop if */
/*         // key q is pressed */
/*         int key = waitKey(20); */
/*         if (key == 'q') { */
/*             cout << "q key is pressed by the user. Stopping the " */
/*                     "video" */
/*                  << endl; */
/*             break; */
/*         } */
/*     } */
/*     // Release the video capture object */
/*     vid_capture.release(); */
/*     destroyAllWindows(); */
/*     return 0; */
/* } */

#include "opencv2/core/cuda.hpp"
#include <iostream>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;
using namespace cv::cuda;

int main() {

    // Print cuda device details
    cv::cuda::setDevice(0);
    printCudaDeviceInfo(0);

    // Read in image
    std::string filename = "/Users/alex/test.jpg";
    Mat image = imread(filename.c_str(), IMREAD_GRAYSCALE);
    if (image.empty()) {
        cout << "Could not open or find the image" << endl;
        exit(-1);
    }
    cout << "image loaded: " << image.size() << endl;

    // Upload image to GPU
    GpuMat gpuImage = GpuMat(image);

    // Set up feature detector
    int maxCorners = 100;
    double qualityLevel = 0.01;
    double minDistance = 30;
    int blockSize = 3;
    bool useHarrisDetector = false;
    double harrisK = 0.04;
    cv::Ptr<cv::cuda::CornersDetector> cornerDetector =
        cv::cuda::createGoodFeaturesToTrackDetector(
            CV_8UC1, maxCorners, qualityLevel, minDistance, blockSize,
            useHarrisDetector, harrisK);

    // Corners
    cv::cuda::GpuMat corners;

    // Detect corners
    std::cout << "Detecting corners..." << std::endl;
    cornerDetector->detect(gpuImage, corners);

    // Print number of corners detected
    std::cout << "Corners detected: " << corners.cols << std::endl;

    return 0;
}
