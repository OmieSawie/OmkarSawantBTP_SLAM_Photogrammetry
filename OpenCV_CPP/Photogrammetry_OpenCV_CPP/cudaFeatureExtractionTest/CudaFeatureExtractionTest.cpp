#include <cstddef>
#include <iostream>
#include <iterator>
#include <opencv2/core.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <streambuf>
#include <unistd.h>
#include <vector>

#include <cmath>
#include <ctime>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudafeatures2d.hpp>
#include <opencv2/cudaimgproc.hpp>

using namespace std;

int maxCorners = 3000;
cv::RNG rng(12345);
const char *source_window = "Image";

int nfeatures = 500;
float scaleFactor = 1.2f;
int nlevels = 8;
int edgeThreshold = 31;
int firstLevel = 0;
int WTA_K = 2;
int scoreType = cv::ORB::HARRIS_SCORE;
int patchSize = 31;
int fastThreshold = 20;
bool blurForDescriptor = false;

cv::Ptr<cv::cuda::ORB> extractor =
    cv::cuda::ORB::create(5000, 1.2f, 8, 31, 0, 2, 0, 31, 20, true);

cv::Mat src, src_gray, prev_src_gray;

class FeatureExtractor {
  public:
    FeatureExtractor(){};

    void extractFeatures_goodFeaturesToTrack(int, void *) {

        /* cv::cuda::FastFeatureDetector::create( */
        /* 10, true, cv::FastFeatureDetector::TYPE_9_16, 5000); */

        cv::cuda::GpuMat prev_src_gray_gpu = cv::cuda::GpuMat(prev_src_gray);
        cv::cuda::GpuMat src_gray_gpu = cv::cuda::GpuMat(src_gray);
        cout << " Size is:" << prev_src_gray.size() << endl;

        maxCorners = MAX(maxCorners, 1);
        /* vector<cv::Point2f> corners, prev_corners; */
        double qualityLevel = 0.01;
        double minDistance = 3;
        int blockSize = 3, gradientSize = 3;
        bool useHarrisDetector = false;
        double k = 0.04;
        cv::cuda::GpuMat copy = src_gray_gpu.clone();
        double harrisK = 0.04;

        /* cv::Mat descriptors, prev_descriptors; */
        /* vector<cv::KeyPoint> keypoints, prev_keypoints; */

        /* cv::cuda::createGoodFeaturesToTrackDetector( */
        /*     src_gray, corners, maxCorners, qualityLevel, minDistance,
         * cv::Mat(), */

        /*     blockSize, gradientSize, useHarrisDetector, k); */

        /* cv::Ptr<cv::cuda::CornersDetector> cornerDetector = */
        /*     cv::cuda::createGoodFeaturesToTrackDetector( */
        /*         CV_8UC1, maxCorners, qualityLevel, minDistance, blockSize, */
        /*         useHarrisDetector, harrisK); */
        /* cv::cuda::GpuMat corners, prev_corners; */

        /* cornerDetector->detect(src_gray_gpu, corners); */
        /* cornerDetector->detect(prev_src_gray_gpu, prev_corners); */

        cv::cuda::GpuMat descriptors_gpu, prev_descriptors_gpu, keypoints_gpu,
            prev_keypoints_gpu;

        extractor->detectAndComputeAsync(src_gray_gpu, cv::cuda::GpuMat(),
                                         keypoints_gpu, descriptors_gpu);
        extractor->detectAndComputeAsync(prev_src_gray_gpu, cv::cuda::GpuMat(),
                                         prev_keypoints_gpu,
                                         prev_descriptors_gpu);

        std::cout << "HelloBoi " << descriptors_gpu.size() << endl;

        cv::Ptr<cv::cuda::DescriptorMatcher> matcher =
            cv::cuda::DescriptorMatcher::createBFMatcher(cv::NORM_HAMMING);

        vector<vector<cv::DMatch>> matches;
        matcher->knnMatch(descriptors_gpu, prev_descriptors_gpu, matches, 2);

        std::vector<cv::DMatch> good_matches;
        for (int k = 0; k < min(descriptors_gpu.rows - 1, (int)matches.size());
             k++) {
            if ((matches[k][0].distance < 0.6 * (matches[k][1].distance)) &&
                ((int)matches[k].size() <= 2 && (int)matches[k].size() > 0)) {
                good_matches.push_back(matches[k][0]);
            }
        }

        /* t_BBtime = getTickCount(); */
        /* t_pt = (t_BBtime - t_AAtime) / getTickFrequency(); */
        /* t_fpt = 1 / t_pt; */
        /* printf("%.4lf sec/ %.4lf fps\n", t_pt, t_fpt); */

        vector<cv::KeyPoint> keypoints_cpu, prev_keypoints_cpu;

        extractor->convert(keypoints_gpu, keypoints_cpu);
        extractor->convert(prev_keypoints_gpu, prev_keypoints_cpu);

        vector<float> descriptors_cpu, prev_descriptors_cpu;

        /* printf("%d %d\n", keypoints1.size(), keypoints2.size() ); */

        cv::Mat img_matches;

        drawMatches(src, keypoints_cpu, prev_src_gray, prev_keypoints_cpu,
                    good_matches, img_matches);

        cv::namedWindow("matches", 0);
        imshow("matches", img_matches);

        /* for (int i = 0; i < corners.size(); i++) { */
        /*     KeyPoint kp(corners[i].x, corners[i].y, 2, -1, 0, 0, -1);
         */
        /*     keypoints.push_back(kp); */
        /* } */
        /* for (int i = 0; i < prev_corners.size(); i++) { */
        /*     KeyPoint kp(prev_corners[i].x, prev_corners[i].y, 2, -1,
         * 0, 0, -1); */
        /*     prev_keypoints.push_back(kp); */
        /* } */

        /* extractor->compute(src_gray, keypoints, descriptors); */
        /* prev_extractor->compute(prev_src_gray, prev_keypoints, */
        /*                         prev_descriptors); */

        /* drawKeypoints(copy, keypoints, copy, cv::Scalar(255, 0, 0),
         */

        /*               cv::DrawMatchesFlags::DRAW_OVER_OUTIMG); */

        int radius = 3;
        /* for (size_t i = 0; i < corners.size(); i++) { */
        /*     circle(copy, corners[i], radius, */
        /*            Scalar(rng.uniform(0, 255), rng.uniform(0, 256),
         */
        /*                   rng.uniform(0, 255)), */
        /*            FILLED); */
        /* } */

        //
        //  ________________________________________________________________

        /* cv::BFMatcher matcher(cv::NORM_L2, true); */
        /* std::vector<cv::DMatch> matches; */
        /* matcher.match(prev_descriptors, descriptors, matches); */

        // Sort matches by score
        /* std::sort(matches.begin(), matches.end()); */

        // Remove not so good matches
        /* const int numGoodMatches = matches.size() * 0.4f; */
        /* matches.erase(matches.begin() + numGoodMatches,
         * matches.end()); */

        /* cout << "match="; */
        /* cout << matches.size() << endl; */

        // the number of matched features between the two images
        /* if (matches.size() != 0) { */
        /* cout << matches.size() << endl; */
        /* cv::Mat imageMatches; */

        /* std::vector<char> mask(matches.size(), 1); */
        /* cv::drawMatches(prev_src_gray, prev_keypoints, src_gray,
         * keypoints,
         */
        /* matches, imageMatches, cv::Scalar::all(-1), */
        /* cv::Scalar::all(-1), mask, */
        /* cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS); */

        /* cv::drawMatches(prev_src_gray, prev_keypoints, src_gray,
         * keypoints,
         */
        /* matches, imageMatches); */
        /* cv::namedWindow("matches"); */
        /* cv::imshow("matches", imageMatches); */

        // */
        //________________________________________________________________

        /* cv::namedWindow(source_window); */
        /* cv::moveWindow(source_window, 0, 0); */
        /* imshow(source_window, src_gray); */

        prev_src_gray = src_gray;
    }
}

;

int main() {
    // initialize a video capture object
    //

    cv::VideoCapture vid_capture(
        "/home/omie_sawie/Code_Code/OmkarSawantBTP_SLAM_Photogrammetry/"

        "OpenCV_CPP/Photogrammetry_OpenCV_CPP/videoFeatureMatching/"

        "/resources/carOnLonelyRoads.mp4");

    // Print error message if the stream is invalid */
    if (!vid_capture.isOpened()) {
        cout << "Error opening video stream or file" << endl;
    }

    else {
        // Obtain fps and frame count by get() method and print
        // You can replace 5 with CAP_PROP_FPS as well, they are
        // enumerations
        int fps = vid_capture.get(5);
        cout << "Frames per second :" << fps;

        // Obtain frame_count using opencv built in frame count reading
        // method You can replace 7 with CAP_PROP_FRAME_COUNT as well,
        // they are enumerations
        int frame_count = vid_capture.get(7);
        cout << "  Frame count :" << frame_count;

        cv::Mat frame;
        // Initialize a boolean to check if frames are there or not

        bool isSuccess = vid_capture.read(frame);

        int down_width = 1900;
        int down_height = 900;

        // resize down
        resize(frame, src, cv::Size(down_width, down_height), cv::INTER_LINEAR);

        cv::cvtColor(src, prev_src_gray, cv::COLOR_BGR2GRAY);
    }

    while (vid_capture.isOpened()) {
        // Initialise frame matrix
        cv::Mat frame;
        // Initialize a boolean to check if frames are there or not

        bool isSuccess = vid_capture.read(frame);

        int down_width = 1900;
        int down_height = 900;

        // resize down
        resize(frame, src, cv::Size(down_width, down_height), cv::INTER_LINEAR);

        cv::cvtColor(src, src_gray, cv::COLOR_BGR2GRAY);
        cv::namedWindow(source_window);
        cv::namedWindow("matches");
        cv::moveWindow("matches", 0, 0);

        cv::setTrackbarPos("Max corners", source_window, maxCorners);

        imshow(source_window, src);

        FeatureExtractor feature_extractor;

        feature_extractor.extractFeatures_goodFeaturesToTrack(0, 0);

        // If frames are present, show it
        if (isSuccess == true) {
            // display frames
            /* imshow("Frame", frame); */
            /* imshow("PrevFrame", prevFrame); */
        }

        if (isSuccess == false) {
            cout << "Video camera is disconnected" << endl;
            break;
        }

        //-- Step 1: Detect the keypoints using SURF Detector,compute
        // the
        // descriptors

        // wait 20 ms between successive frames and break the loop ifkey
        // q is pressed
        int key = cv::waitKey(20);
        if (key == 'q') {
            cout << "q key is pressed by the user. Stopping the video" << endl;
            break;
        }
    }
    // Release the video capture object
    vid_capture.release();
    cv::destroyAllWindows();
    return 0;
}

// ________________________________________________________________

/* #include <iostream> */
/* #include <iterator> */
/* #include <opencv2/core.hpp> */
/* #include <opencv2/core/cuda.hpp> */

/* #include <opencv2/core/utility.hpp> */
/* #include <opencv2/dnn.hpp> */
/* #include <opencv2/highgui.hpp> */
/* #include <opencv2/highgui/highgui.hpp> */
/* #include <opencv2/imgcodecs.hpp> */
/* #include <opencv2/imgproc.hpp> */
/* #include <opencv2/imgproc/imgproc.hpp> */
/* #include <string> */

/* #include "bits/time.h" */
/* #include <cmath> */
/* #include <ctime> */

/* #include <opencv2/core.hpp> */
/* #include <opencv2/highgui.hpp> */
/* #include <opencv2/imgcodecs.hpp> */
/* #include <opencv2/imgproc.hpp> */

/* #include <opencv2/core/cuda.hpp> */
/* #include <opencv2/cudaarithm.hpp> */
/* #include <opencv2/cudaimgproc.hpp> */

/* /1* //
 * ________________________________________________________________ *1/
 */

/* using namespace std; */
/* using namespace cv; */
/* using namespace cv::cuda; */

/* int main() { */

/*     /1*     // Print cuda device details *1/ */
/*     cv::cuda::setDevice(0); */
/*     printCudaDeviceInfo(0); */

/*     // Read in image */
/*     std::string filename = */
/*         "/home/omie_sawie/Pictures/ArchLinux_SnowMountains.jpg"; */
/*     Mat image = imread(filename.c_str(), IMREAD_GRAYSCALE); */
/*     if (image.empty()) { */
/*         cout << "Could not open or find the image" << endl; */
/*         exit(-1); */
/*     } */
/*     cout << "image loaded: " << image.size() << endl; */

/*     /1* // Upload image to GPU *1/ */
/*     GpuMat gpuImage = GpuMat(image); */

/*     /1* // Set up feature detector *1/ */
/*     int maxCorners = 100; */
/*     double qualityLevel = 0.01; */
/*     double minDistance = 30; */
/*     int blockSize = 3; */
/*     bool useHarrisDetector = false; */
/*     double harrisK = 0.04; */
/*     cv::Ptr<cv::cuda::CornersDetector> cornerDetector = */
/*         cv::cuda::createGoodFeaturesToTrackDetector( */
/*             CV_8UC1, maxCorners, qualityLevel, minDistance,
 * blockSize, */
/*             useHarrisDetector, harrisK); */

/*     // Corners */
/*     cv::cuda::GpuMat corners; */

/*     // Detect corners */
/*     std::cout << "Detecting corners..." << std::endl; */
/*     cornerDetector->detect(gpuImage, corners); */

/*     // Print number of corners detected */
/*     std::cout << "Corners detected: " << corners.cols << std::endl;
 */

/*     return 0; */
/* } */
