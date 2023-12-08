#include <cstddef>
#include <iostream>
#include <iterator>
#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>
#include <opencv2/core/cvstd_wrapper.hpp>
#include <opencv2/core/hal/interface.h>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
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

int nfeatures = 1000;
float scaleFactor = 1.2f;
int nlevels = 16;
int edgeThreshold = 31;
int firstLevel = 0;
int WTA_K = 2;
int scoreType = cv::ORB::HARRIS_SCORE;
int patchSize = 31;
int fastThreshold = 20;
bool blurForDescriptor = false;

cv::Mat src, src_gray, prev_src_gray;
cv::cuda::GpuMat src_gray_gpu, prev_src_gray_gpu;

class FeatureExtractor {
  public:
    cv::Ptr<cv::cuda::ORB> extractor = cv::cuda::ORB::create(
        nfeatures, scaleFactor, nlevels, edgeThreshold, firstLevel, WTA_K,
        scoreType, patchSize, fastThreshold, blurForDescriptor);

    FeatureExtractor(){};

    void extractFeatures_goodFeaturesToTrack(int, void *) {

        cv::cuda::GpuMat src_gray_gpu = cv::cuda::GpuMat(src_gray);

        if (!prev_src_gray_gpu.empty()) {
            cout << " Size of image:" << prev_src_gray_gpu.size() << endl;
            maxCorners = MAX(maxCorners, 1000);
            double qualityLevel = 0.01;
            double minDistance = 0;
            int blockSize = 10, gradientSize = 3;
            bool useHarrisDetector = false;
            double k = 0.04;
            cv::cuda::GpuMat copy = src_gray_gpu.clone();
            double harrisK = 0.04;

            cv::cuda::GpuMat descriptors_gpu, prev_descriptors_gpu,
                keypoints_gpu, prev_keypoints_gpu;

            extractor->detectAndComputeAsync(src_gray_gpu, cv::cuda::GpuMat(),
                                             keypoints_gpu, descriptors_gpu);
            extractor->detectAndComputeAsync(
                prev_src_gray_gpu, cv::cuda::GpuMat(), prev_keypoints_gpu,
                prev_descriptors_gpu);

            std::cout << "Descriptors Size " << descriptors_gpu.size() << endl;

            // Match the features
            cv::Ptr<cv::cuda::DescriptorMatcher> matcher =
                cv::cuda::DescriptorMatcher::createBFMatcher(cv::NORM_HAMMING);
            vector<vector<cv::DMatch>> matches;
            matcher->knnMatch(descriptors_gpu, prev_descriptors_gpu, matches,
                              2);
            // Filter the good matches
            std::vector<cv::DMatch> good_matches;
            for (int k = 0;
                 k < min(descriptors_gpu.rows - 1, (int)matches.size()); k++) {
                if ((matches[k][0].distance < 0.8 * (matches[k][1].distance))) {
                    good_matches.push_back(matches[k][0]);
                    /* cout << matches[k][1].queryIdx << " " */
                    /*      << matches[k][1].trainIdx << " "; */
                }
            }
            cout << endl;
            cout << "MatchesSize: " << good_matches.size() << endl;

            /* t_BBtime = getTickCount(); */
            /* t_pt = (t_BBtime - t_AAtime) / getTickFrequency(); */
            /* t_fpt = 1 / t_pt; */
            /* printf("%.4lf sec/ %.4lf fps\n", t_pt, t_fpt); */

            // Download the Keypoints from GPU
            vector<cv::KeyPoint> keypoints_cpu, prev_keypoints_cpu;
            extractor->convert(keypoints_gpu, keypoints_cpu);
            extractor->convert(prev_keypoints_gpu, prev_keypoints_cpu);
            cout << "Keypoints Size :" << keypoints_cpu[100].size << endl;

            // Draw the Keypoints on the source gray image
            drawKeypoints(src_gray, keypoints_cpu, src_gray,
                          cv::Scalar(255, 0, 0),
                          cv::DrawMatchesFlags::DRAW_OVER_OUTIMG);
            cv::imshow("Kps", src_gray);

            // Draw the Matches
            cv::Mat img_matches;
            drawMatches(src, keypoints_cpu, prev_src_gray, prev_keypoints_cpu,
                        good_matches, img_matches);
            cv::namedWindow("matches", 0);
            imshow("matches", img_matches);

            // Estimating the Fundamental Matrix
            int point_count = 100;
            vector<cv::Point2f> points1(point_count);
            vector<cv::Point2f> points2(point_count);
            // initialize the points here ...
            for (int i = 0; i < point_count; i++) {
                points1[i] = keypoints_cpu[i].pt;
                points2[i] = prev_keypoints_cpu[i].pt;
            }
            cout << cv::findFundamentalMat(points1, points2, cv::FM_RANSAC, 3,
                                           0.95)
                 << endl;

        } // End of if statement

        src_gray_gpu.copyTo(prev_src_gray_gpu);
        src_gray.copyTo(prev_src_gray);
        cout << "Size 1" << prev_src_gray.size() << endl;
    }

    void estimateFundamentalMatrix() {}
}

;

int main() {
    /* cv::cuda::setDevice(0); */
    /* cv::cuda::printCudaDeviceInfo(0); */

    // initialize a video capture object
    cv::VideoCapture vid_capture(
        "/home/omie_sawie/Code_Code/OmkarSawantBTP_SLAM_Photogrammetry/"

        "OpenCV_CPP/Photogrammetry_OpenCV_CPP/videoFeatureMatching/"

        "/resources/carOnLonelyRoads.mp4");

    // Print error message if the stream is invalid */
    if (!vid_capture.isOpened()) {
        cout << "Error opening video stream or file" << endl;
    }

    while (vid_capture.isOpened()) {
        // Initialise frame matrix
        // Initialize a boolean to check if frames are there or not
        cv::Mat frame;
        bool isSuccess = vid_capture.read(frame);

        src = frame.clone();

        cv::cvtColor(src, src_gray, cv::COLOR_RGB2GRAY);

        /* cv::setTrackbarPos("Max corners", source_window, maxCorners); */

        /* imshow(source_window, src); */

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
        int key = cv::waitKey(1000);
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
