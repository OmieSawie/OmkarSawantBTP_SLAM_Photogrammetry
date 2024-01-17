#include <complex>
#include <cstddef>
#include <iostream>
#include <iterator>
#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>
#include <opencv2/core/base.hpp>
#include <opencv2/core/cvstd_wrapper.hpp>
#include <opencv2/core/hal/interface.h>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/matx.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/video/tracking.hpp>
#include <opencv2/videoio.hpp>
#include <streambuf>
#include <unistd.h>
#include <vector>

#include <cmath>
#include <ctime>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudafeatures2d.hpp>
#include <opencv2/cudaimgproc.hpp>

using namespace std;

cv::RNG rng(12345);
const char *source_window = "Image";

int nfeatures = 100000;
float scaleFactor = 1.2f;
int nlevels = 16;
int edgeThreshold = 31;
int firstLevel = 0;
int WTA_K = 2;
int scoreType = cv::ORB::HARRIS_SCORE;
int patchSize = 31;
int fastThreshold = 20;
bool blurForDescriptor = true;

cv::Mat src, src_gray, prev_src_gray;
cv::cuda::GpuMat src_gray_gpu, prev_src_gray_gpu;

class FeatureExtractor {
  public:
    cv::Ptr<cv::cuda::ORB> extractor = cv::cuda::ORB::create(
        nfeatures, scaleFactor, nlevels, edgeThreshold, firstLevel, WTA_K,
        scoreType, patchSize, fastThreshold, blurForDescriptor);

    cv::Mat cameraMatrix = (cv::Mat_<float>(3, 3) << 3048., 0., 2312., 0.,
                            2046., 1734., 0., 0., 1.);

    /* FeatureExtractor(){}; */

    cv::Mat computeM(cv::Mat rot, cv::Mat trans) {
        cv::Mat temp = (cv::Mat_<float>(3, 1) << 0., 0., 0.);
        cv::Mat hom_cameraMatrix;
        cv::hconcat(cameraMatrix, temp, hom_cameraMatrix);
        cout << "Camera Matrix Homogenised: " << hom_cameraMatrix << endl;
        return hom_cameraMatrix;
    }

    cv::Mat comptueP(cv::Mat rot, cv::Mat trans) {
        cv::Mat hom_cameraMatrix = computeM(rot, trans);
        cv::Mat rpt, hom_rpt;
        cv::hconcat(rot, trans, rpt);
        rpt.convertTo(rpt, CV_32F);
        cv::Mat temp = (cv::Mat_<float>(1, 4) << 0., 0., 0., 0.);
        cv::vconcat(rpt, temp, hom_rpt);
        cout << "RPT homogenised: " << hom_rpt << endl;
        cout << "cam*rpt: " << hom_cameraMatrix * hom_rpt << endl;
        return hom_cameraMatrix * hom_rpt;
    }

    /* cv::Mat computingDepth(cv::Mat rot, cv::Mat trans) { */
    /*     cv::Mat temp = (cv::Mat_<float>(3, 1) << 0., 0., 0.); */
    /*     cv::Mat hom_cameraMatrix; */
    /*     cv::hconcat(cameraMatrix, temp, hom_cameraMatrix); */
    /*     cout << "Camera Matrix Homogenised: " << hom_cameraMatrix << endl; */

    /*     cv::Mat rpt, hom_rpt; */
    /*     cv::hconcat(rot, trans, rpt); */
    /*     rpt.convertTo(rpt, CV_32F); */
    /*     temp = (cv::Mat_<float>(1, 4) << 0., 0., 0., 0.); */
    /*     cv::vconcat(rpt, temp, hom_rpt); */
    /*     cout << "RPT homogenised: " << hom_rpt << endl; */
    /*     cout << "cam*rpt: " << hom_cameraMatrix * hom_rpt << endl; */
    /*     cv::Mat returnAns = {hom_cameraMatrix, hom_cameraMatrix * hom_rpt};
     */
    /*     return returnAns; */
    /* } */

    void solnFinalEquation(cv::Point2f r, cv::Point2f l, cv::Mat M, cv::Mat P) {
        float ur = r.x, ul = l.x, vr = r.y, vl = l.y;

        float m11, m12, m13, m14, m21, m22, m23, m24, m31, m32, m33, m34, m41,
            m42, m43, m44;
        float p11, p12, p13, p14, p21, p22, p23, p24, p31, p32, p33, p34, p41,
            p42, p43, p44;

        m11 = M.at<float>(0, 0);
        m12 = M.at<float>(0, 1);
        m13 = M.at<float>(0, 2);
        m14 = M.at<float>(0, 3);
        m21 = M.at<float>(1, 0);
        m22 = M.at<float>(1, 1);
        m23 = M.at<float>(1, 2);
        m24 = M.at<float>(1, 3);
        m31 = M.at<float>(2, 0);
        m32 = M.at<float>(2, 1);
        m33 = M.at<float>(2, 2);
        m34 = M.at<float>(2, 3);

        p11 = P.at<float>(0, 0);
        p12 = P.at<float>(0, 1);
        p13 = P.at<float>(0, 2);
        p14 = P.at<float>(0, 3);
        p21 = P.at<float>(1, 0);
        p22 = P.at<float>(1, 1);
        p23 = P.at<float>(1, 2);
        p24 = P.at<float>(1, 3);
        p31 = P.at<float>(2, 0);
        p32 = P.at<float>(2, 1);
        p33 = P.at<float>(2, 2);
        p34 = P.at<float>(2, 3);

        cout << "ur= " << ur << endl;
        cout << "m33= " << m33 << endl;
        cout << "m13= " << m13 << endl;

        cout << "M: " << M << endl;

        cv::Mat finl =
            (cv::Mat_<float>(4, 3) << ur * m31 - m11, ur * m32 - m12,
             ur * m33 - m13, vr * m31 - m21, vr * m32 - m22, vr * m33 - m23,
             ul * p31 - p11, ul * p32 - p12, ul * p33 - p13, vl * p31 - p21,
             vl * p32 - p22, vl * p33 - p23);
        cv::Mat finr = (cv::Mat_<float>(4, 1) << m14 - m34, m24 - m34,
                        p14 - p34, p24 - p34);
        cout << "finl and finr: " << finl << endl << finr << endl;

        cv::Mat mulTransposedMat, mulTransposedMatInverted, transposedMat;
        cv::mulTransposed(finl, mulTransposedMat, true);
        cout << "mulTransposed: " << mulTransposedMat << endl;
        cv::invert(mulTransposedMat, mulTransposedMatInverted);
        cv::transpose(finl, transposedMat);
        cout << "mulTransposedMatInverted: " << mulTransposedMatInverted
             << endl;
        cv::Mat answer = mulTransposedMatInverted * transposedMat * finr;
        if (answer.at<float>(0) == answer.at<float>(0)) {
            cout << "x,y,z: " << answer << endl;
        }
    }

    void extractFeatures_goodFeaturesToTrack(int, void *) {
        cv::cuda::GpuMat src_gray_gpu = cv::cuda::GpuMat(src_gray);
        cv::cuda::GpuMat prev_src_gray_gpu = cv::cuda::GpuMat(prev_src_gray);

        if (!prev_src_gray_gpu.empty()) {
            int img_width = src_gray_gpu.size().width,
                img_height = src_gray_gpu.size().height;
            cout << " Size of image: " << img_width << " x " << img_height
                 << endl;
            double qualityLevel = 0.01;
            double minDistance = 0;
            int blockSize = 10, gradientSize = 3;
            bool useHarrisDetector = false;
            double k = 0.04;
            double harrisK = 0.04;

            cv::cuda::GpuMat descriptors_gpu, prev_descriptors_gpu,
                keypoints_gpu, prev_keypoints_gpu, matches_gpu;
            std::vector<cv::KeyPoint> keypoints_cpu, prev_keypoints_cpu;

            // Detect and Compute features
            // prev_src_gray is	#1 query image
            // src_gray is the  #2 train image
            /* extractor->detectAndCompute(prev_src_gray_gpu,
             * cv::noArray(), */
            /*                             prev_keypoints_cpu, */
            /*                             prev_descriptors_gpu); */
            /* extractor->detectAndCompute(src_gray_gpu, cv::noArray(),
             */
            /*                             keypoints_cpu,
             * descriptors_gpu); */
            extractor->detectAndComputeAsync(prev_src_gray_gpu, cv::noArray(),
                                             prev_keypoints_gpu,
                                             prev_descriptors_gpu, false);

            extractor->detectAndComputeAsync(src_gray_gpu, cv::noArray(),
                                             keypoints_gpu, descriptors_gpu,
                                             false);

            std::cout << "Descriptors Size " << descriptors_gpu.size() << endl;

            // Match the features
            cv::Ptr<cv::cuda::DescriptorMatcher> matcher =
                cv::cuda::DescriptorMatcher::createBFMatcher(cv::NORM_HAMMING);
            vector<vector<cv::DMatch>> matches;
            /* matcher->knnMatch(prev_descriptors_gpu, descriptors_gpu,
             * matches,
             */
            /*                   2); */
            matcher->knnMatchAsync(prev_descriptors_gpu, descriptors_gpu,
                                   matches_gpu, 2);
            matcher->knnMatchConvert(matches_gpu, matches, false);
            // Filter the good matches
            std::vector<cv::DMatch> good_matches;
            for (int k = 0; k < (int)matches.size(); k++) {
                if ((matches[k][0].distance < 0.7 * (matches[k][1].distance))) {
                    good_matches.push_back(matches[k][0]);
                    /* cout << matches[k][0].distance << " " */
                    /*      << matches[k][1].distance << " " << endl; */
                }
            }
            cout << endl;
            cout << "MatchesSize: " << good_matches.size() << endl;

            /* t_BBtime = getTickCount(); */
            /* t_pt = (t_BBtime - t_AAtime) / getTickFrequency(); */
            /* t_fpt = 1 / t_pt; */
            /* printf("%.4lf sec/ %.4lf fps\n", t_pt, t_fpt); */

            // Download the Keypoints from GPU
            extractor->convert(keypoints_gpu, keypoints_cpu);
            extractor->convert(prev_keypoints_gpu, prev_keypoints_cpu);
            cout << "Keypoints Size :" << keypoints_cpu.size() << endl;

            // Draw the Keypoints on the source gray image
            /* drawKeypoints(src_gray, keypoints_cpu, src_gray, */
            /*               cv::Scalar(255, 0, 190), */
            /*               cv::DrawMatchesFlags::DRAW_OVER_OUTIMG); */
            /* drawKeypoints(prev_src_gray, prev_keypoints_cpu,
             * prev_src_gray,
             */
            /*               cv::Scalar(190, 0, 190), */
            /*               cv::DrawMatchesFlags::DRAW_OVER_OUTIMG); */
            /* cv::imshow("Kps", src_gray); */

            // Draw the Matches
            cv::Mat img_matches;
            /* drawMatches(prev_src_gray, prev_keypoints_cpu, src_gray,
             */
            /*             keypoints_cpu, good_matches, img_matches); */
            /* cv::namedWindow("matches", 0); */
            /* imshow("matches", img_matches); */

            // Estimating the Fundamental Matrix
            int point_count = good_matches.size();
            vector<cv::Point2f> points1(point_count);
            vector<cv::Point2f> points2(point_count);
            // initialize the points here ...
            for (int i = 0; i < point_count; i++) {
                points1[i] = prev_keypoints_cpu[good_matches[i].queryIdx].pt;
                points2[i] = keypoints_cpu[good_matches[i].trainIdx].pt;
            }
            cv::Mat fund_mask;
            cv::Mat fundamentalMatrix = cv::findFundamentalMat(
                points1, points2, cv::FM_RANSAC, 3., 0.9, fund_mask);
            /*  cout << fundamentalMatrix << endl; */
            /* cout << fund_mask; */
            vector<cv::Point2f> points1_masked, points2_masked;
            for (int i = 0; i < point_count; i++) {
                /* if (fund_mask.at<int>(i) == 1) { */
                points1_masked.push_back(points1[i]);
                points2_masked.push_back(points2[i]);
                /* } */
            }
            cout << "masked points count: " << points1_masked.size() << endl;

            // Compute epilines
            vector<cv::Vec3f> epilines1, epilines2;
            cv::computeCorrespondEpilines(points1_masked, 1, fundamentalMatrix,
                                          epilines2);
            cv::computeCorrespondEpilines(points2_masked, 2, fundamentalMatrix,
                                          epilines1);
            /* cout << epilines << endl; */

            if (false) {
                for (vector<cv::Vec3f>::const_iterator it = epilines1.begin();
                     it != epilines1.end(); ++it) {
                    // draw the line between first and last column
                    cv::line(prev_src_gray, cv::Point(0, -(*it)[2] / (*it)[1]),
                             cv::Point(4624., -((*it)[2] + (*it)[0] * 4624.) /
                                                  ((*it)[1])),
                             cv::Scalar(255, 255, 255), cv::LINE_4);
                }
                for (vector<cv::Vec3f>::const_iterator it = epilines2.begin();
                     it != epilines2.end(); ++it) {
                    // draw the line between first and last column
                    cv::line(src_gray, cv::Point(0, -(*it)[2] / (*it)[1]),
                             cv::Point(4624., -((*it)[2] + (*it)[0] * 4624.) /
                                                  ((*it)[1])),
                             cv::Scalar(255, 255, 255), cv::LINE_4);
                }
                // Draw trhe masked keypoints
                for (int i = 0; i < points1_masked.size(); i++) {
                    cv::circle(prev_src_gray, points1_masked[i], 10,
                               cv::Scalar(255, 0, 255), 10, cv::LINE_8, 0);
                    cv::circle(src_gray, points2_masked[i], 10,
                               cv::Scalar(255, 0, 255), 10, cv::LINE_8, 0);
                }
            }
            /* cv::imshow("Right Image Epilines (FM_7POINT2)",
             * src_gray); */
            /* cv::imshow("Right Image Epilines (FM_7POINT1)",
             * prev_src_gray);
             */

            drawMatches(prev_src_gray, prev_keypoints_cpu, src_gray,

                        keypoints_cpu, good_matches, img_matches);
            cv::namedWindow("matches", 0);
            imshow("matches", img_matches);
            cv::imshow("src_gray", src_gray);
            cv::imshow("prev_src_gray", prev_src_gray);

            // Estimating the Essential Matrix

            cv::Mat essentialMatrix;
            essentialMatrix = cv::findEssentialMat(
                points1, points2, cameraMatrix, cv::RANSAC, 0.9, 1.0, 10000);
            /* cout << essentialMatrix << endl; */
            cv::Mat R1, R2, t, R1R, R2R;
            cv::decomposeEssentialMat(essentialMatrix, R1, R2, t);
            cout << "R1" << R1 << endl
                 << "R2" << R2 << endl
                 << "t" << t << endl;
            cv::Rodrigues(R1, R1R);
            cv::Rodrigues(R2, R2R);
            cout << "R1R" << R1R << endl << "R2R" << R2R << endl;

            cv::Mat P = comptueP(R1, t);
            cv::Mat M = computeM(R1, t);

            for (int i = 0; i < point_count; i++) {
                solnFinalEquation(points2[i], points1[i], M, P);
            }

        } // End of if statement

        src_gray_gpu.copyTo(prev_src_gray_gpu);
        src_gray.copyTo(prev_src_gray);
        /* cout << "Size 1" << prev_src_gray.size() << endl; */
    }

    void estimateFundamentalMatrix() {}
}

;

int main() {
    cv::cuda::setDevice(0);
    /* cv::cuda::printCudaDeviceInfo(0); */

    // initialize a video capture object
    /* cv::VideoCapture vid_capture( */
    /*     "/home/omie_sawie/Code_Code/OmkarSawantBTP_SLAM_Photogrammetry/"
     */

    /*     "OpenCV_CPP/Photogrammetry_OpenCV_CPP/videoFeatureMatching/" */

    /*     "/resources/carOnLonelyRoads.mp4"); */

    /* cv::VideoCapture vid_capture("../testVideo1.mp4"); */

    /* // Print error message if the stream is invalid *1/ */
    /* if (!vid_capture.isOpened()) { */
    /*     cout << "Error opening video stream or file" << endl; */
    /* } */

    /* while (vid_capture.isOpened()) { */
    // Initialise frame matrix
    // Initialize a boolean to check if frames are there or not
    /* cv::Mat frame; */

    /* bool isSuccess = vid_capture.read(frame); */
    /* src = frame.clone(); */

    src_gray = cv::imread("../testimgR.jpg", cv::IMREAD_GRAYSCALE);
    prev_src_gray = cv::imread("../testimgL.jpg", cv::IMREAD_GRAYSCALE);

    /* cv::cvtColor(src, src_gray, cv::COLOR_RGB2GRAY); */

    /* cv::setTrackbarPos("Max corners", source_window, maxCorners); */

    /* imshow(source_window, src); */

    FeatureExtractor feature_extractor;

    feature_extractor.extractFeatures_goodFeaturesToTrack(0, 0);

    /* // If frames are present, show it */
    /* if (isSuccess == true) { */
    /*     // display frames */
    /*     /1* imshow("Frame", frame); *1/ */
    /*     /1* imshow("PrevFrame", prevFrame); *1/ */
    /* } */

    /* if (isSuccess == false) { */
    /*     cout << "Video camera is disconnected" << endl; */
    /*     break; */
    /* } */

    //-- Step 1: Detect the keypoints using SURF Detector,compute
    // the
    // descriptors

    // wait 20 ms between successive frames and break the loop ifkey
    // q is pressed
    while (true) {
        int key = cv::waitKey();
        if (key == 'q') {
            cout << "q key is pressed by the user. Stopping the "
                    "video"
                 << endl;
            break;
        }
    }
    /* } */
    // Release the video capture object
    /* vid_capture.release(); */
    cv::destroyAllWindows();
    return 0;
}
