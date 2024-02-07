#include <complex>
#include <cstddef>
#include <iostream>
#include <iterator>
#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>
#include <opencv2/core/base.hpp>
#include <opencv2/core/cvdef.h>
#include <opencv2/core/cvstd.hpp>
#include <opencv2/core/cvstd_wrapper.hpp>
#include <opencv2/core/hal/interface.h>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/matx.hpp>
#include <opencv2/core/persistence.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/video/tracking.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/viz/vizcore.hpp>

#include <ostream>
#include <streambuf>
#include <string>
#include <unistd.h>
#include <vector>

#include <cmath>
#include <ctime>

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

int negativeZCount = 0;

cv::Mat src, src_gray, prev_src_gray;

vector<string> imageList;

class FeatureExtractor {
  public:
    cv::Ptr<cv::SIFT> extractor =
        cv::SIFT::create(100000, 3, 0.09, 31, 1.2f, CV_8U, false);

    cv::Mat cameraMatrix = (cv::Mat_<double>(3, 3) << 3658.26, 0., 2311.5, 0.,
                            3658.26, 1734., 0., 0., 1.);
    /* cv::Mat cameraMatrix = (cv::Mat_<double>(3, 3) << 3048., 0., 2312., 0.,
     */
    /* 2046., 1734., 0., 0., 1.); */

    cv::Mat PointCloudMatrix = cv::Mat(0, 0, CV_64F),
            PointColorsMatrix = cv::Mat(0, 0, CV_8UC3);

    cv::Mat baseRot =
        (cv::Mat_<double>(3, 3, CV_64F) << 1, 0, 0, 0, 1, 0, 0, 0, 1);
    cv::Mat baseTrans = (cv::Mat_<double>(3, 1, CV_64F) << 0, 0, 0);

    cv::Mat baseRPT = (cv::Mat_<double>(4, 4, CV_64F) << 1, 0, 0, 0, 0, 1, 0, 0,
                       0, 0, 1, 0, 0, 0, 0, 1);

    static bool readStringList(const string &filename, vector<string> &l) {
        l.clear();
        cv::FileStorage fs(filename, cv::FileStorage::READ);
        if (!fs.isOpened())
            return false;
        cv::FileNode n = fs.getFirstTopLevelNode();
        if (n.type() != cv::FileNode::SEQ)
            return false;
        cv::FileNodeIterator it = n.begin(), it_end = n.end();
        for (; it != it_end; ++it)
            l.push_back((string)*it);
        return true;
    }
    cv::Mat computeM(cv::Mat rot, cv::Mat trans) {
        cv::Mat temp = (cv::Mat_<double>(3, 1) << 0., 0., 0.);
        cv::Mat hom_cameraMatrix;
        cv::hconcat(cameraMatrix, temp, hom_cameraMatrix);
        /* cout << "Camera Matrix Homogenised: " << hom_cameraMatrix << endl; */
        return hom_cameraMatrix;
    }
    cv::Mat composeHomRPT(cv::Mat rot, cv::Mat trans) {
        cv::Mat rpt, hom_rpt;
        cv::hconcat(rot, trans, rpt);
        cv::Mat temp = (cv::Mat_<double>(1, 4) << 0., 0., 0., 1.);
        cv::vconcat(rpt, temp, hom_rpt);
        cout << "Hom RPT" << hom_rpt << endl;
        return hom_rpt;
    }

    cv::Mat comptueP(cv::Mat rot, cv::Mat trans) {
        cv::Mat hom_cameraMatrix = computeM(rot, trans);
        cv::Mat hom_rpt = composeHomRPT(rot, trans);
        baseRPT = baseRPT * hom_rpt;
        cout << "Base RPT" << baseRPT << endl;
        /* cout << "RPT homogenised: " << hom_rpt << endl; */
        /* cout << "cam*rpt: " << hom_cameraMatrix * hom_rpt << endl; */
        return hom_cameraMatrix * hom_rpt;
    }

    /* cv::Mat computingDepth(cv::Mat rot, cv::Mat trans) { */
    /*     cv::Mat temp = (cv::Mat_<double>(3, 1) << 0., 0., 0.); */
    /*     cv::Mat hom_cameraMatrix; */
    /*     cv::hconcat(cameraMatrix, temp, hom_cameraMatrix); */
    /*     cout << "Camera Matrix Homogenised: " << hom_cameraMatrix << endl; */

    /*     cv::Mat rpt, hom_rpt; */
    /*     cv::hconcat(rot, trans, rpt); */
    /*     rpt.convertTo(rpt, CV_32F); */
    /*     temp = (cv::Mat_<double>(1, 4) << 0., 0., 0., 0.); */
    /*     cv::vconcat(rpt, temp, hom_rpt); */
    /*     cout << "RPT homogenised: " << hom_rpt << endl; */
    /*     cout << "cam*rpt: " << hom_cameraMatrix * hom_rpt << endl; */
    /*     cv::Mat returnAns = {hom_cameraMatrix, hom_cameraMatrix * hom_rpt};
     */
    /*     return returnAns; */
    /* } */

    cv::Mat getLocalCoordinates(cv::Point2f r, cv::Point2f l, cv::Mat M,
                                cv::Mat P) {
        /* cout << "getting local coordinates " << l.x << " " << l.y << endl; */

        cv::Mat colors(1, 1, CV_8UC3, cv::Scalar(src.at<cv::Vec3b>(l.y, l.x)));
        PointColorsMatrix.push_back(colors);

        double ur = r.x, ul = l.x, vr = r.y, vl = l.y;

        double m11, m12, m13, m14, m21, m22, m23, m24, m31, m32, m33, m34, m41,
            m42, m43, m44;
        double p11, p12, p13, p14, p21, p22, p23, p24, p31, p32, p33, p34, p41,
            p42, p43, p44;

        m11 = M.at<double>(0, 0);
        m12 = M.at<double>(0, 1);
        m13 = M.at<double>(0, 2);
        m14 = M.at<double>(0, 3);
        m21 = M.at<double>(1, 0);
        m22 = M.at<double>(1, 1);
        m23 = M.at<double>(1, 2);
        m24 = M.at<double>(1, 3);
        m31 = M.at<double>(2, 0);
        m32 = M.at<double>(2, 1);
        m33 = M.at<double>(2, 2);
        m34 = M.at<double>(2, 3);

        p11 = P.at<double>(0, 0);
        p12 = P.at<double>(0, 1);
        p13 = P.at<double>(0, 2);
        p14 = P.at<double>(0, 3);
        p21 = P.at<double>(1, 0);
        p22 = P.at<double>(1, 1);
        p23 = P.at<double>(1, 2);
        p24 = P.at<double>(1, 3);
        p31 = P.at<double>(2, 0);
        p32 = P.at<double>(2, 1);
        p33 = P.at<double>(2, 2);
        p34 = P.at<double>(2, 3);

        /* cout << "ur= " << ur << endl; */
        /* cout << "m33= " << m33 << endl; */
        /* cout << "m13= " << m13 << endl; */

        /* cout << "M: " << M << endl; */

        cv::Mat finl =
            (cv::Mat_<double>(4, 3) << ur * m31 - m11, ur * m32 - m12,
             ur * m33 - m13, vr * m31 - m21, vr * m32 - m22, vr * m33 - m23,
             ul * p31 - p11, ul * p32 - p12, ul * p33 - p13, vl * p31 - p21,
             vl * p32 - p22, vl * p33 - p23);
        cv::Mat finr = (cv::Mat_<double>(4, 1) << m14 - m34, m24 - m34,
                        p14 - p34, p24 - p34);
        /* cout << "finl and finr: " << finl << endl << finr << endl; */

        cv::Mat mulTransposedMat, mulTransposedMatInverted, transposedMat;
        cv::mulTransposed(finl, mulTransposedMat, true);
        /* cout << "mulTransposed: " << mulTransposedMat << endl; */
        cv::invert(mulTransposedMat, mulTransposedMatInverted);
        cv::transpose(finl, transposedMat);
        /* cout << "mulTransposedMatInverted: " << mulTransposedMatInverted */
        /* << endl; */
        cv::Mat answer = mulTransposedMatInverted * transposedMat * finr;
        /* cout << "answer" << answer << endl; */

        return answer;
    }

    void getWorldCoordinates(cv::Mat baseTrans, cv::Mat baseRot,
                             cv::Mat localCoordinates) {
        cv::Mat worldCoordinates = cv::Mat_<double>(3, 3, CV_64F);

        cv::Mat baseRotInv;
        /* cout << "baseRot" << baseRot << endl; */
        cv::transpose(baseRot, baseRotInv);
        /* worldCoordinates = baseRotInv * worldCoordinates; */
        worldCoordinates = localCoordinates + baseTrans;
        /* cout << "worldCoordinates" << worldCoordinates << endl; */
        cv::Point3d pointCoordinate;
        pointCoordinate.x = worldCoordinates.at<double>(0);
        pointCoordinate.y = worldCoordinates.at<double>(1);
        pointCoordinate.z = worldCoordinates.at<double>(2);

        PointCloudMatrix.push_back(pointCoordinate);
    }

    void writeCloudToFile(string filename, cv::Mat PointCloudMatrix,
                          cv::Mat PointColorsMatrix) {
        cv::viz::writeCloud(filename, PointCloudMatrix, PointColorsMatrix);
    }

    void extractFeatures_goodFeaturesToTrack(int, void *) {
        /* cv::cuda::GpuMat src_gray_gpu = cv::cuda::GpuMat(src_gray); */
        /* cv::cuda::GpuMat prev_src_gray_gpu = cv::cuda::GpuMat(prev_src_gray);
         */

        if (!prev_src_gray.empty()) {
            int img_width = src_gray.size().width,
                img_height = src_gray.size().height;
            cout << " Size of image: " << img_width << " x " << img_height
                 << endl;
            double qualityLevel = 0.01;
            double minDistance = 0;
            int blockSize = 10, gradientSize = 3;
            bool useHarrisDetector = false;
            double k = 0.04;
            double harrisK = 0.04;

            cv::Mat descriptors_cpu, prev_descriptors_cpu;
            std::vector<cv::KeyPoint> keypoints_cpu, prev_keypoints_cpu;

            // Detect and Compute features
            // prev_src_gray is	#1 query image
            // src_gray is the  #2 train image

            extractor->detectAndCompute(src_gray, cv::noArray(), keypoints_cpu,
                                        descriptors_cpu);
            extractor->detectAndCompute(prev_src_gray, cv::noArray(),
                                        prev_keypoints_cpu,
                                        prev_descriptors_cpu);

            std::cout << "Descriptors Size " << descriptors_cpu.size() << endl;

            // Match the features
            cv::Ptr<cv::DescriptorMatcher> matcher =
                cv::DescriptorMatcher::create("BruteForce");
            vector<vector<cv::DMatch>> matches;

            matcher->knnMatch(prev_descriptors_cpu, descriptors_cpu, matches,
                              2);
            // Filter the good matches
            std::vector<cv::DMatch> good_matches;
            for (int k = 0; k < (int)matches.size(); k++) {
                if ((matches[k][0].distance <
                     0.75 * (matches[k][1].distance))) {
                    good_matches.push_back(matches[k][0]);
                }
            }
            cout << endl;
            cout << "MatchesSize: " << good_matches.size() << endl;

            /* t_BBtime = getTickCount(); */
            /* t_pt = (t_BBtime - t_AAtime) / getTickFrequency(); */
            /* t_fpt = 1 / t_pt; */
            /* printf("%.4lf sec/ %.4lf fps\n", t_pt, t_fpt); */

            // Download the Keypoints from GPU
            /* extractor->convert(keypoints_gpu, keypoints_cpu); */
            /* extractor->convert(prev_keypoints_gpu, prev_keypoints_cpu); */
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
                points1, points2, cv::FM_RANSAC, 3., 0.90, fund_mask);
            /*  cout << fundamentalMatrix << endl; */
            /* cout << fund_mask; */
            /* cv::correctMatches(fundamentalMatrix, points1, points2, points1,
             */
            /*                    points2); */
            vector<cv::Point2f> points1_masked, points2_masked;
            for (int i = 0; i < point_count; i++) {
                if (fund_mask.at<int>(i) == 1) {
                    points1_masked.push_back(points1[i]);
                    points2_masked.push_back(points2[i]);
                }
            }
            cout << "masked points count: " << points1_masked.size() << endl;

            // Compute epilines
            vector<cv::Vec3f> epilines1, epilines2;
            cv::computeCorrespondEpilines(points1_masked, 1, fundamentalMatrix,
                                          epilines2);
            cv::computeCorrespondEpilines(points2_masked, 2, fundamentalMatrix,
                                          epilines1);
            /* cout << epilines << endl; */

            if (true) {
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
            cv::imwrite("Matches.jpg", img_matches);

            /* cv::resize(src_gray, src_gray, cv::Size(), 0.25, 0.25); */
            /* cv::resize(prev_src_gray, prev_src_gray, cv::Size(), 0.25, 0.25);
             */
            /* cv::imshow("src_gray", src_gray); */
            /* cv::imshow("prev_src_gray", prev_src_gray); */

            // Estimating the Essential Matrix

            cv::Mat essentialMatrix;
            essentialMatrix = cv::findEssentialMat(
                points1, points2, cameraMatrix, cv::RANSAC, 0.99, 1.0, 10000);
            /* cout << essentialMatrix << endl; */
            cv::Mat R1, R2, t, R1R, R2R;
            cv::decomposeEssentialMat(essentialMatrix, R1, R2, t);
            /* cout << "R1" << R1 << endl << "R2" << R2 << endl; */
            if (t.at<double>(0) < 0.) {
                t = -t;
            }
            cv::Rodrigues(R1, R1R);
            cv::Rodrigues(R2, R2R);

            cout << "t" << t << endl
                 << "R1R" << R1R << endl
                 << "R2R" << R2R << endl;
            cv::Mat correctRot;
            if (abs(R1R.at<double>(0)) < abs(R2R.at<double>(0))) {
                correctRot = R1;
                cout << "R1 is Correct" << endl;
            } else {
                correctRot = R2;
                cout << "R2 is Correct" << endl;
            }

            cv::Mat P = comptueP(correctRot, t);
            cv::Mat M = computeM(correctRot, t);

            for (int i = 0; i < point_count; i++) {
                cv::Mat localCoordinates =
                    getLocalCoordinates(points2[i], points1[i], M, P);
                getWorldCoordinates(baseTrans, baseRot, localCoordinates);
            }
            cout << "REached" << endl;

            /* cv::Mat temp = (cv::Mat_<double>(1, 3) << 0., 0., 0.); */
            baseTrans.at<double>(0) = baseRPT.at<double>(0, 3);
            baseTrans.at<double>(1) = baseRPT.at<double>(1, 3);
            baseTrans.at<double>(2) = baseRPT.at<double>(2, 3);
            /* baseTrans = temp; */

            baseRot.at<double>(0, 0) = baseRPT.at<double>(0, 0);
            baseRot.at<double>(0, 1) = baseRPT.at<double>(0, 1);
            baseRot.at<double>(0, 2) = baseRPT.at<double>(0, 2);
            baseRot.at<double>(1, 0) = baseRPT.at<double>(1, 0);
            baseRot.at<double>(1, 1) = baseRPT.at<double>(1, 1);
            baseRot.at<double>(1, 2) = baseRPT.at<double>(1, 2);
            baseRot.at<double>(2, 0) = baseRPT.at<double>(2, 0);
            baseRot.at<double>(2, 1) = baseRPT.at<double>(2, 1);
            baseRot.at<double>(2, 2) = baseRPT.at<double>(2, 2);

            cout << "Base Trans " << baseTrans << endl;
            cv::Mat baseRot_rodrigues;
            cv::Rodrigues(baseRot, baseRot_rodrigues);
            cout << "Base Rot " << baseRot_rodrigues << endl;
            /* baseRot = correctRot; */
            /* cout << "baseTrans " << baseTrans << endl << "t" << t << endl; */

            cout << "negativeZCount " << negativeZCount << endl;
            negativeZCount = 0;

        } // End of if statement
        //
        cout << "baseRPT" << baseRPT << endl;

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

    /* cv::cvtColor(src, src_gray, cv::COLOR_RGB2GRAY); */

    /* cv::setTrackbarPos("Max corners", source_window, maxCorners); */

    /* imshow(source_window, src); */

    FeatureExtractor feature_extractor;

    feature_extractor.readStringList("./src/imageList.xml", imageList);

    cout << "Image List: " << imageList.size();

    for (int i = 1; i < imageList.size(); i++) {

        src = cv::imread(imageList[i]);
        src_gray = cv::imread(imageList[i], cv::IMREAD_GRAYSCALE);
        prev_src_gray = cv::imread(imageList[i - 1], cv::IMREAD_GRAYSCALE);

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

        int key = cv::waitKey(5000);
        feature_extractor.writeCloudToFile("PointCLoud.ply",
                                           feature_extractor.PointCloudMatrix,
                                           feature_extractor.PointColorsMatrix);

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
