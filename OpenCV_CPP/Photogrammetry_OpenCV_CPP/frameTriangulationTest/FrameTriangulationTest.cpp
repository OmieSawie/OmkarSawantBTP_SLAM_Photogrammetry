#include <unistd.h>

#include <cmath>
// #include <ctime>
#include <iostream>
#include <opencv2/calib3d.hpp>
#include <opencv2/core/mat.hpp>
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
#include <string>
#include <vector>

using namespace std;

int nfeatures = 100000;
// float scaleFactor = 1.1f;
int nlevels = 16;
int edgeThreshold = 31;
int firstLevel = 0;
int WTA_K = 2;
int scoreType = cv::ORB::HARRIS_SCORE;
int patchSize = 31;
int fastThreshold = 20;
bool blurForDescriptor = true;
int negativeZCount = 0;

// Class of Storing Point Cloud data
class PointCLoud {
public:
  cv::Mat pointCloudMatrix = cv::Mat(0, 0, CV_64F);
  cv::Mat pointColorsMatrix = cv::Mat(0, 0, CV_64F);

  void writeCloudToFile(string filename, cv::Mat PointCloudMatrix) {
    cv::viz::writeCloud(filename, PointCloudMatrix);
  }
};

class Frame {
public:
  cv::Mat cameraMatrix = (cv::Mat_<double>(3, 3) << 3658.26, 0., 2311.5, 0.,
                          3658.26, 1734., 0., 0., 1.);

  // Distortion coefficients
  cv::Mat distortionCoefficients =
      (cv::Mat_<double>(5, 1) << 0.21044, -1.0387, 0., 0., 1.6132);
  cv::Mat src;
  cv::Mat src_gray;
  cv::Mat descriptors_cpu;
  std::vector<cv::KeyPoint> keypoints_cpu;
  // double scaleFactor = 2.;

  vector<cv::Point2f> points1;

  cv::Mat relPose = (cv::Mat_<double>(4, 4, CV_64F) << 1, 0, 0, 0, 0, 1, 0, 0,
                     0, 0, 1, 0, 0, 0, 0, 1);

  cv::Mat basePose = (cv::Mat_<double>(4, 4, CV_64F) << 1, 0, 0, 0, 0, 1, 0, 0,
                      0, 0, 1, 0, 0, 0, 0, 1);
  Frame(cv::Mat basePose) { this->basePose = basePose; }
};

class Compute3D {
private:
  cv::Mat cameraMatrix = (cv::Mat_<double>(3, 3) << 3658.26, 0., 2311.5, 0.,
                          3658.26, 1734., 0., 0., 1.);

public:
  cv::Mat extractTransFromPose(cv::Mat pose) {
    cv::Mat trans = (cv::Mat_<double>(3, 1));
    trans.at<double>(0, 0) = pose.at<double>(0, 3);
    trans.at<double>(1, 0) = pose.at<double>(1, 3);
    trans.at<double>(2, 0) = pose.at<double>(2, 3);
    return trans;
  }
  cv::Mat extractRotFromPose(cv::Mat pose) {
    cv::Mat rot = (cv::Mat_<double>(3, 1));
    rot.at<double>(0, 0) = pose.at<double>(0, 0);
    rot.at<double>(0, 1) = pose.at<double>(0, 1);
    rot.at<double>(0, 2) = pose.at<double>(0, 2);
    rot.at<double>(1, 0) = pose.at<double>(1, 0);
    rot.at<double>(1, 1) = pose.at<double>(1, 1);
    rot.at<double>(1, 2) = pose.at<double>(1, 2);
    rot.at<double>(2, 0) = pose.at<double>(2, 0);
    rot.at<double>(2, 1) = pose.at<double>(2, 1);
    rot.at<double>(2, 2) = pose.at<double>(2, 2);
    return rot;
  }

  // Adjust the scale factor to match the point clouds by triangulation of 3
  // frames
  void triangulateFramesForScale(Frame &prevFrame, Frame &frame,
                                 cv::Mat relPose, cv::Mat basePose,
                                 PointCLoud &pointCloud,
                                 vector<cv::Point3d> &worldCoordinateVector,
                                 vector<cv::DMatch> good_matches0,
                                 vector<cv::DMatch> good_matches1) {
    int maxMatches = max(good_matches0.size(), good_matches1.size());

    vector<pair<int, int>> tracker;
    // vector<cv::DMatch> good_matches3;
    for (int i = 0; i < good_matches0.size(); i++) {
      // cout << goodMatchesDataVector[0][i].trainIdx << " ";
      for (int j = 0; j < good_matches1.size(); j++) {
        // cout << goodMatchesDataVector[1][i].queryIdx << " ";
        if (good_matches0[i].trainIdx == good_matches1[j].queryIdx) {
          // cout << good_matches0[i].trainIdx << " ";
          // cv::DMatch match;
          // match.trainIdx = good_matches1[j].trainIdx;
          // match.queryIdx = good_matches0[i].queryIdx;
          // good_matches3.push_back(match);
          pair<int, int> track;
          track.first = i;
          track.second = j;
          tracker.push_back(track);
        }
      }
    }
    cout << "Tracker size " << tracker.size() << endl;

    vector<cv::Point3d> prevWorldCoordinateVector = worldCoordinateVector;
    cv::Point3d diff, prevDiff;

    double scaleFactor = 1;
    for (int i = 0; i < 100; i++) {
      diff.x = 0;
      diff.y = 0;
      diff.z = 0;
      if (i == 0) {
        scaleFactor = 0.8;
      }
      worldCoordinateVector.clear();
      computeCoordinates(prevFrame, frame, relPose, basePose, pointCloud,
                         worldCoordinateVector, scaleFactor);

      for (int i = 0; i < tracker.size(); i++) {
        diff += prevWorldCoordinateVector[tracker[i].first] -
                worldCoordinateVector[tracker[i].second];
        // cout << prevWorldCoordinateVector[tracker[i].first] -
        //             worldCoordinateVector[tracker[i].second]
        //      << " " << endl;
        // cout << prevWorldCoordinateVector[tracker[i].first] << " " <<
        // endl;
        // cout << endl;
      }
      cout << "Diff " << cv::norm(diff) << " " << cv::norm(prevDiff) << endl;
      cout << "scaleFactor  " << scaleFactor << endl;
      if (i == 0) {
        scaleFactor = 1.2;
      }
      if (i >= 1) {
        if (norm(diff) < norm(prevDiff)) {
          scaleFactor += 0.01;
        } else {
          scaleFactor -= 0.01;
        }
      }
      prevDiff = diff;
    }
  }

  // Find the relative pose in between the two input frames
  void findPose(Frame &prevFrame, Frame &frame, double scaleFactor) {
    // Estimating the Essential Matrix
    cv::Mat essentialMatrix;
    essentialMatrix =
        cv::findEssentialMat(prevFrame.points1, frame.points1, cameraMatrix,
                             cv::RANSAC, 0.99, 1.0, 10000);
    /* cout << essentialMatrix << endl; */
    cv::Mat R1, R2, t, R1R, R2R;
    cv::decomposeEssentialMat(essentialMatrix, R1, R2, t);
    /* cout << "R1" << R1 << endl << "R2" << R2 << endl; */
    if (t.at<double>(0) < 0.) {
      t = -t;
    }
    cv::Rodrigues(R1, R1R);
    cv::Rodrigues(R2, R2R);

    cout << "t" << t << endl << "R1R" << R1R << endl << "R2R" << R2R << endl;
    cv::Mat correctRot;
    if (abs(R1R.at<double>(0)) < abs(R2R.at<double>(0))) {
      correctRot = R1;
      cout << "R1 is Correct" << endl;
    } else {
      correctRot = R2;
      cout << "R2 is Correct" << endl;
    }

    t = t * scaleFactor;
    frame.relPose = composeHomPose(correctRot, t);
  }

  // Compute the homogenised camera matrix
  cv::Mat computeM() {
    cv::Mat temp = (cv::Mat_<double>(3, 1) << 0., 0., 0.);
    cv::Mat hom_cameraMatrix;
    cv::hconcat(cameraMatrix, temp, hom_cameraMatrix);
    /* cout << "Camera Matrix Homogenised: " << hom_cameraMatrix << endl; */
    return hom_cameraMatrix;
  }

  // Compute the euclidian transform from input rotation and translation
  // matrices
  cv::Mat composeHomPose(cv::Mat rot, cv::Mat trans) {
    cv::Mat pose, hom_pose;
    cv::hconcat(rot, trans, pose);
    cv::Mat temp = (cv::Mat_<double>(1, 4) << 0., 0., 0., 1);
    cv::vconcat(pose, temp, hom_pose);
    // cout << "Hom Pose" << hom_pose << endl;
    return hom_pose;
  }

  // Compute the product of homogenised camera matrix and the euclidian
  // transform
  cv::Mat comptueP(cv::Mat pose) {
    cv::Mat hom_cameraMatrix = computeM();
    /* baseRPT = baseRPT * hom_rpt; */
    /* cout << "Base RPT" << baseRPT << endl; */
    /* cout << "RPT homogenised: " << hom_rpt << endl; */
    /* cout << "cam*rpt: " << hom_cameraMatrix * hom_rpt << endl; */
    return hom_cameraMatrix * pose;
  }

  // Used by computeCoordinates to find the local coordinates
  cv::Mat getLocalCoordinates(cv::Mat src, cv::Point2f l, cv::Point2f r,
                              cv::Mat relPose, PointCLoud &pointCloud,
                              double scaleFactor) {
    /* cout << "getting local coordinates " << l.x << " " << l.y << endl; */

    cv::Mat M = computeM();
    cv::Mat P = comptueP(relPose);
    cv::Mat colors(1, 1, CV_8UC3, cv::Scalar(src.at<cv::Vec3b>(l.y, l.x)));
    pointCloud.pointColorsMatrix.push_back(colors);

    double ur = r.x, ul = l.x, vr = r.y, vl = l.y;

    double m11, m12, m13, m14, m21, m22, m23, m24, m31, m32, m33, m34, m41, m42,
        m43, m44;
    double p11, p12, p13, p14, p21, p22, p23, p24, p31, p32, p33, p34, p41, p42,
        p43, p44;

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
    p14 = P.at<double>(0, 3) * scaleFactor;
    p21 = P.at<double>(1, 0);
    p22 = P.at<double>(1, 1);
    p23 = P.at<double>(1, 2);
    p24 = P.at<double>(1, 3) * scaleFactor;
    p31 = P.at<double>(2, 0);
    p32 = P.at<double>(2, 1);
    p33 = P.at<double>(2, 2);
    p34 = P.at<double>(2, 3) * scaleFactor;

    /* cout << "ur= " << ur << endl; */
    /* cout << "m33= " << m33 << endl; */
    /* cout << "m13= " << m13 << endl; */
    /* cout << "M: " << M << endl; */

    cv::Mat finl =
        (cv::Mat_<double>(4, 3) << ur * m31 - m11, ur * m32 - m12,
         ur * m33 - m13, vr * m31 - m21, vr * m32 - m22, vr * m33 - m23,
         ul * p31 - p11, ul * p32 - p12, ul * p33 - p13, vl * p31 - p21,
         vl * p32 - p22, vl * p33 - p23);
    cv::Mat finr =
        (cv::Mat_<double>(4, 1) << m14 - m34, m24 - m34, p14 - p34, p24 - p34);
    /* cout << "finl and finr: " << finl << endl << finr << endl; */

    cv::Mat mulTransposedMat, mulTransposedMatInverted, transposedMat;
    cv::mulTransposed(finl, mulTransposedMat, true);
    /* cout << "mulTransposed: " << mulTransposedMat << endl; */
    cv::invert(mulTransposedMat, mulTransposedMatInverted);
    cv::transpose(finl, transposedMat);
    /* cout << "mulTransposedMatInverted: " << mulTransposedMatInverted */
    /* << endl; */
    cv::Mat rightFrameCoordiantes =
        mulTransposedMatInverted * transposedMat * finr;
    cv::Mat temp = (cv::Mat_<double>(1, 1) << 1);
    cv::Mat rightFrameCoordiantesHom;
    cv::vconcat(rightFrameCoordiantes, temp, rightFrameCoordiantesHom);
    /* cout << "rightFrameCoordiantes" << rightFrameCoordiantes << endl; */
    cv::Mat leftFrameCoordinatesHom = relPose * rightFrameCoordiantesHom;
    /* cout << "leftFrameCoordinatesHom" << leftFrameCoordinatesHom << endl;
     */

    cv::Mat leftFrameCoordinates =
        cv::Mat(3, 1, CV_64F, leftFrameCoordinatesHom.data);
    /* cout << "leftFrameCoordinates" << leftFrameCoordinates << endl; */

    return leftFrameCoordinatesHom;
    /* return rightFrameCoordiantes; */
  }

  // Used by computeCoordinates to calculate world coordinates from local
  // coordinates
  cv::Point3d getWorldCoordinates(cv::Mat basePose, cv::Mat localCoordinates,
                                  PointCLoud &pointCloud) {
    cv::Mat worldCoordinates = cv::Mat_<double>(3, 3, CV_64F);

    /* cout << "baseRot" << baseRot << endl; */
    /* worldCoordinates = baseRot * worldCoordinates; */
    /* worldCoordinates = localCoordinates + baseTrans; */
    /* cout << "worldCoordinates" << worldCoordinates << endl; */
    worldCoordinates = basePose * localCoordinates;
    cv::Point3d pointCoordinate;
    pointCoordinate.x = worldCoordinates.at<double>(0);
    pointCoordinate.y = worldCoordinates.at<double>(1);
    pointCoordinate.z = worldCoordinates.at<double>(2);

    // pointCloud.pointCloudMatrix.push_back(pointCoordinate);
    return pointCoordinate;
  }

  // Plce the camera reference in the point cloud
  void placeCameraReferencePointCloud(cv::Mat basePose,
                                      PointCLoud &pointCloud) {
    cv::Point3d cameraPointCLoud;
    cv::Mat arrowTip = (cv::Mat_<double>(4, 1) << 0., 0., 0., 1);
    arrowTip = basePose * arrowTip;
    cv::Mat arrowBase = (cv::Mat_<double>(4, 1) << 0., 0., 1., 1);
    arrowBase = basePose * arrowBase;

    for (int i = -2; i < 2; i++) {
      for (int j = -2; j < 2; j++) {
        for (int k = -2; k < 2; k++) {
          cameraPointCLoud.x = arrowTip.at<double>(0) + i * 0.1;
          cameraPointCLoud.y = arrowTip.at<double>(1) + j * 0.1;
          cameraPointCLoud.z = arrowTip.at<double>(2) + k * 0.1;
          pointCloud.pointCloudMatrix.push_back(cameraPointCLoud);
          cv::Mat colors1(1, 1, CV_8UC3, cv::Scalar(100, 100, 100));
          pointCloud.pointColorsMatrix.push_back(colors1);

          cameraPointCLoud.x = arrowBase.at<double>(0) + i * 0.1;
          cameraPointCLoud.y = arrowBase.at<double>(1) + j * 0.1;
          cameraPointCLoud.z = arrowBase.at<double>(2) + k * 0.1;
          pointCloud.pointCloudMatrix.push_back(cameraPointCLoud);
          cv::Mat colors2(1, 1, CV_8UC3, cv::Scalar(200, 200, 200));
          pointCloud.pointColorsMatrix.push_back(colors2);
        }
      }
    }
  }

  cv::Mat computeCoordinates(Frame prevFrame, Frame frame, cv::Mat relPose,
                             cv::Mat &basePose, PointCLoud &pointCloud,
                             vector<cv::Point3d> &worldCoordinateVector,
                             double scaleFactor) {
    int point_count = frame.points1.size();
    cv::Mat correctRot = extractRotFromPose(relPose);
    cv::Mat P = comptueP(relPose);
    cv::Mat M = computeM();
    for (int i = 0; i < point_count; i++) {
      cv::Mat localCoordinates = getLocalCoordinates(
          prevFrame.src, prevFrame.points1[i], frame.points1[i], relPose,
          pointCloud, scaleFactor);
      cv::Point3d worldCoordinate =
          getWorldCoordinates(basePose, localCoordinates, pointCloud);
      worldCoordinateVector.push_back(worldCoordinate);
    }
    placeCameraReferencePointCloud(basePose, pointCloud);

    basePose.at<double>(0, 3) = basePose.at<double>(0, 3);
    basePose.at<double>(1, 3) = basePose.at<double>(1, 3);
    basePose.at<double>(2, 3) = basePose.at<double>(2, 3);
    basePose.at<double>(3, 3) = basePose.at<double>(3, 3);

    basePose = basePose * relPose;
    /* cv::Mat temp = (cv::Mat_<double>(1, 3) << 0., 0., 0.); */

    /* baseTrans.at<double>(0) = baseRPT.at<double>(0, 3); */
    /* baseTrans.at<double>(1) = baseRPT.at<double>(1, 3); */
    /* baseTrans.at<double>(2) = baseRPT.at<double>(2, 3); */
    /* /1* baseTrans = temp; *1/ */

    /* baseRot.at<double>(0, 0) = baseRPT.at<double>(0, 0); */
    /* baseRot.at<double>(0, 1) = baseRPT.at<double>(0, 1); */
    /* baseRot.at<double>(0, 2) = baseRPT.at<double>(0, 2); */
    /* baseRot.at<double>(1, 0) = baseRPT.at<double>(1, 0); */
    /* baseRot.at<double>(1, 1) = baseRPT.at<double>(1, 1); */
    /* baseRot.at<double>(1, 2) = baseRPT.at<double>(1, 2); */
    /* baseRot.at<double>(2, 0) = baseRPT.at<double>(2, 0); */
    /* baseRot.at<double>(2, 1) = baseRPT.at<double>(2, 1); */
    /* baseRot.at<double>(2, 2) = baseRPT.at<double>(2, 2); */

    cout << "Pose " << basePose << endl;
    cout << "Base Rot " << correctRot << endl;

    negativeZCount = 0;

    //
    /* cout << "baseRPT" << baseRPT << endl; */

    /* cout << "Size 1" << prev_src_gray.size() << endl; */
    return basePose;
  }
};

class FeatureExtractorAndMatcher {
public:
  // Make the SIFT Object with the appropriate parameters
  cv::Ptr<cv::SIFT> extractor =
      cv::SIFT::create(100000, 3, 0.09, 31, 1.2f, CV_8U, false);

  // Camera Matrix as obtained from the camera calibration
  cv::Mat cameraMatrix = (cv::Mat_<double>(3, 3) << 3658.26, 0., 2311.5, 0.,
                          3658.26, 1734., 0., 0., 1.);
  /* cv::Mat cameraMatrix = (cv::Mat_<double>(3, 3) << 5000., 0., 2312.1, 0.,
   */

  /*                         5000., 1734.1, 0., 0., 1.); */

  // Point Cloud Output data and Color matrices
  cv::Mat PointCloudMatrix = cv::Mat(0., 0., CV_64F),
          PointColorsMatrix = cv::Mat(0., 0., CV_8UC3);

  // Rotation and translation of the camera frame with respect to the world
  // coordinate frame ie. initial point
  /* cv::Mat baseRot = */
  /*     (cv::Mat_<double>(3, 3, CV_64F) << 1., 0., 0., 0., 1., 0., 0.,
   * 0., 1.); */
  /* cv::Mat baseTrans = (cv::Mat_<double>(3, 1, CV_64F) << 0., 0., 0.); */

  // Frame &prevFrame;
  // Frame &frame;
  //
  // FeatureExtractorAndMatcher(Frame &prevFrame, Frame &frame)
  //     : prevFrame(prevFrame), frame(frame) {
  //   // this->prevFrame = prevFrame;
  //   // this->frame = frame;
  // }

  // cv::Mat prev_descriptors_cpu, descriptors_cpu;
  // std::vector<cv::KeyPoint> prev_keypoints_cpu, keypoints_cpu;

  // Euclidian Transform of camera frame with respect to the world
  // coordinate frame ie. the initial point
  cv::Mat baseRPT = (cv::Mat_<double>(4, 4, CV_64F) << 1, 0, 0, 0, 0, 1, 0, 0,
                     0, 0, 1, 0, 0, 0, 0, 1);

  void extractFeaturesSIFT(Frame &frame) {
    int img_width = frame.src_gray.size().width,
        img_height = frame.src_gray.size().height;
    cout << " Size of image: " << img_width << " x " << img_height << endl;
    double qualityLevel = 0.01;
    double minDistance = 0;
    int blockSize = 10, gradientSize = 3;
    bool useHarrisDetector = false;
    double k = 0.04;
    double harrisK = 0.04;

    // Detect and Compute features
    // prev_src_gray is	#1 query image
    // src_gray is the  #2 train image
    extractor->detectAndCompute(frame.src_gray, cv::noArray(),
                                frame.keypoints_cpu, frame.descriptors_cpu);

    cout << "Keypoints Size :" << frame.keypoints_cpu.size() << endl;
    std::cout << "Descriptors Size " << frame.descriptors_cpu.size() << endl;
    // this->frame = frame;
    // this->prevFrame = prevFrame;
  }

  std::vector<cv::DMatch>
  matchFeaturesBrutForce(Frame &prevFrame, Frame &frame,
                         vector<cv::DMatch> &good_matches) {
    // cv::Mat prev_src_gray = this->prevFrame.src_gray;
    // cv::Mat src_gray = this->frame.src_gray;
    // cout << "Keypoints size in matching features "
    //      << prevFrame.descriptors_cpu.size() << endl;
    //
    // Match the features
    cv::Ptr<cv::DescriptorMatcher> matcher =
        cv::DescriptorMatcher::create("BruteForce");

    vector<vector<cv::DMatch>> matches;
    matcher->knnMatch(prevFrame.descriptors_cpu, frame.descriptors_cpu, matches,
                      2);

    // Filter the good matches
    good_matches.clear();
    for (int k = 0; k < (int)matches.size(); k++) {
      if ((matches[k][0].distance < 0.8 * (matches[k][1].distance))) {
        good_matches.push_back(matches[k][0]);
      }
    }
    cout << endl;
    cout << "MatchesSize: " << good_matches.size() << endl;

    /* t_BBtime = getTickCount(); */
    /* t_pt = (t_BBtime - t_AAtime) / getTickFrequency(); */
    /* t_fpt = 1 / t_pt; */
    /* printf("%.4lf sec/ %.4lf fps\n", t_pt, t_fpt); */

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

    // Estimating the Fundamental Matrix
    int point_count = good_matches.size();
    vector<cv::Point2f> points1(point_count);
    vector<cv::Point2f> points2(point_count);

    // Clear the previously made points1 Matrix in prevFrame
    prevFrame.points1.clear();
    frame.points1.clear();
    // initialize the points here ...
    for (int i = 0; i < point_count; i++) {
      prevFrame.points1.push_back(
          prevFrame.keypoints_cpu[good_matches[i].queryIdx].pt);
      frame.points1.push_back(frame.keypoints_cpu[good_matches[i].trainIdx].pt);
    }

    vector<cv::Point2f> points1_masked, points2_masked;

    cv::Mat fund_mask;
    // cout << "Here " << frame.points1.size() << endl;
    cv::Mat fundamentalMatrix = cv::findFundamentalMat(
        prevFrame.points1, frame.points1, cv::FM_RANSAC, 3., 0.90, fund_mask);
    // cout << "Here" << endl;
    for (int i = 0; i < point_count; i++) {
      if (fund_mask.at<int>(i) == 1) {
        points1_masked.push_back(prevFrame.points1[i]);
        points2_masked.push_back(frame.points1[i]);
      }
    }
    cout << "masked points count: " << points1_masked.size() << endl;
    /*  cout << fundamentalMatrix << endl; */
    /* cout << fund_mask; */
    /* cv::correctMatches(fundamentalMatrix, points1, points2, points1,
     */
    /*                    points2); */

    // Compute epilines
    vector<cv::Vec3f> epilines1, epilines2;
    cv::computeCorrespondEpilines(points1_masked, 1, fundamentalMatrix,
                                  epilines2);
    cv::computeCorrespondEpilines(points2_masked, 2, fundamentalMatrix,
                                  epilines1);

    if (false) {
      for (vector<cv::Vec3f>::const_iterator it = epilines1.begin();
           it != epilines1.end(); ++it) {
        // draw the line between first and last column
        cv::line(prevFrame.src_gray, cv::Point(0, -(*it)[2] / (*it)[1]),
                 cv::Point(4624., -((*it)[2] + (*it)[0] * 4624.) / ((*it)[1])),
                 cv::Scalar(255, 255, 255), cv::LINE_4);
      }
      for (vector<cv::Vec3f>::const_iterator it = epilines2.begin();
           it != epilines2.end(); ++it) {
        // draw the line between first and last column
        cv::line(frame.src_gray, cv::Point(0, -(*it)[2] / (*it)[1]),
                 cv::Point(4624., -((*it)[2] + (*it)[0] * 4624.) / ((*it)[1])),
                 cv::Scalar(255, 255, 255), cv::LINE_4);
      }
      // Draw trhe masked keypoints
      for (int i = 0; i < points1_masked.size(); i++) {
        cv::circle(prevFrame.src_gray, points1_masked[i], 10,
                   cv::Scalar(255, 0, 255), 10, cv::LINE_8, 0);
        cv::circle(frame.src_gray, points2_masked[i], 10,
                   cv::Scalar(255, 0, 255), 10, cv::LINE_8, 0);
      }
    }
    /* cv::imshow("Right Image Epilines (FM_7POINT2)",
     * src_gray); */
    /* cv::imshow("Right Image Epilines (FM_7POINT1)",
     * prev_src_gray);
     */

    // Draw the Matches
    cv::Mat img_matches;
    drawMatches(prevFrame.src_gray, prevFrame.keypoints_cpu, frame.src_gray,
                frame.keypoints_cpu, good_matches, img_matches);

    cv::namedWindow("matches", 0);
    imshow("matches", img_matches);
    cv::imwrite("Matches.jpg", img_matches);

    /* cv::resize(src_gray, src_gray, cv::Size(), 0.25, 0.25); */
    /* cv::resize(prev_src_gray, prev_src_gray, cv::Size(), 0.25, 0.25);
     */
    /* cv::imshow("src_gray", src_gray); */
    /* cv::imshow("prev_src_gray", prev_src_gray); */
    return good_matches;
  }
};
// Take input image files from xml file
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

int main() {
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

  PointCLoud pointCloud;
  cv::Mat basePose = (cv::Mat_<double>(4, 4, CV_64F) << 1, 0, 0, 0, 0, 1, 0, 0,
                      0, 0, 1, 0, 0, 0, 0, 1);

  vector<string> imageList;
  readStringList("./src/imageList.xml", imageList);

  cout << "Image List: " << imageList.size() << endl;

  vector<Frame> framesData;
  vector<vector<cv::DMatch>> goodMatchesDataVector;
  // vector<cv::Point3d> pointsDataVector;
  vector<cv::Point3d> worldCoordinateVector;

  for (int i = 0; i < imageList.size(); i++) {
    cv::Mat distorted_src, distorted_src_gray, src, src_gray;

    Frame frame(basePose);

    distorted_src = cv::imread(imageList[i]);
    cv::undistort(distorted_src, src, frame.cameraMatrix,
                  frame.distortionCoefficients, cv::noArray());
    distorted_src_gray = cv::imread(imageList[i], cv::IMREAD_GRAYSCALE);
    cv::undistort(distorted_src_gray, src_gray, frame.cameraMatrix,
                  frame.distortionCoefficients, cv::noArray());

    frame.src = src;
    frame.src_gray = src_gray;
    FeatureExtractorAndMatcher featureExtractorAndMatcher;

    featureExtractorAndMatcher.extractFeaturesSIFT(frame);
    if (i >= 1) {
      Frame prevFrame = framesData[i - 1];

      vector<cv::DMatch> goodMatchesData;
      featureExtractorAndMatcher.matchFeaturesBrutForce(prevFrame, frame,
                                                        goodMatchesData);
      goodMatchesDataVector.push_back(goodMatchesData);

      Compute3D compute3D;
      // cout << "FrameSize: " <<
      // featureExtractorAndMatcher.frame.src_gray.size()
      //      << std::endl;

      cout << "Main frame object size " << frame.descriptors_cpu.size() << endl;
      cout << "Main Frame object data " << frame.keypoints_cpu.size() << endl;

      // change in frame is not reflected in the main frame object after
      // feature extraction
      compute3D.findPose(prevFrame, frame, 1.);

      cout << "relative pose " << frame.relPose << endl;

      if (i >= 2) {
        // cout << goodMatchesDataVector[i - 2].size() << " "
        //      << goodMatchesDataVector[i - 1].size() << endl;
        compute3D.triangulateFramesForScale(
            prevFrame, frame, frame.relPose, prevFrame.basePose, pointCloud,
            worldCoordinateVector, goodMatchesDataVector[i - 2],
            goodMatchesDataVector[i - 1]);
      }

      worldCoordinateVector.clear();
      basePose = compute3D.computeCoordinates(prevFrame, frame, frame.relPose,
                                              prevFrame.basePose, pointCloud,
                                              worldCoordinateVector, 1.);
      cout << "DataPointSaves size: " << worldCoordinateVector.size() << endl;

      pointCloud.pointCloudMatrix.push_back(worldCoordinateVector);

      // pointCloud.writeCloudToFile("PointCLoud.ply",
      // pointCloud.pointCloudMatrix,
      //                             pointCloud.pointColorsMatrix);
      pointCloud.writeCloudToFile("PointCLoud.ply",
                                  pointCloud.pointCloudMatrix);

      framesData[i - 1] = prevFrame;
    }
    framesData.push_back(frame);

    int key = cv::waitKey(5000);
    if (key == 'q') {
      cout << "q key is pressed by the user. Stopping the "
              "video"
           << endl;
      break;
    }
  }
  cout << "FraneData Vector data: " << framesData[1].keypoints_cpu.size();
  cout << "GoodMatchesData Vector data: " << goodMatchesDataVector[1].size()
       << endl;

  // vector<cv::DMatch> good_matches;
  // for (int i = 0; i < goodMatchesDataVector[0].size(); i++) {
  //   // cout << goodMatchesDataVector[0][i].trainIdx << " ";
  //   for (int j = 0; j < goodMatchesDataVector[1].size(); j++) {
  //     // cout << goodMatchesDataVector[1][i].queryIdx << " ";
  //     if (goodMatchesDataVector[0][i].trainIdx ==
  //         goodMatchesDataVector[1][j].queryIdx) {
  //       cout << goodMatchesDataVector[0][i].trainIdx << " ";
  //       cv::DMatch match;
  //       // match.trainIdx =
  //     }
  //   }
  // }
  cout << endl;
  // for (int i = 0; i < goodMatchesDataVector[1].size(); i++) {
  //   cout << goodMatchesDataVector[1][i].queryIdx << " ";
  // }

  /* } */
  // Release the video capture object
  /* vid_capture.release(); */
  cv::destroyAllWindows();
  return 0;
}
