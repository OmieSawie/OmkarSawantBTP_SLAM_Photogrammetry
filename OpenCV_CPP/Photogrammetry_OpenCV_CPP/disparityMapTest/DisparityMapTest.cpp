#include "opencv2/cudastereo.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include <iomanip>
#include <iostream>
#include <opencv2/calib3d.hpp>
#include <opencv2/core/base.hpp>
#include <opencv2/core/hal/interface.h>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/matx.hpp>
#include <opencv2/core/persistence.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/viz/vizcore.hpp>

#include <sstream>
#include <stdexcept>
#include <string>

using namespace cv;
using namespace std;

void removeInfPoints(const Mat points, Mat left_src, Mat &maskPoints,
                     Mat &maskColors) {
    float inf = std::numeric_limits<float>::infinity();

    /* removeInfPoints = points.clone(); */
    // Create a mask to remove inf points.
    for (int i = 0; i < points.rows; i++) {
        for (int j = 0; j < points.cols; j++) {

            if (points.at<Point3f>(i, j).x == inf ||
                points.at<Point3f>(i, j).y == inf ||
                points.at<Point3f>(i, j).z == inf) {

                /* points.at<Point3f>(i, j).x = 0.f; */
                /* points.at<Point3f>(i, j).y = 0.f; */
                /* points.at<Point3f>(i, j).z = 0.f; */
                /* removeInfPoints.at<Point3f>(i, j); */

                /* cout << removeInfPoints.at<Point3f>(i, j) << " "; */

                /* cout << "Hello"; */
            } else {
                /* removeInfPoints.at<Point3f>(i, j).x = */
                /*     points.at<Point3f>(i, j).x; */
                /* removeInfPoints.at<Point3f>(i, j).y = */
                /*     points.at<Point3f>(i, j).y; */
                /* removeInfPoints.at<Point3f>(i, j).z = */
                /*     points.at<Point3f>(i, j).z; */
                /* removeInfPoints.push_back({points.at<Point3f>(i, j).x, */
                /*                            points.at<Point3f>(i, j).y, */
                /*                            points.at<Point3f>(i, j).z}); */

                /* cout << points.at<Point3f>(i, j); */

                Mat colors(1, 1, CV_8UC3, Scalar(left_src.at<Vec3b>(i, j)));

                maskColors.push_back(colors);
                maskPoints.push_back(points.at<Point3f>(i, j));

                /* mask.push_back({points.at<Point3f>(i, j).x, */
                /*                 points.at<Point3f>(i, j).y, */
                /*                 points.at<Point3f>(i, j).z}); */
            }
        }
    }

    // Copy the points to the output image.
    /* removeInfPoints = points.clone(); */
    /* removeInfPoints.copyTo(removeInfPoints, mask); */
    /* cout << removeInfPoints; */
}

int main(int argc, char **argv) {

    bool running;
    Mat left_src, right_src;
    Mat left, right;
    cuda::GpuMat d_left, d_right;

    int ndisp = 16;    // 256;
    int blockSize = 3; // 51;

    Ptr<cuda::StereoBM> bm;

    bm = cv::cuda::createStereoBM(ndisp, blockSize);

    /* VideoCapture vid_capture( */
    /*     "/home/omie_sawie/Code_Code/OmkarSawantBTP_SLAM_Photogrammetry/"
     */
    /*     "OpenCV_CPP/Photogrammetry_OpenCV_CPP/videoFeatureMatching/" */
    /*     "/resources/carOnLonelyRoads.mp4"); */

    // Print error message if the stream is invalid
    /* if (!vid_capture.isOpened()) { */
    /*     cout << "Error opening video stream or file" << endl; */
    /* } */
    left_src = cv::imread("../imageL0.png");
    right_src = cv::imread("../imageR0.png");

    /* while (vid_capture.isOpened()) { */
    // Initialise frame matrix
    // Initialize a boolean to check if frames are there or not
    /* vid_capture.read(left_src); */
    /* vid_capture.read(right_src); */
    /* int down_width = 1900; */
    /* int down_height = 900; */

    cvtColor(left_src, left, COLOR_BGR2GRAY);
    cvtColor(right_src, right, COLOR_BGR2GRAY);

    /* cv::resize(left, left, cv::Size(), 0.1, 0.1); */
    /* cv::resize(right, right, cv::Size(), 0.1, 0.1); */

    // resize down

    d_left.upload(left);
    d_right.upload(right);

    imshow("left", left);
    imshow("right", right);
    /* cout << "src:" << left.size() << left_src.size() << endl; */

    // Prepare disparity map of specified type
    Mat disp(left.size(), CV_32F);
    cuda::GpuMat d_disp(left.size(), CV_32F);

    bm->compute(d_left, d_right, d_disp);

    // Show results
    d_disp.download(disp);

    Mat disparity;

    disp.convertTo(disparity, CV_32F, 1.0f / 16.f);

    /* // Scaling down the disparity values and normalizing them */
    /* disparity = (disparity * 16.0f); */

    // cout << disp << endl;
    normalize(disparity, disparity, 0., 255., NORM_MINMAX, CV_32F);

    Mat Q = (Mat_<float>(4, 4) << 1., 0., 0., -3.1932437133789062e+02, 0., 1.,
             0., -2.3945363616943359e+02, 0., 0., 0., 4.3964859406340838e+02,
             0., 0., 2.9912905731253359e-01, 0.);

    Mat Img3D;
    /* cout << disparity.size(); */

    cv::reprojectImageTo3D(disparity, Img3D, Q, false, CV_32F);

    /* cout << Img3D << endl; */

    Mat maskPoints = Mat(0, 0, CV_32F);
    Mat maskColors = Mat(0, 0, CV_8UC3);

    removeInfPoints(Img3D, right_src, maskPoints, maskColors);

    cout << left_src.type() << endl;

    cv::viz::writeCloud("pointCloud.ply", maskPoints, maskColors);
    /* cout << "3D:" << Img3D.size() << endl; */
    /* cout << Img3D.size(); */
    imshow("disparity", Img3D);

    while (true) {
        int key = waitKey(10000);
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
    destroyAllWindows();
    return 0;
}
