#include "opencv2/cudastereo.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include <iomanip>
#include <iostream>
#include <opencv2/calib3d.hpp>
#include <opencv2/core/base.hpp>
#include <opencv2/core/hal/interface.h>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/persistence.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/viz/vizcore.hpp>

#include <sstream>
#include <stdexcept>
#include <string>

using namespace cv;
using namespace std;

void removeInfPoints(const Mat &points, Mat &removeInfPoints, Mat &left) {
    float inf = std::numeric_limits<float>::infinity();

    removeInfPoints = points.clone();
    // Create a mask to remove inf points.
    Mat mask = Mat(points.rows, points.cols, CV_8UC1);
    for (int i = 0; i < mask.rows; i++) {
        for (int j = 0; j < mask.cols; j++) {
            if (points.at<Point3f>(i, j).x == inf ||
                points.at<Point3f>(i, j).y == inf ||
                points.at<Point3f>(i, j).z == inf) {
                mask.at<uchar>(i, j) = 0;

                removeInfPoints.at<Point3f>(i, j).x = 0.f;
                removeInfPoints.at<Point3f>(i, j).y = 0.f;
                removeInfPoints.at<Point3f>(i, j).z = 0.f;
                removeInfPoints.at<Point3f>(i, j);

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

                mask.at<uchar>(i, j) = 1;
            }
        }
    }

    // Copy the points to the output image.
    /* removeInfPoints = points.clone(); */
    /* removeInfPoints.copyTo(removeInfPoints, mask); */
    /* cout << removeInfPoints; */
}

/* int main(int argc, char **argv) { */

/*     bool running; */
/*     Mat left_src, right_src; */
/*     Mat left, right; */
/*     cuda::GpuMat d_left, d_right; */

/*     int ndisp = 16; */

/*     Ptr<cuda::StereoBM> bm; */

/*     bm = cuda::createStereoBM(ndisp, 3); */

/*     /1* VideoCapture vid_capture( *1/ */
/*     /1*     "/home/omie_sawie/Code_Code/OmkarSawantBTP_SLAM_Photogrammetry/"
 */
/*      *1/ */
/*     /1*     "OpenCV_CPP/Photogrammetry_OpenCV_CPP/videoFeatureMatching/" *1/
 */
/*     /1*     "/resources/carOnLonelyRoads.mp4"); *1/ */

/*     // Print error message if the stream is invalid */
/*     /1* if (!vid_capture.isOpened()) { *1/ */
/*     /1*     cout << "Error opening video stream or file" << endl; *1/ */
/*     /1* } *1/ */
/*     left_src = cv::imread("../imageL0.jpeg"); */
/*     right_src = cv::imread("../imageR0.jpeg"); */

/*     /1* while (vid_capture.isOpened()) { *1/ */
/*     // Initialise frame matrix */
/*     // Initialize a boolean to check if frames are there or not */
/*     /1* vid_capture.read(left_src); *1/ */
/*     /1* vid_capture.read(right_src); *1/ */
/*     /1* int down_width = 1900; *1/ */
/*     /1* int down_height = 900; *1/ */

/*     cvtColor(left_src, left, COLOR_BGR2GRAY); */
/*     cvtColor(right_src, right, COLOR_BGR2GRAY); */

/*     /1* cv::resize(left, left, cv::Size(), 0.5, 0.5); *1/ */
/*     /1* cv::resize(right, right, cv::Size(), 0.5, 0.5); *1/ */

/*     // resize down */

/*     d_left.upload(left); */
/*     d_right.upload(right); */

/*     imshow("left", left); */
/*     imshow("right", right); */

/*     // Prepare disparity map of specified type */
/*     Mat disp(left.size(), CV_32F); */
/*     cuda::GpuMat d_disp(left.size(), CV_32F); */

/*     bm->compute(d_left, d_right, d_disp); */

/*     // Show results */
/*     d_disp.download(disp); */

/*     Mat disparity; */

/*     disp.convertTo(disparity, CV_32F, 1.0f); */

/*     // Scaling down the disparity values and normalizing them */
/*     disparity = (disparity / 16.0f); */

/*     // cout << disp << endl; */
/*     /1* normalize(disparity, disparity, 0., 255., NORM_MINMAX, CV_32F); *1/
 */

/*     Mat Q = (Mat_<double>(4, 4) << 1., 0., 0., -3.1932437133789062e+02,
 * 0., 1., */
/*              0., -2.3945363616943359e+02, 0., 0., 0., 4.3964859406340838e+02,
 */
/*              0., 0., 2.9912905731253359e-01, 0.); */

/*     Mat Img3D, Img3D_removeINF; */

/*     cv::reprojectImageTo3D(disparity, Img3D, Q, false, CV_32F); */
/*     removeInfPoints(Img3D, Img3D_removeINF, left_src); */
/*     cv::viz::writeCloud("pointCloud.ply", Img3D_removeINF); */
/*     /1* cout << Img3D_removeINF; *1/ */
/*     cout << Img3D.size(); */
/*     imshow("disparity", Img3D_removeINF); */

/*     /1* Q = np.array(([ 1.0, 0.0, 0.0, -160.0 ], [ 0.0, 1.0, 0.0, -120.0 ],
 */
/*      *1/ */
/*     /1*               [ 0.0, 0.0, 0.0, 350.0 ], [ 0.0, 0.0, 1.0 / 90.0, 0.0
 */
/*      * ]), *1/ */
/*     /1*              dtype = np.float32) *1/ */

/*     /1* cv::stereoRectify(InputArray cameraMatrix1, InputArray */
/*      * distCoeffs1, */
/*      *1/ */
/*     /1*                   InputArray cameraMatrix2, InputArray */
/*      * distCoeffs2, */
/*      *1/ */
/*     /1*                   Size imageSize, InputArray R, InputArray T, *1/ */
/*     /1*                   OutputArray R1, OutputArray R2, OutputArray P1, */
/*      *1/ */
/*     /1*                   OutputArray P2, OutputArray Q); *1/ */

/*     while (true) { */
/*         int key = waitKey(10000); */
/*         if (key == 'q') { */
/*             cout << "q key is pressed by the user. Stopping the " */
/*                     "video" */
/*                  << endl; */
/*             break; */
/*         } */
/*     } */
/*     /1* } *1/ */
/*     // Release the video capture object */
/*     /1* vid_capture.release(); *1/ */
/*     destroyAllWindows(); */
/*     return 0; */
/* } */

#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/core/utility.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/ximgproc/disparity_filter.hpp"

#include <iostream>
#include <stdio.h>
#include <string.h>

// to compile: g++ filteredDisparityMap.cpp -lopencv_core -lopencv_videoio
// -lopencv_highgui -lopencv_imgcodecs -lopencv_imgproc -lopencv_calib3d
// -lopencv_features2d -lopencv_ximgproc -o veRun

using namespace cv;
using namespace std;
using namespace cv::ximgproc;

Rect computeROI(Size2i src_sz, Ptr<StereoMatcher> matcher_instance) {
    int min_disparity = matcher_instance->getMinDisparity();
    int num_disparities = matcher_instance->getNumDisparities();
    int block_size = matcher_instance->getBlockSize();

    int bs2 = block_size / 2;
    int minD = min_disparity, maxD = min_disparity + num_disparities - 1;

    int xmin = maxD + bs2;
    int xmax = src_sz.width + minD - bs2;
    int ymin = bs2;
    int ymax = src_sz.height - bs2;

    Rect r(xmin, ymin, xmax - xmin, ymax - ymin);
    return r;
}

int main(int, char **) {

    bool no_display;
    bool no_downscale;
    int max_disp, wsize;  // Stereo correspondence parameters
    double lambda, sigma; // Post-filter parameters
    double
        vis_mult; // Coefficient used for Disparity Map (DM) visualization scale
    Mat imgDisparity8U;
    double minVal, maxVal;
    String filter;
    String algo;         // Which OpenCV algorithm was used, BM or SGBM
    String dst_path;     // Optional path to save filtered DM result
    String dst_raw_path; // Optional to save DM without filter
    String
        dst_conf_path; // Optional path to save the trust map used for filtering

    char key = 0;

    Ptr<DisparityWLSFilter> wls_filter;
    double matching_time, filtering_time;

    Mat m_imageRight, m_imageLeft, img1, img2;
    Mat imgU1, imgU2, grayDisp1, grayDisp2;

    Mat GT_disp, left_for_matcher, right_for_matcher;
    Mat left_disp, right_disp, filtered_disp, conf_map;

    Mat filtered_disp_vis, raw_disp_vis;
    Mat imgCalorHSV, imgAdd, imgCalorHOT, imgCalorBONE;

    /*Caution: the images path is absolute (hardcoded). You can change to user
    argv, or change these two lines below to your desired path. The path here is
    defined for KITTI dataset images
    (http://www.cvlibs.net/datasets/kitti/raw_data.php).*/
    VideoCapture videoOne("./imageL0.png"); // Absolute path to
                                            // the KITTI left
                                            // grey frames
    VideoCapture videoTwo("./imageR0.png"); // Absolute path to
                                            // the KITTI right
                                            // grey frames

    int width, height;

    width = videoOne.get(3);
    height = videoOne.get(4);

    cout << "width: " << width << endl;
    cout << "height: " << height << endl;

    VideoWriter videoOutAllTwo, videoOutAllFour, videoOutAllFive,
        videoOutAllSix;

    videoOutAllTwo = cv::VideoWriter(
        "originalEsq.avi", VideoWriter::fourcc('M', 'J', 'P', 'G'), 30,
        Size(width, height), true); // To save the original left images as video
    videoOutAllFive =
        cv::VideoWriter("MDBONE.avi", VideoWriter::fourcc('M', 'J', 'P', 'G'),
                        30, Size(width, height),
                        true); // To save the DM results as BONE colormap video.
    videoOutAllSix =
        cv::VideoWriter("MDHOT.avi", VideoWriter::fourcc('M', 'J', 'P', 'G'),
                        30, Size(width, height),
                        true); // To save the DM results as HOT colormap video.

    while (1) {

        videoOne >> m_imageLeft;
        videoTwo >> m_imageRight;

        // Histogram equalization to deal with illumination problems
        /*cv::cvtColor(m_imageLeft, m_imageLeft, CV_BGR2Lab);
        std::vector<cv::Mat> channels;
        cv::split(m_imageLeft, channels);
        cv::equalizeHist(channels[0], channels[0]);
        cv::merge(channels, m_imageLeft);
        cv::cvtColor(m_imageLeft, m_imageLeft, CV_Lab2BGR);


        cv::cvtColor(m_imageRight, m_imageRight, CV_BGR2Lab);
        std::vector<cv::Mat> channelsTwo;
        cv::split(m_imageRight, channelsTwo);
        cv::equalizeHist(channels[0], channelsTwo[0]);
        cv::merge(channelsTwo, m_imageRight);
        cv::cvtColor(m_imageRight, m_imageRight, CV_Lab2BGR);*/
        // Histogram equalization ends here

        if (!m_imageLeft.data || !m_imageRight.data) {
            printf(" No image data \n ");
            /* return -1; */
        }

        imgDisparity8U = Mat(m_imageRight.rows, m_imageRight.cols, CV_8UC1);
        filter = "wls_conf"; // Post-filter
        algo = "sgbm"; // Defines which OpenCV algorithm was used, BM or SGBM
        dst_path = "None";
        dst_raw_path = "None";
        dst_conf_path = "None";

        max_disp = 160; // 160
        lambda = 8000.0;
        sigma = 3.5;
        vis_mult = 3.0;

        wsize = 3; // 3 if SGBM
                   // wsize = 15; // if BM, 7 or 15

        conf_map = Mat(m_imageLeft.rows, m_imageLeft.cols, CV_8U);
        conf_map = Scalar(255);
        Rect ROI;

        // Results better than "wls_no_conf"
        if (filter == "wls_conf") {
            if (!no_downscale) { // This is done to leave faster, but for a
                                 // better result, avoid using.
                max_disp /= 2;
                if (max_disp % 16 != 0) {
                    max_disp += 16 - (max_disp % 16);
                }
                resize(m_imageLeft, left_for_matcher, Size(), 0.5, 0.5);
                resize(m_imageRight, right_for_matcher, Size(), 0.5, 0.5);
            } else {
                left_for_matcher = m_imageLeft.clone();
                right_for_matcher = m_imageRight.clone();
            }

            /* The filter instance is created by providing the instance of the
             * StereoMatcher Another instance is returned by createRightMatcher.
             * These two instances are used to calculate the DM's for the right
             * and left images, this is necessary for filtering afterwards.
             */
            if (algo == "bm") {
                Ptr<StereoBM> left_matcher = StereoBM::create(max_disp, wsize);
                wls_filter = createDisparityWLSFilter(left_matcher);
                Ptr<StereoMatcher> right_matcher =
                    createRightMatcher(left_matcher);

                cvtColor(left_for_matcher, left_for_matcher, COLOR_BGR2GRAY);
                cvtColor(right_for_matcher, right_for_matcher, COLOR_BGR2GRAY);

                matching_time = (double)getTickCount();
                left_matcher->compute(left_for_matcher, right_for_matcher,
                                      left_disp);
                right_matcher->compute(right_for_matcher, left_for_matcher,
                                       right_disp);
                matching_time = ((double)getTickCount() - matching_time) /
                                getTickFrequency();
            } else if (algo == "sgbm") {
                Ptr<StereoSGBM> left_matcher =
                    StereoSGBM::create(0, max_disp, wsize);
                left_matcher->setP1(24 * wsize * wsize);
                left_matcher->setP2(96 * wsize * wsize);
                left_matcher->setPreFilterCap(63);
                left_matcher->setMode(StereoSGBM::MODE_SGBM_3WAY);
                wls_filter = createDisparityWLSFilter(left_matcher);
                Ptr<StereoMatcher> right_matcher =
                    createRightMatcher(left_matcher);

                matching_time = (double)getTickCount();
                left_matcher->compute(left_for_matcher, right_for_matcher,
                                      left_disp);
                right_matcher->compute(right_for_matcher, left_for_matcher,
                                       right_disp);
                matching_time = ((double)getTickCount() - matching_time) /
                                getTickFrequency();
            }

            /* Filter
             * MD calculated by the respective match instances, just as the
             * left image is passed to the filter.
             * Note that we are using the original image to guide the filtering
             * process.
             */
            wls_filter->setLambda(lambda);
            wls_filter->setSigmaColor(sigma);
            filtering_time = (double)getTickCount();
            wls_filter->filter(left_disp, m_imageLeft, filtered_disp,
                               right_disp);
            filtering_time =
                ((double)getTickCount() - filtering_time) / getTickFrequency();

            conf_map = wls_filter->getConfidenceMap();

            // Get the ROI that was used in the last filter call:
            ROI = wls_filter->getROI();
            if (!no_downscale) {
                // Upscale raw disparity and ROI back for a proper comparison:
                resize(left_disp, left_disp, Size(), 2.0, 2.0);
                left_disp = left_disp * 2.0;
                ROI = Rect(ROI.x * 2, ROI.y * 2, ROI.width * 2, ROI.height * 2);
            }
        }

        else if (filter == "wls_no_conf") {
            /* There is no convenience function for the case of filtering with
            no confidence, so we will need to set the ROI and matcher parameters
            manually */

            left_for_matcher = m_imageLeft.clone();
            right_for_matcher = m_imageRight.clone();

            if (algo == "bm") {
                Ptr<StereoBM> matcher = StereoBM::create(max_disp, wsize);
                matcher->setTextureThreshold(0);
                matcher->setUniquenessRatio(0);
                cvtColor(left_for_matcher, left_for_matcher, COLOR_BGR2GRAY);
                cvtColor(right_for_matcher, right_for_matcher, COLOR_BGR2GRAY);
                ROI = computeROI(left_for_matcher.size(), matcher);
                wls_filter = createDisparityWLSFilterGeneric(false);
                wls_filter->setDepthDiscontinuityRadius(
                    (int)ceil(0.33 * wsize));

                matching_time = (double)getTickCount();
                matcher->compute(left_for_matcher, right_for_matcher,
                                 left_disp);
                matching_time = ((double)getTickCount() - matching_time) /
                                getTickFrequency();
            } else if (algo == "sgbm") {
                Ptr<StereoSGBM> matcher =
                    StereoSGBM::create(0, max_disp, wsize);
                matcher->setUniquenessRatio(0);
                matcher->setDisp12MaxDiff(1000000);
                matcher->setSpeckleWindowSize(0);
                matcher->setP1(24 * wsize * wsize);
                matcher->setP2(96 * wsize * wsize);
                matcher->setMode(StereoSGBM::MODE_SGBM_3WAY);
                ROI = computeROI(left_for_matcher.size(), matcher);
                wls_filter = createDisparityWLSFilterGeneric(false);
                wls_filter->setDepthDiscontinuityRadius((int)ceil(0.5 * wsize));

                matching_time = (double)getTickCount();
                matcher->compute(left_for_matcher, right_for_matcher,
                                 left_disp);
                matching_time = ((double)getTickCount() - matching_time) /
                                getTickFrequency();
            }

            wls_filter->setLambda(lambda);
            wls_filter->setSigmaColor(sigma);
            filtering_time = (double)getTickCount();
            wls_filter->filter(left_disp, m_imageLeft, filtered_disp, Mat(),
                               ROI);
            filtering_time =
                ((double)getTickCount() - filtering_time) / getTickFrequency();
        }

        // collect and print all the stats:
        // cout.precision(2);
        // cout<<"Matching time:  "<<matching_time<<"s"<<endl;
        // cout<<"Filtering time: "<<filtering_time<<"s"<<endl;
        // cout<<endl;

        if (dst_path != "None") {
            // Mat filtered_disp_vis;
            getDisparityVis(filtered_disp, filtered_disp_vis, vis_mult);
            imwrite(dst_path, filtered_disp_vis);
        }
        if (dst_raw_path != "None") {
            // Mat raw_disp_vis;
            getDisparityVis(left_disp, raw_disp_vis, vis_mult);
            imwrite(dst_raw_path, raw_disp_vis);
        }
        if (dst_conf_path != "None") {
            imwrite(dst_conf_path, conf_map);
        }

        if (!no_display) {
            /*//Displays the original images
            namedWindow("left", WINDOW_AUTOSIZE);
            imshow("left", imgLeft);
            namedWindow("right", WINDOW_AUTOSIZE);
            imshow("right", imgRight);*/

            /*if(!noGT)
            {
                Mat GT_disp_vis;
                getDisparityVis(GT_disp,GT_disp_vis,vis_mult);
                namedWindow("ground-truth disparity", WINDOW_AUTOSIZE);
                imshow("ground-truth disparity", GT_disp_vis);
            }*/

            /*//Displays DM without filter
            Mat raw_disp_vis;
            getDisparityVis(left_disp,raw_disp_vis,vis_mult);
            namedWindow("raw disparity", WINDOW_AUTOSIZE);
            imshow("raw disparity", raw_disp_vis);*/

            // Displays filtered DM
            getDisparityVis(filtered_disp, filtered_disp_vis, vis_mult);
            namedWindow("filtered disparity", WINDOW_AUTOSIZE);
            imshow("filtered disparity", filtered_disp_vis);

            /* Color Maps:
             * OpenCV method to change a grayscale image to a color model.
             * The human vision may have difficulty perceiving small
             * differences in shades of gray, but better perceives the
             * changes between colors.
             *
             * More Info:
             * http://docs.opencv.org/3.1.0/d3/d50/group__imgproc__colormap.html#gsc.tab=0
             */
            // Applying color maps (different DM visualization)
            applyColorMap(filtered_disp_vis, imgCalorBONE, COLORMAP_BONE);
            applyColorMap(filtered_disp_vis, imgCalorHOT, COLORMAP_HOT);

            // imshow("Left image", m_imageLeft);
            // imshow("Right image", m_imageRight);

            videoOutAllTwo.write(m_imageLeft);
            videoOutAllFive.write(imgCalorBONE);
            videoOutAllSix.write(imgCalorHOT);

            key = (char)waitKey(10000);
            if (key == 27) {
                break;
            }
        }
    }
    return 0;
}
