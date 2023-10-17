#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>

int main() {
    // Load the video
    cv::VideoCapture cap(
        "/home/omie_sawie/Code_Code/OmkarSawantBTP_SLAM_Photogrammetry/"
        "OpenCV_CPP/Photogrammetry_OpenCV_CPP/videoFeatureMatching/"
        "/resources/carOnLonelyRoads.mp4");
    if (!cap.isOpened()) {
        std::cerr << "Error: Could not open video file." << std::endl;
        return -1;
    }

    // Load camera calibration parameters
    cv::Mat cameraMatrix; // Intrinsic matrix
    cv::Mat distortionCoefficients;
    // Load or calculate cameraMatrix and distortionCoefficients

    // Loop through video frames
    cv::Mat frame;
    while (cap.read(frame)) {
        // Undistort the frame using camera calibration parameters if needed

        // Feature detection and tracking
        std::vector<cv::KeyPoint> keypoints;
        cv::Mat descriptors;
        // Use feature detection and tracking algorithms (e.g., ORB, KLT) here

        // Stereo matching or depth estimation
        cv::Mat depthMap;
        // Calculate depth map using stereo vision or other methods

        // Generate 3D point cloud
        std::vector<cv::Point3f> pointCloud;
        for (int i = 0; i < keypoints.size(); i++) {
            float x = keypoints[i].pt.x;
            float y = keypoints[i].pt.y;
            float depth = depthMap.at<float>(y, x); // Get depth from depth map
            cv::Point3f point(x, y, depth);
            pointCloud.push_back(point);
        }
        std::cout << pointCloud.size() << std::endl;

        cv::imshow("frame", frame);

        // Visualize the 3D point cloud or perform further processing
        //
        cv::namedWindow("Point Cloud", cv::WINDOW_AUTOSIZE);

        // Create a blank image for visualization
        cv::Mat pointCloudImage = cv::Mat::zeros(500, 500, CV_8UC3);

        // Example: Create a vector of 3D points (x, y, z)
        /* std::vector<cv::Point3f> pointCloud; */
        /* pointCloud.push_back(cv::Point3f(0.0, 0.0, 1.0)); // Point at (0, 0,
         * 1) */
        /* pointCloud.push_back(cv::Point3f(1.0, 1.0, 2.0)); // Point at (1, 1,
         * 2) */
        /* pointCloud.push_back( */
        /*     cv::Point3f(-1.0, -1.0, 0.5)); // Point at (-1, -1, 0.5) */

        // Define parameters for visualization
        double focalLength = 500.0; // Focal length (adjust as needed)
        cv::Point2f principalPoint(pointCloudImage.cols / 2.0,
                                   pointCloudImage.rows /
                                       2.0); // Principal point

        // Visualize each point in the point cloud
        for (const cv::Point3f &point3D : pointCloud) {
            // Project 3D point to 2D using focal length and principal point
            double x2D =
                (focalLength * point3D.x / point3D.z) + principalPoint.x;
            double y2D =
                (focalLength * point3D.y / point3D.z) + principalPoint.y;

            // Draw a point on the image
            cv::circle(pointCloudImage, cv::Point(x2D, y2D), 2,
                       cv::Scalar(0, 0, 255), -1); // Red point
        }

        // Display the point cloud image
        cv::imshow("Point Cloud", pointCloudImage);
        cv::waitKey(0);
    }
    // Wait for a key press and close the window when a key is pressed
    cv::waitKey(0);
    cv::destroyAllWindows();

    return 0;
}
