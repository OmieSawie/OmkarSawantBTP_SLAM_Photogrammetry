#include <iostream>
#include <iterator>
#include <opencv2/core.hpp>
#include <opencv2/core/cuda.hpp>

#include <opencv2/core/utility.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <string>

#include "bits/time.h"
#include <cmath>
#include <ctime>
#include <iostream>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <opencv2/core/cuda.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudaimgproc.hpp>

#define TestCUDA true

using namespace std;
using namespace cv;

VideoCapture cap(0);

void cpuSpeedTest() {
    while (cap.isOpened()) {
        Mat image;
        bool isSuccess = cap.read(image);
        if (image.empty()) {
            cout << "Could not load image! " << endl;
        }
        auto start = getTickCount();
        Mat result;
        bilateralFilter(image, result, 30, 100, 100);
        auto end = getTickCount();

        auto totalTIme = (end - start) / getTickFrequency();
        auto fps = 1 / totalTIme;
        cout << "FPS: " << fps << endl;

        putText(result, "FPS: " + to_string(int(fps)), Point(50, 50),
                FONT_HERSHEY_DUPLEX, 1, Scalar(0, 255, 255));
        imshow("Image", result);

        int k = waitKey(10);
        if (k == 119) {
            break;
        }
    }
    cap.release();
    destroyAllWindows();
}

void gpuSpeedTest() {
    while (cap.isOpened()) {
        Mat image;
        bool isSuccess = cap.read(image);

        cuda::GpuMat imgGPU;

        cout << cuda::getCudaEnabledDeviceCount() << endl;
        imgGPU.upload(image);

        if (imgGPU.empty()) {
            cout << "Could not load image on GPU! " << endl;
        }
        auto start = getTickCount();

        cuda::bilateralFilter(imgGPU, imgGPU, 30, 100, 100);
        auto end = getTickCount();

        auto totalTIme = (end - start) / getTickFrequency();
        auto fps = 1 / totalTIme;
        cout << "FPS: " << fps << endl;
        imgGPU.download(image);
        putText(image, "FPS: " + to_string(int(fps)), Point(50, 50),
                FONT_HERSHEY_DUPLEX, 1, Scalar(0, 255, 255));
        imshow("Image", image);

        int k = waitKey(10);
        if (k == 119) {
            break;
        }
    }
    cap.release();
    destroyAllWindows();
}

int main(int, char **) {
    gpuSpeedTest();
    /* cpuSpeedTest(); */
    return 1;
}

/* int main() { */
/*     std::clock_t begin = std::clock(); */

/*     try { */
/*         cv::String filename = */
/*             "/home/omie_sawie/Pictures/ArchLinux_SnowMountains.jpg"; */
/*         cv::Mat srcHost = cv::imread(filename, cv::IMREAD_GRAYSCALE); */

/*         for (int i = 0; i < 1000; i++) { */
/*             if (TestCUDA) { */
/*                 cv::cuda::GpuMat dst, src; */
/*                 src.upload(srcHost); */

/*                 // cv::cuda::threshold(src,dst,128.0,255.0,
 * CV_THRESH_BINARY); */
/*                 /1* cv::cuda::bilateralFilter(src, dst, 3, 1, 1); *1/ */

/*                 cv::Mat resultHost; */
/*                 dst.download(resultHost); */
/*             } else { */
/*                 cv::Mat dst; */
/*                 cv::bilateralFilter(srcHost, dst, 3, 1, 1); */
/*             } */
/*         } */

/*         // cv::imshow("Result",resultHost); */
/*         // cv::waitKey(); */

/*     } catch (const cv::Exception &ex) { */
/*         std::cout << "Error: " << ex.what() << std::endl; */
/*     } */

/*     std::clock_t end = std::clock(); */
/*     std::cout << double(end - begin) / CLOCKS_PER_SEC << std::endl; */
/* } */
