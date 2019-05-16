#include <cmath>
#include <iostream>
#include <stdio.h>

#include "opencv2/core.hpp"
#include <opencv2/core/utility.hpp>
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/cudaimgproc.hpp"

#include <librealsense2/rs.hpp>
#include <librealsense2/rsutil.h>

#include <atomic>
#include <chrono>
#include <cmath>
#include <iostream>
#include <map>
#include <mutex>
#include <thread>
using namespace std;
using namespace cv;
using namespace cv::cuda;

static void help()
{
    cout << "This program demonstrates line finding with the Hough transform." << endl;
    cout << "Usage:" << endl;
    cout << "./gpu-example-houghlines <image_name>, Default is ../data/pic1.png\n" << endl;
}
/* Helper function to display window. Hit escape to close application */
static bool displayUi(int xTarget, int yTarget, Mat& image)
{
    namedWindow("Show Image", WINDOW_OPENGL);
    imshow("Display Image", image);

    char button;
    button = waitKey(1);

    if (button == 27) {
        destroyAllWindows();
        return true;
    }

    return false;
}
int main(int argc, const char* argv[])
{	
	const char* gst =  "nvcamerasrc ! video/x-raw(memory:NVMM), width=(int)1280, height=(int)720, format=(string)I420, framerate=(fraction)120/1 ! \
           	    nvvidconv   ! video/x-raw, format=(string)BGRx ! \
                    videoconvert ! video/x-raw, format=(string)BGR ! \
                    appsink";
	VideoCapture cap(gst); // open the onboard camera
	if(!cap.isOpened())  // check if we succeeded
        	return -1;
	
	auto start = std::chrono::high_resolution_clock::now(); //Get start time for frame rate
	char frameRate[10];
	bool alive = true;
	while(alive) {
		Mat frame;
		Mat mask;
		Mat dst_cpu;
        	cap >> frame; // get a new frame from camera
		GaussianBlur( frame, frame, Size( 3, 3 ), 0, 0 );	

		
		cv::Canny(frame, mask, 195, 250, 3);

		cv::cvtColor(mask, dst_cpu, COLOR_GRAY2BGR);
		Mat dst_gpu = dst_cpu.clone();

		//Uncomment following block for CPU
		/*
		    vector<Vec4i> lines_cpu;
		    {
			const int64 start = getTickCount();
			cv::HoughLinesP(mask, lines_cpu, 1, CV_PI / 180, 50, 60, 5);
			const double timeSec = (getTickCount() - start) / getTickFrequency();
			cout << "CPU Time : " << timeSec * 1000 << " ms" << endl;
			cout << "CPU Found : " << lines_cpu.size() << endl;
		    }

		    for (size_t i = 0; i < lines_cpu.size(); ++i)
		    {
			Vec4i l = lines_cpu[i];
			line(dst_cpu, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(0, 0, 255), 3, LINE_AA);
		    }
		*/

		GpuMat d_src(mask);
		GpuMat d_lines;
		{
		const int64 start = getTickCount();
		Ptr<cuda::HoughSegmentDetector> hough = cuda::createHoughSegmentDetector(1.0f, (float) (CV_PI / 180.0f), 50, 5);
		hough->detect(d_src, d_lines);
		const double timeSec = (getTickCount() - start) / getTickFrequency();
		cout << "GPU Time : " << timeSec * 1000 << " ms" << endl;
		cout << "GPU Found : " << d_lines.cols << endl;
		}
		vector<Vec4i> lines_gpu;
		if (!d_lines.empty())
		{
		lines_gpu.resize(d_lines.cols);
		Mat h_lines(1, d_lines.cols, CV_32SC4, &lines_gpu[0]);
		d_lines.download(h_lines);
		}

		for (size_t i = 0; i < lines_gpu.size(); ++i)
		{
		Vec4i l = lines_gpu[i];
		line(dst_gpu, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(0, 0, 255), 3, LINE_AA);
		}
		
		auto elapsed = std::chrono::high_resolution_clock::now() - start;
		float milliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count();
		float frames = 1000/milliseconds;			
		snprintf(frameRate, sizeof(frameRate), "%02f\n", frames);
		putText(dst_gpu, frameRate, Point(50, 50), FONT_HERSHEY_SCRIPT_SIMPLEX, 2, Scalar::all(255), 3, 3);

		start = std::chrono::high_resolution_clock::now();

		auto startUI = std::chrono::high_resolution_clock::now();
		if (displayUi(0, 0, dst_gpu)) {
	        	/* Signal to threads to end */
	        	alive = false;
	    	}
		auto endUI = std::chrono::high_resolution_clock::now();
		auto elapsedUI = endUI - startUI;
		float timeUI = std::chrono::duration_cast<std::chrono::milliseconds>(elapsedUI).count();
		printf("Displaying the UI: %f\n", timeUI);
		if( waitKey (30) >= 0) break;
	}
    return 0;
}
