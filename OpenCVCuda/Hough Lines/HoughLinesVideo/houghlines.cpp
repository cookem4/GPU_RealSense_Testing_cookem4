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
#include "rsCam.hpp"

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
    //Attempt to initialize CUDA Device
	cv::cuda::setDevice(0);
	cv::cuda::printCudaDeviceInfo(0);

	struct rsConfig rsCfg;

	/* Set up parameters.  Note this application does not use IR, but only depth and color */
	rsCfg.depthRes = {1280, 720};
	rsCfg.irRes    = {1280, 720};
	rsCfg.colorRes = {1920, 1080};
	rsCfg.colorFps = 30;
	rsCfg.irFps    = 30;
	rsCfg.depthFps = 30;

	/* Initialize camera */
	initRsCam(rsCfg);
	
	auto stream = rsCfg.rsProfile.get_stream(RS2_STREAM_COLOR).as<rs2::video_stream_profile>();
	
	const struct rs2_intrinsics intrinsics = stream.get_intrinsics();

	rs2::frame_queue frameQueue(5);
	std::atomic_bool alive {true};

	/* This thread used solely to receive frames and check if color and depth frames are valid */
	std::thread rxFrame([&]() {
	while(alive) {

	    rs2::frameset frames = rsCfg.rsPipe.wait_for_frames();

	    auto colorFrame = frames.get_color_frame();
	    auto depthFrame = frames.get_depth_frame();

	   if (!colorFrame || !depthFrame) {
		continue;
	   }
	    frameQueue.enqueue(frames);    
	}
	});
	
	rs2::frameset curFrame;
	auto start = std::chrono::high_resolution_clock::now(); //Get start time for frame rate
	char frameRate[10];
	rs2::align align(rsCfg.rsAlignTo);
	while(alive) {

		/* Receive frames from other thread here */
        	frameQueue.poll_for_frame(&curFrame);
		if (curFrame) {
			
			auto startRS= std::chrono::high_resolution_clock::now();	
			auto processed = align.process(curFrame);
			//Obtain colour and depth frames
			rs2::video_frame other_frame = processed.first_or_default(rsCfg.rsAlignTo);
			rs2::depth_frame depth       = processed.get_depth_frame();
			auto endRS = std::chrono::high_resolution_clock::now();
			auto elapsedRS = endRS- startRS;
			float timeRS = std::chrono::duration_cast<std::chrono::milliseconds>(elapsedRS).count();
			printf("Obtaining the RealSense Frames: %f\n", timeRS);


			int color_width  = other_frame.get_width();
			int color_height = other_frame.get_height();
			int depth_width  = depth.get_width();
			int depth_height = depth.get_height();

			auto startMats = std::chrono::high_resolution_clock::now();		
			Mat origFrame(Size(color_width, color_height), CV_8UC3,  (void*)other_frame.get_data(), Mat::AUTO_STEP);
			Mat depthFrame(Size(depth_width, depth_height), CV_16UC1, (void*)depth.get_data(), Mat::AUTO_STEP);		
			auto endMats = std::chrono::high_resolution_clock::now();
			auto elapsedMats = endMats- startMats;
			float timeMats = std::chrono::duration_cast<std::chrono::milliseconds>(elapsedMats).count();
			printf("RealSense to Mats Conversion: %f\n", timeMats);

			Mat mask;
			cv::Canny(origFrame, mask, 100, 200, 3);

			Mat dst_cpu;
			cv::cvtColor(mask, dst_cpu, COLOR_GRAY2BGR);
			Mat dst_gpu = dst_cpu.clone();

			//Uncomment following block for CPU
			
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
		}
	}
    return 0;
}
