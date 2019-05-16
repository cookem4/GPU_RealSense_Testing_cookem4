#include <iostream>
#include "opencv2/opencv.hpp"
#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/core/cuda.hpp"
#include "opencv2/core/opengl.hpp"
#include "opencv2/imgproc.hpp"

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

using namespace cv;
using namespace std;

//To go from Mat to GpuMat
//cv::cuda::GpuMat src
//src.upload(REGULAR MAT)
//src now has the Mat copied as a gpu mat

//To go from GpuMat to Mat
//send in the GpuMat to the Mat constructor


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
Ptr<cv::cuda::CannyEdgeDetector> canny;
int main (int argc, char* argv[])
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

	rs2::align align(rsCfg.rsAlignTo);
	printf("TEST");
	printf("%d\n",cv::cuda::getCudaEnabledDeviceCount());
	canny = cv::cuda::createCannyEdgeDetector(10,200,3,false);
	auto start = std::chrono::high_resolution_clock::now(); //Get start time for frame rate
	char frameRate[10];

	//Variables to be used within the loop
	/////////////////////////////////////////////////////////////
	cv::cuda::GpuMat dst1, dst2, src;
	cv::cuda::GpuMat shsv[3];
	cv::cuda::GpuMat thresc[3];
	cv::cuda::GpuMat temp;
	/////////////////////////////////////////////////////////////
	
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

			

			//////////////////////////////////////
			//Uncomment for CPU Processing////////
			//////////////////////////////////////
			//Mat result_host;
			/*
			auto startColor = std::chrono::high_resolution_clock::now();		
			cvtColor(origFrame,origFrame,CV_BGR2GRAY);
			auto endColor = std::chrono::high_resolution_clock::now();
			auto elapsedColor = endColor- startColor;
			float timeColor = std::chrono::duration_cast<std::chrono::milliseconds>(elapsedColor).count();
			printf("Color Conversion Step: %f\n", timeColor);
			*/
			/*
			auto startCanny = std::chrono::high_resolution_clock::now();		
			Canny(origFrame,result_host,10,200,3);
			auto endCanny = std::chrono::high_resolution_clock::now();
			auto elapsedCanny = endCanny- startCanny;
			float timeCanny = std::chrono::duration_cast<std::chrono::milliseconds>(elapsedCanny).count();
			printf("Canny Step: %f\n", timeCanny);
			*/
			
			/*
			//Take the square of a matrix
			auto startMult = std::chrono::high_resolution_clock::now();
			multiply(origFrame,origFrame,result_host);
			for(int i = 0; i < 100; i++){
				multiply(result_host,result_host,result_host);
			}
			auto endMult = std::chrono::high_resolution_clock::now();
			auto elapsedMult = endMult - startMult;
			float timeMult = std::chrono::duration_cast<std::chrono::milliseconds>(elapsedMult).count();
			printf("CPU Matrix Multiplication: %f\n", timeMult);
			*/

			///////////////////////////////////////
			//Uncomment next 3 blocks for GPU//////
			///////////////////////////////////////
			
			
			//Putting Mat to GPU Memory
			
			auto startConvert = std::chrono::high_resolution_clock::now();
			src.upload(origFrame);
			auto endConvert = std::chrono::high_resolution_clock::now();
			auto elapsedConvert = endConvert - startConvert;
			float timeConvert = std::chrono::duration_cast<std::chrono::milliseconds>(elapsedConvert).count();
			printf("Mat to GpuMat: %f\n", timeConvert);
			
			//cv::cuda::threshold(src, dst, 128.0, 255.0, cv::THRESH_BINARY);
			
			
			///////////////////////////////////////
			//Uncomment for Matrix Multiplication//
			///////////////////////////////////////
			//Take the square of a matrix
			auto startMult = std::chrono::high_resolution_clock::now();
			cv::cuda::multiply(src,src,dst1);
			for(int i = 0; i < 100; i++){
				cv::cuda::multiply(dst1,dst1,dst1);
			}
			auto endMult = std::chrono::high_resolution_clock::now();
			auto elapsedMult = endMult - startMult;
			float timeMult = std::chrono::duration_cast<std::chrono::milliseconds>(elapsedMult).count();
			printf("GPU Matrix Multiplication: %f\n", timeMult);			

			auto startConvert2 = std::chrono::high_resolution_clock::now();
			Mat result_host(dst1);
			auto endConvert2 = std::chrono::high_resolution_clock::now();
			auto elapsedConvert2 = endConvert2 - startConvert2;
			float timeConvert2 = std::chrono::duration_cast<std::chrono::milliseconds>(elapsedConvert2).count();
			printf("GpuMat to Mat: %f\n", timeConvert2);

			///////////////////////////////////////
			//Uncomment for Colour Filetering//////
			///////////////////////////////////////
			/*
			auto startColor = std::chrono::high_resolution_clock::now();
			cv::cuda::cvtColor(src,dst1,CV_BGR2HSV);
			cv::cuda::split(dst1,shsv);
			cv::cuda::threshold(shsv[0], thresc[0], 45, 90, THRESH_BINARY);
			cv::cuda::threshold(shsv[1], thresc[1], 100, 225, THRESH_BINARY);
			cv::cuda::threshold(shsv[2], thresc[2], 20, 225, THRESH_BINARY);			
			cv::cuda::bitwise_and(thresc[0], thresc[1], temp);
			cv::cuda::bitwise_and(temp, thresc[2], dst2);
			auto endColor = std::chrono::high_resolution_clock::now();
			auto elapsedColor = endColor- startColor;
			float timeColor = std::chrono::duration_cast<std::chrono::milliseconds>(elapsedColor).count();
			printf("Color Conversion Step: %f\n", timeColor);

			auto startConvert2 = std::chrono::high_resolution_clock::now();
			Mat result_host(dst2);
			auto endConvert2 = std::chrono::high_resolution_clock::now();
			auto elapsedConvert2 = endConvert2 - startConvert2;
			float timeConvert2 = std::chrono::duration_cast<std::chrono::milliseconds>(elapsedConvert).count();
			printf("GpuMat to Mat: %f\n", timeConvert2);
			*/

			/////////////////////////////////////
			//Uncomment for Canny Edge Detection/
			/////////////////////////////////////
			
			/*
			auto startColor = std::chrono::high_resolution_clock::now();			
			cv::cuda::cvtColor(src,dst1,CV_BGR2GRAY);		
			auto endColor = std::chrono::high_resolution_clock::now();
			auto elapsedColor = endColor- startColor;
			float timeColor = std::chrono::duration_cast<std::chrono::milliseconds>(elapsedColor).count();
			printf("Color Conversion Step: %f\n", timeColor);


	
			auto startCanny = std::chrono::high_resolution_clock::now();			
			canny->detect(dst1,dst2);
			auto endCanny = std::chrono::high_resolution_clock::now();
			auto elapsedCanny = endCanny- startCanny;
			float timeCanny = std::chrono::duration_cast<std::chrono::milliseconds>(elapsedCanny).count();
			printf("Canny Step: %f\n", timeCanny);
			

			auto startConvert2 = std::chrono::high_resolution_clock::now();
			Mat result_host(dst2);
			auto endConvert2 = std::chrono::high_resolution_clock::now();
			auto elapsedConvert2 = endConvert2 - startConvert2;
			float timeConvert2 = std::chrono::duration_cast<std::chrono::milliseconds>(elapsedConvert2).count();
			printf("GpuMat to Mat: %f\n", timeConvert2);

			*/

			//Do not uncomment this
			auto elapsed = std::chrono::high_resolution_clock::now() - start;
			float milliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count();
			float frames = 1000/milliseconds;			
			snprintf(frameRate, sizeof(frameRate), "%02f\n", frames);
			putText(result_host, frameRate, Point(50, 50), FONT_HERSHEY_SCRIPT_SIMPLEX, 2, Scalar::all(255), 3, 3);

			start = std::chrono::high_resolution_clock::now();

			auto startUI = std::chrono::high_resolution_clock::now();
			if (displayUi(0, 0, result_host)) {
		        	/* Signal to threads to end */
		        	alive = false;
		    	}
			auto endUI = std::chrono::high_resolution_clock::now();
			auto elapsedUI = endUI - startUI;
			float timeUI = std::chrono::duration_cast<std::chrono::milliseconds>(elapsedUI).count();
			printf("Displaying the UI: %f\n", timeUI);
		}
	}

/*
	rs2::pipeline pipe;
	rs2::config cfg;
	cfg.enable_stream(RS2_STREAM_COLOR, 848, 480, RS2_FORMAT_BGR8, 60);
	pipe.start(cfg);

	// Camera warmup - dropping several first frames to let auto-exposure stabilize
	rs2::frameset frames;
	for(int i = 0; i < 30; i++)
	{
		//Wait for all configured streams to produce a frame
		frames = pipe.wait_for_frames();
	}
		
	//Get each frame
	rs2::frame color_frame = frames.get_color_frame();
	cv::cuda::GpuMat dst, src;
	//Build GPU mat from regular mat	
	Mat color(Size(848, 480), CV_8UC3, (void*)color_frame.get_data(), Mat::AUTO_STEP);
	src.upload(color);
	cv::cuda::threshold(src, dst, 128.0, 255.0, cv::THRESH_BINARY);
	Mat result_host(dst);
	

	namedWindow("Realsense Feed", WINDOW_AUTOSIZE);
	imshow("Display Image", result_host);

	waitKey(0);

        cv::Mat src_host = cv::imread("/home/nvidia/image.jpeg", cv::IMREAD_GRAYSCALE);
        cv::cuda::GpuMat dst, src;
        src.upload(src_host);
        cv::cuda::threshold(src, dst, 128.0, 255.0, cv::THRESH_BINARY);
        cv::Mat result_host(dst);
        cv::imshow("Result", result_host);
        cv::waitKey();
*/
    
    rxFrame.join();
    return 0;
}
