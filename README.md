# GPU_RealSense_Testing_cookem4
### A variety of files exploring the cause for latency seen with interfacing the Intel RealSense D435 and the Nvidia Jetson TX2 ###

## Summary of Findings: ##
The results of performing tests with these projects are as follows:
#### Main Findings: ####
* CUDA works properly on the board
* A cuda program was created to multiply two matrices that is able to run much faster on the GPU than the CPU. When running on the GPU the device was able to reach a full 99% load while with all other openCV applications load peaks around 50%
* Display of the UI in openCV takes about 10-50ms which can prevent fast framerates
* The main bottle neck is obtaining the depth and color frames from Realsense which takes about 100ms
Matrix multiplication with the GpuMat class was 5 times slower than matrix multiplication using purely the CPU – perhaps the openCV GPU class is not well optimized or that the architecture being used is not very compatible with the CUDA code in openCV GPU
* There are some openCV GPU functions that are more efficient on the CPU than the GPU – for example canny edge detection. On the other hand some functions run faster on the GPU such as drawing Hough lines
* An openCV program with GPU acceleration that performs hough edge detection (a favorable GPU operation) running at 1920x1080 via the Realsense runs at about 3 FPS. The GPU operations for hough lines takes about 50ms while the majority of the time is spent obtaining Realsense frames and displaying the UI
* An openCV program with GPU acceleration that performs hough edge detection at 1280x720 running off of the jetson onboard camera runs at about 10FPS – Canny edge detection via CPU, GPU processing of hough lines, and displaying the UI are the major steps
* GPU operations vs CPU operations in openCV were never more than 50% faster
* An openGL application was able to display video feed at close to expected rate
* It might be thought that pre-processing on the realsense camera could cause the slow frame rates seen in openCV. However, the frames can be received fast with openGL

#### Summary: ####
Receiving frames from the Realsense camera in openCV is the main bottleneck to increasing frequency and takes around 100ms. Displaying the UI window in openCV also causes a drop in frequency as it can take up to 50ms, averaging around 20ms. The GPU has varying performance with openCV that is completely dependent on the nature of the calculations of the image manipulation function. The performance increase of the GPU vs the CPU in openCV is nowhere near as effective compared to running pure matrix operations with CUDA. 

#### There are a few options: ####
* Find a solution that uses purely openGL perhaps creating custom CUDA code for some basic image operations
Create a solution that combines openCV processing with openGL display
* Look at using a different computer vision library that takes advantage of the graphics card more effectively
* Do not use the Realsense camera and move to using a regular RGB camera so that frames can be received much faster in openCV. Position and orientation can be found of a marker based on its relative size and shape to the camera. Simple color filtering can be done as with my project last year and the center of mass of the selected color marker can be found
	

