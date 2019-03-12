#include <opencv2/core/utility.hpp>
#include <opencv2/tracking/tracking.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <cstring>
using namespace std;
using namespace cv;

int __main(int argc, char** argv) {
	// 
	int obj_n = 1;
	clock_t startTime, endTime;
	// 初始化，创建
	string trackingAlg = "MIL";//"BOOSTING","MIL","TLD","MEDIANFLOW","KCF","MOSSE"
	Ptr<Tracker> tracker = Tracker::create(trackingAlg);
	Rect2d objects;


	//设置video
	//string video_name = "E:\\work_min\\test.mp4";
	string video_name = "E:\\work_min\\1.avi";
	//string video_name = "E:\\work_min\\demo.mp4";
	
	
	string write_video_name = "E:\\work_min\\res_mil.avi";
	VideoCapture cap(video_name);
	VideoWriter writer;
	if (!cap.isOpened())
	{
		cout << "open video error\n" << endl;
		return 0;
	}

	
	Mat frame;
	cap >> frame;
	bool isColor = (frame.type() == CV_8UC3);
	writer.open(write_video_name, CV_FOURCC('M', 'J', 'P', 'G'), 10.0, frame.size(), true);
	if (!writer.isOpened()) {
		cerr << "Could not open the output video file for write\n";
		return -1;
	}
	//   !!!!! do 
	int count = 0;
	bool first_frame = true;
	double time_all = 0.0;
	while (1)
	{
		cap >> frame;
		if (frame.rows == 0 || frame.cols == 0) break;
		count++;
		if (count < 0) 
		{ 
			continue;
		}
		if (first_frame)
		{// 获取 目标框, 初始化 tracker	
			objects = selectROI("tracker", frame, false,false);
			tracker->init(frame, objects);
			first_frame = false;
		}
		else 
		{
			startTime = clock();
			tracker->update(frame,objects);
			endTime = clock();
			time_all += (double)(endTime - startTime) / CLOCKS_PER_SEC;
			//画框
			rectangle(frame, objects, Scalar(0, 255, 0), 2);
			//show
			imshow("tracker", frame);
			writer<<frame;
			if (waitKey(1) == 27)break; //按esc退出
		}
		
	}
	cout << "time average cost " << time_all / count << " s" << endl;
	writer.release();
	cap.release();
	return 1;
}