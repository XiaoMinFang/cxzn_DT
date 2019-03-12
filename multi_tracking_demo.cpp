#include <opencv2/core/utility.hpp>
#include <opencv2/tracking/tracking.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <cstring>
using namespace std;
using namespace cv;

//version opencv 3.1
//��֪��Ϊʲô��û�����������������


int _main(int argc, char** argv) {
	// 
	int obj_n = 1;
	// ��ʼ��������
	string trackingAlg = "KCF";//"BOOSTING","MIL","TLD","MEDIANFLOW","KCF"
	MultiTracker multi_tracker(trackingAlg);

	//
	vector<Rect2d> objects(obj_n);
	vector<Rect2d> outputs;
	Rect2d box;


	//����video
	//string video_name = "E:\\work_min\\test.mp4";
	//string video_name = "E:\\work_min\\1.avi";
	string video_name = "E:\\work_min\\demo.mp4";
	VideoCapture cap(video_name);
	if (!cap.isOpened())
	{
		cout << "open video error\n" << endl;
		return 0;
	}


	Mat frame;

	//   !!!!! do 
	int count = 0;
	bool first_frame = true;
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
		{// ��ȡ Ŀ���, ��ʼ�� tracker	
			for (int i = 0; i < obj_n; i++)
			{
				objects[i] = selectROI("tracker", frame, false);
				multi_tracker.add(trackingAlg, frame, objects[i]);
			}
			first_frame = false;
		}
		else
		{
			multi_tracker.update(frame);
			//tracker->update(frame,box);

			//����
			for (int i = 0; i<multi_tracker.objects.size(); i++)
				rectangle(frame, multi_tracker.objects[i], Scalar(255, 0, 0), 2, 1);
			//show
			imshow("tracker", frame);
			if (waitKey(1) == 27)break; //��esc�˳�
		}

	}

	return 1;
}