#include <iostream>
#include <iomanip>
#include <string>
#include <cstdlib>
#include <sstream>
#include <cstring>

#include <vector>
#include <queue>
#include <fstream>
#include <thread>
#include <atomic>
#include <mutex>              // mutex, unique_lock
#include <condition_variable> // condition_variable


#include "yolo_v2_class.hpp"    // imported functions from DLL


#include <opencv2/opencv.hpp>            // C++
#include <opencv2/core/version.hpp>
#include <opencv2/tracking/tracking.hpp>
#include <opencv2/videoio/videoio.hpp>

#include "load_data.hpp"
#include "cxzn_tracking.hpp"







using namespace std;
using namespace cv;



void read_velo_data(string velofilename)
{
    int arr_counts[1] = {0};
	int counts = 0;

	string velo_filename = "/home/kid/min/Annotations/LiDar/anno3/veloseq/VLP160/1/648.bin";
	ifstream fin(velo_filename, ios::binary);
	if(!fin)
	{
		cout << "read "<<velofilename<<" error\n" <<endl;
	}
	fin.read((char*)arr_counts,sizeof(int));
	if (arr_counts[0] != 0) counts = arr_counts[0];

	cout<<counts<<endl;

	float x[counts];float y[counts];
	float z[counts];float r[counts];

	fin.read((char*)x,counts*sizeof(float));
	fin.read((char*)y,counts*sizeof(float));
	fin.read((char*)z,counts*sizeof(float));
	fin.read((char*)r,counts*sizeof(float));

	fin.close();

}




void draw_boxes(Mat img, Rect2d box,Scalar color,string text)
{
	rectangle(img, box, color, 1);
	Size const text_size = getTextSize(text, FONT_HERSHEY_COMPLEX_SMALL, 1.2, 2, 0);
	int const max_width = (text_size.width > box.width + 2) ? text_size.width : (box.width + 2);
	/*rectangle(img, Point2f(max((int)box.x - 1, 0), max((int)box.y - 30, 0)),
		Point2f(min((int)box.x + max_width, img.cols - 1), min((int)box.y, img.rows - 1)),color, CV_FILLED, 8, 0);*/
	putText(img, text, Point2f(box.x, box.y - 10), FONT_HERSHEY_COMPLEX_SMALL, 1.2, color, 1);

}
void draw_detect_boxes(Mat mat_img, vector<bbox_t> result_vec, vector<string> obj_names,
	int current_det_fps = -1, int current_cap_fps = -1)
{
	int const colors[6][3] = { { 1,0,1 },{ 0,0,1 },{ 0,1,1 },{ 0,1,0 },{ 1,1,0 },{ 1,0,0 } };

	for (auto &i : result_vec) {
		Scalar color = obj_id_to_color(i.obj_id);
		rectangle(mat_img, Rect(i.x, i.y, i.w, i.h), color, 1);
		if (obj_names.size() > i.obj_id) {
			string obj_name = obj_names[i.obj_id];
			if (i.track_id > 0) obj_name += " - " + to_string(i.track_id);
			Size const text_size = getTextSize(obj_name, FONT_HERSHEY_COMPLEX_SMALL, 1.2, 2, 0);
			int const max_width = (text_size.width > i.w + 2) ? text_size.width : (i.w + 2);
			/*rectangle(mat_img, Point2f(max((int)i.x - 1, 0), max((int)i.y - 30, 0)),
				Point2f(min((int)i.x + max_width, mat_img.cols - 1), min((int)i.y, mat_img.rows - 1)),
				color, CV_FILLED, 8, 0);*/
			putText(mat_img, obj_name, Point2f(i.x, i.y - 10), FONT_HERSHEY_COMPLEX_SMALL, 1.2, color, 1);
		}
	}
	if (current_det_fps >= 0 && current_cap_fps >= 0) {
		string fps_str = "FPS detection: " + to_string(current_det_fps) + "   FPS capture: " + to_string(current_cap_fps);
		putText(mat_img, fps_str, Point2f(10, 20), FONT_HERSHEY_COMPLEX_SMALL, 1.2, Scalar(50, 255, 0), 2);
	}
}

vector<string> get_objNames_fromFile(string const filename) {
	ifstream file(filename);
	vector<string> file_lines;
	if (!file.is_open()) return file_lines;
	for (string line; getline(file, line);) file_lines.push_back(line);
	cout << "object names loaded \n";
	return file_lines;
}


int video_main()
{
	int n_interval = 10;//间隔几帧，进行检测，更新tracking结果

	string  names_file = "/home/kid/min/lidar_dl/data/names.list";
    string  cfg_file = "/home/kid/min/lidar_dl/data/model_480_480_tiny/lidar_tiny.cfg";
    string  weights_file = "/home/kid/min/lidar_dl/data/model_480_480_tiny/lidar_tiny_final.weights";
	//string filename = "E:\\work_min\\data\\lidar\\lidar_data\\datas\\feature_3c_anno1\\feature_3c\\2\\1499.png";
	string filename = "/home/kid/min/1.avi";
	string write_video_name = "/home/kid/min/res_track.avi";
	float const thresh = 0.25;

	Detector_YOLO detector(cfg_file, weights_file);
	auto obj_names = get_objNames_fromFile(names_file);

	//Mat mat_img = imread(filename);
	Mat mat_img;
	VideoWriter writer;
	VideoCapture cap(filename);

    cap >> mat_img;

    bool isColor = (mat_img.type() == CV_8UC3);
    writer.open(write_video_name, CV_FOURCC('M', 'J', 'P', 'G'), 5.0, mat_img.size(), true);
    if (!writer.isOpened()) {
        cerr << "Could not open the output video file for write\n";
        return -1;
    }

    vector<track_t> tracking;
    //Ptr<TrackerMOSSE> tracking_test;
    Rect2d rect_obj;
    vector<bbox_t> obj_result_vec;
    vector<res_track_t> res_trcking;

    int n_tracks=0;
    bool first_frame = true;
    int count = 0;
    while (1){
        cap >> mat_img;
        count++;
        if (mat_img.rows == 0 || mat_img.cols == 0)
            break;
        if (first_frame)
        {
            obj_result_vec = detector.detect(mat_img);
            draw_detect_boxes(mat_img, obj_result_vec, obj_names);
            n_tracks = obj_result_vec.size();
            if (n_tracks < 1) continue; //没有检测到目标就再来一帧
            //初始化track
            init_tracking(tracking,obj_result_vec, mat_img);
            init_res_tracking(res_trcking,obj_result_vec);

            first_frame = false;
        }
        else
        {
            obj_result_vec = detector.detect(mat_img);
            draw_detect_boxes(mat_img, obj_result_vec, obj_names);

            update_tracking(tracking,res_trcking,mat_img,n_tracks); //更新



            if (count%n_interval == 0)
            {//重新更新跟踪结果
                //若某跟踪框与所有的检测框的iou<0.5,就删除该跟踪框；
                //若某检测框与所有的跟踪框的iou<0.5,就添加一个跟踪框；
                /*imshow("detect", mat_img);
                waitKey(1);*/
                if (obj_result_vec.size() >= 1)
                {
                    n_tracks = obj_result_vec.size();
                    init_tracking(tracking,obj_result_vec, mat_img);
                    init_res_tracking(tracking,obj_result_vec);
                    putText(mat_img, "reset tracking", Point2f(20, 20), FONT_HERSHEY_COMPLEX_SMALL, 1.2, Scalar(255, 0, 0), 1);
                    cout << "reset tracking\n" << endl;
                }
                //rectangle(mat_img, objects, Scalar(0, 255, 0), 2);

                // ...........

            }


            //result_vec = detector.tracking_id(result_vec);    // comment it - if track_id is not required

        }
        //
        imshow("detect", mat_img);
        waitKey(1);
        writer << mat_img;

    }

	writer.release();
	cap.release();
	return 1;
}


int pcl_main()
{
	int n_interval = 10;//间隔几帧，进行检测，更新tracking结果

	string  names_file = "/home/kid/min/lidar_dl/data/names.list";
    string  cfg_file = "/home/kid/min/detectAndtrack/lidar_1c_tiny_pcl.cfg";
    string  weights_file = "/home/kid/min/detectAndtrack/lidar_1c_tiny_pcl_45000.weights";
	//string filename = "E:\\work_min\\data\\lidar\\lidar_data\\datas\\feature_3c_anno1\\feature_3c\\2\\1499.png";

	string write_video_name = "/home/kid/min/detectAndtrack/res_track.avi";
	float const thresh = 0.25;

	Detector_YOLO detector(cfg_file, weights_file);
	auto obj_names = get_objNames_fromFile(names_file);

	string velo_path_160 = "/home/kid/min/Annotations/LiDar/anno3/veloseq/0/VLP160/";
	string velo_path_161 = "/home/kid/min/Annotations/LiDar/anno3/veloseq/0/VLP161/";

	int s_id = 97;int e_id = 603;

	Mat mat_img;
	VideoWriter writer;
	string bin_s = ".bin";
    string velo_160 = velo_path_160+to_string(s_id) + bin_s;
    string velo_161 = velo_path_161+to_string(s_id) + bin_s;
    cout<< velo_160<<","<<velo_161<<endl;
    velo_data_t velo_points;
    velo_points = read_velo_data(velo_path_160,velo_path_161);

    //get_img(mat_img,velo_points);
    if (mat_img.rows == 0 || mat_img.cols == 0) return (0);
    cv::imshow("img",mat_img);
    cv::waitKey(0);


    vector<track_t> tracking;
    //Ptr<TrackerMOSSE> tracking_test;
    Rect2d rect_obj;
    vector<bbox_t> obj_result_vec;
    vector<res_track_t> res_trcking;
    int n_tracks=0;
    bool first_frame = true;
    int count = 0;
	for (int i=s_id;i<e_id;i++)
    {
        string bin_s = ".bin";
        string velo_160 = velo_path_160+to_string(i) + bin_s;
        string velo_161 = velo_path_161+to_string(i) + bin_s;
        velo_points = read_velo_data(velo_path_160,velo_path_161);
        //get_img(mat_img,velo_points);
        cv::imshow("img",mat_img);
        cv::waitKey(0);

        count++;
        if (first_frame){
            obj_result_vec = detector.detect(mat_img);
            draw_detect_boxes(mat_img, obj_result_vec, obj_names);
            n_tracks = obj_result_vec.size();
            if (n_tracks < 1) continue; //没有检测到目标就再来一帧
            //初始化track
            tracking = init_tracking(obj_result_vec, mat_img);
            res_trcking = init_res_tracking(obj_result_vec);
            first_frame = false;
        }else
        {
            obj_result_vec = detector.detect(mat_img);
            draw_detect_boxes(mat_img, obj_result_vec, obj_names);
            for (int i = 0; i < n_tracks; i++)
            {
                Rect2d track_res;
                tracking[i].tracker->update(mat_img, track_res);
                if (track_res.x == 0 & track_res.y==0 & track_res.width == 0 & track_res.height == 0)
                {
                    cout<<">>>>>> no trackings\n"<<endl;
                }
                res_trcking[i].res_track = track_res;
                res_trcking[i].id = i;
                res_trcking[i].l_tracking++;
                draw_boxes(mat_img, track_res, Scalar(0, 255, 0), "track");
                /*imshow("tracking", mat_img);
                waitKey(1);*/
                //destroyAllWindows;
                //cvWaitKey(0);
            }
            if (count%n_interval == 0)
            {//重新更新跟踪结果
                    //若某跟踪框与所有的检测框的iou<0.5,就删除该跟踪框；
                    //若某检测框与所有的跟踪框的iou<0.5,就添加一个跟踪框；
                    /*imshow("detect", mat_img);
                    waitKey(1);*/
                    if (obj_result_vec.size() >= 1)
                    {
                        n_tracks = obj_result_vec.size();
                        tracking = init_tracking(obj_result_vec, mat_img);
                        res_trcking = init_res_tracking(obj_result_vec);
                        putText(mat_img, "reset tracking", Point2f(20, 20), FONT_HERSHEY_COMPLEX_SMALL, 1.2, Scalar(255, 0, 0), 1);
                        cout << "reset tracking\n" << endl;
                    }
                    //rectangle(mat_img, objects, Scalar(0, 255, 0), 2);

                    // ...........

                }

            imshow("detect", mat_img);
            waitKey(1);
            writer << mat_img;

        }
    }
	writer.release();
	return 1;
}
