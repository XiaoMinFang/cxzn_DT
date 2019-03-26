#include <iostream>
#include <iomanip>
#include <string>
#include <cstdlib>
#include <sstream>
#include <cstring>

#include <pthread.h>

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

#include "load_lidar_data.hpp"
#include "cxzn_tracking.hpp"
#include "DTCommon.hpp"







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



int vedio_main()
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
	Mat show_img;
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
//    Ptr<TrackerMOSSE> tracking_test;
//    tracking_test = TrackerMOSSE::create();
    Rect2d rect_obj;
    vector<bbox_t> obj_result_vec;
    vector<res_track_t> res_trcking;

    int n_tracks=0;
    bool first_frame = true;
    int count = 0;
    float max_iou =-1;
    int max_iou_obj_id = 0;
    while (1){
        cap >> mat_img;
        show_img = mat_img.clone();
        count++;
        if (mat_img.rows == 0 || mat_img.cols == 0)
            break;
        if (first_frame)
        {
            obj_result_vec = detector.detect(mat_img);
            //draw_detect_boxes(show_img, obj_result_vec, obj_names);
            n_tracks = obj_result_vec.size();
            if (n_tracks < 1) continue; //没有检测到目标就再来一帧
            //初始化track
//            init_tracking(tracking,obj_result_vec, mat_img);
//            init_res_tracking(res_trcking,obj_result_vec);
////////////////////////////////////////////////////////////////
//            Rect2d rect_obj;
//            rect_obj.height = obj_result_vec[0].h; rect_obj.width = obj_result_vec[0].w;
//			rect_obj.x = obj_result_vec[0].x; rect_obj.y = obj_result_vec[0].y;
//            tracking_test->init(mat_img,rect_obj);

            tracking = init_tracking(obj_result_vec, mat_img,n_tracks);
            res_trcking = init_res_tracking(obj_result_vec,n_tracks);

            first_frame = false;
        }
        else
        {
            //update_tracking(tracking,res_trcking,mat_img,n_tracks); //更新
            for (int i = 0; i < MAX_TRACKING_NUM; i++)
            {
                if(tracking[i].is_acti)
                {
                    Rect2d track_res;
                    tracking[i].tracker->update(mat_img, track_res);
                    if (track_res.x == 0 & track_res.y==0 & track_res.width == 0 & track_res.height == 0)
                    {
                        cout<<">>>>>> no trackings\n"<<endl;
                        //增加检测 找到与上一个跟踪框iou最大的box，作为本次的跟踪框
                        obj_result_vec = detector.detect(mat_img);

                        if (obj_result_vec.size() < 1) //也没有检测到，就删除吧。。。。。。
                        {
                            cout<<"can not detected obj\n"<<endl;
                        }else{
                            track_res = find_max_iou(obj_result_vec,res_trcking[i].cur_rect);
                            draw_boxes(show_img, track_res, Scalar(0, 255, 0), "track_from_detect");
                            cout<<"detect box >> track box"<<endl;
                            cout<<track_res.x<<","<<track_res.y<<","<<track_res.width<<","<<track_res.height<<endl;
                        }


                    }

                    ////// 计算速度
                    track_speed_t speed = calc_speed(track_res,res_trcking[i].cur_rect);
                    res_trcking[i].vx = speed.vx;
                    res_trcking[i].vy = speed.vy;
                    res_trcking[i].direction = calc_direction(res_trcking[i].cur_rect);
                    res_trcking[i].cur_rect = track_res;
                    res_trcking[i].id = i;
                    res_trcking[i].l_tracking++;
                    //draw_boxes(show_img, track_res, Scalar(0, 255, 0), "track");
                    /*imshow("tracking", mat_img);
                    waitKey(1);*/
                    //destroyAllWindows;
                    //cvWaitKey(0);
                    }
            }
            if (count%n_interval == 0)
            {//重新更新跟踪结果
                if (obj_result_vec.size() == 0) //若前面己经detect过的，就不用再detect一次了
                {
                    obj_result_vec = detector.detect(mat_img);
                    //draw_detect_boxes(mat_img, obj_result_vec, obj_names);
                }
                //若某跟踪框与所有的检测框的iou<0.5,就删除该跟踪框；
                for (int j=0;j<MAX_TRACKING_NUM;j++)
                {
                    if(tracking[j].is_acti)
                    {
                        bool is_del = is_del_rect(obj_result_vec,res_trcking[j].cur_rect);
                        if(is_del)
                        {
                            tracking[j].is_acti = false;
                            //tracking[j].tracker = TrackerMOSSE::clear();
                            res_trcking[j].cur_rect = zero_rect();
                            res_trcking[j].init_rect = zero_rect();
                            res_trcking[j].l_tracking = 0;
                            res_trcking[j].id = -1;
                            res_trcking[j].vx = 0;
                            res_trcking[j].vy =0;
                            res_trcking[j].direction = 0;
                            res_trcking[j].is_acti = 0;
                         }
                    }

                }
                //若某检测框与所有的跟踪框的iou<0.5,就添加一个跟踪框；
                for (int j=0;j<obj_result_vec.size();j++)
                {
                    bool is_add = is_add_rect(res_trcking,obj_result_vec[j]);
                    if(is_add)
                    {//加在哪？？？ 找第一个l_tracking = 0
                        Rect2d obj_rect =bboxt_2_rect(obj_result_vec[j]);
                        int idx = find_unusing_id(tracking);//idx = -1时，说明不止MAX_TRACKING_NUM个tracks
                        tracking[idx].id = idx;
                        tracking[idx].init_rects = obj_rect;
                        tracking[idx].tracker = TrackerMOSSE::create();
                        tracking[idx].tracker->init(mat_img,obj_rect);
                        res_trcking[idx].cur_rect = obj_rect;
                        res_trcking[idx].init_rect = obj_rect;
                        res_trcking[idx].id = idx;
                        res_trcking[idx].l_tracking = 0;
                        res_trcking[idx].vx = 0;
                        res_trcking[idx].vy = 0;
                        res_trcking[idx].direction = 0;
                        res_trcking[idx].is_acti = true;
                    }
                }


//                /*imshow("detect", mat_img);
//                waitKey(1);*/
//                if (obj_result_vec.size() >= 1)
//                {
//                    n_tracks = obj_result_vec.size();
////                    init_tracking(tracking,obj_result_vec, mat_img);
////                    init_res_tracking(res_trcking,obj_result_vec);
//                    tracking = init_tracking(obj_result_vec, mat_img);
//                    res_trcking = init_res_tracking(obj_result_vec);
//                    putText(mat_img, "reset tracking", Point2f(20, 20), FONT_HERSHEY_COMPLEX_SMALL, 1.2, Scalar(255, 0, 0), 1);
//                    cout << "reset tracking\n" << endl;
//                }
//                //rectangle(mat_img, objects, Scalar(0, 255, 0), 2);

            }

            //result_vec = detector.tracking_id(result_vec);    // comment it - if track_id is not required
        }
        //
        draw_tracking_res(show_img,res_trcking,Scalar(0, 255, 0),"track");
        //draw_detect_boxes(show_img, obj_result_vec, obj_names);
        obj_result_vec.clear();
        imshow("res", show_img);
        waitKey(1);
        writer << show_img;

    }

	writer.release();
	cap.release();
	return 1;
}

//
int pcl_main()
{
	int n_interval = 5;//间隔几帧，进行检测，更新tracking结果

	string  names_file = "/home/kid/min/lidar_dl/data/names.list";
    string  cfg_file = "/home/kid/min/detectAndtrack/lidar_3c_tiny_pcl.cfg";
    string  weights_file = "/home/kid/min/detectAndtrack/lidar_3c_tiny_pcl_52000.weights";
	//string filename = "E:\\work_min\\data\\lidar\\lidar_data\\datas\\feature_3c_anno1\\feature_3c\\2\\1499.png";

	string write_video_name = "/home/kid/min/detectAndtrack/res_track.avi";
	float const thresh = 0.25;

	Detector_YOLO detector(cfg_file, weights_file);
	auto obj_names = get_objNames_fromFile(names_file);

	ifstream fin160("/home/kid/min/Annotations/LiDar/anno3/vlp160.txt", ios::in);
    ifstream fin161("/home/kid/min/Annotations/LiDar/anno3/vlp161.txt", ios::in);
    string velo_path_160,velo_path_161;

    Mat mat_img;Mat show_img;
    pcl::PointCloud<pcl::PointXYZI>::Ptr point_cloud_velo(new pcl::PointCloud<pcl::PointXYZI>);
	VideoWriter writer;
    velo_data_t velo_points;

    getline(fin160,velo_path_160);
    getline(fin161,velo_path_161);
    velo_points = read_velo_data(velo_path_160,velo_path_161);
    process_single(point_cloud_velo,velo_points);
    get_img(mat_img,point_cloud_velo);
    writer.open("/home/kid/min/res.avi", CV_FOURCC('M', 'J', 'P', 'G'), 10, Size(480,480), true);


    vector<track_t> tracking;
    Rect2d rect_obj;
    vector<bbox_t> obj_result_vec;
    vector<res_track_t> res_trcking;
    vector<res_ouput_t> outputs;
    int n_tracks=0;
    bool first_frame = true;
    int count = 0;
    while(!fin160.eof() && !fin161.eof())
    {
        getline(fin160,velo_path_160);
        getline(fin161,velo_path_161);
        velo_points = read_velo_data(velo_path_160,velo_path_161);
        pcl::PointCloud<pcl::PointXYZI>::Ptr point_cloud_velo(new pcl::PointCloud<pcl::PointXYZI>);
        process_single(point_cloud_velo,velo_points);
        get_img(mat_img,point_cloud_velo);

        if (mat_img.rows == 0 || mat_img.cols == 0) return (0);
        show_img = mat_img.clone();
//        cv::imshow("img",show_img);
//        cv::waitKey(0);

        count++;

        if (first_frame)
        {
            obj_result_vec = detector.detect(mat_img);
            //draw_detect_boxes(show_img, obj_result_vec, obj_names);
            n_tracks = obj_result_vec.size();
            if (n_tracks < 1) continue; //没有检测到目标就再来一帧
            //初始化track
//            init_tracking(tracking,obj_result_vec, mat_img);
//            init_res_tracking(res_trcking,obj_result_vec);
////////////////////////////////////////////////////////////////
//            Rect2d rect_obj;
//            rect_obj.height = obj_result_vec[0].h; rect_obj.width = obj_result_vec[0].w;
//			rect_obj.x = obj_result_vec[0].x; rect_obj.y = obj_result_vec[0].y;
//            tracking_test->init(mat_img,rect_obj);

            tracking = init_tracking(obj_result_vec, mat_img,n_tracks);
            res_trcking = init_res_tracking(obj_result_vec,n_tracks);

            first_frame = false;
        }
        else
        {
            //update_tracking(tracking,res_trcking,mat_img,n_tracks); //更新
            for (int i = 0; i < MAX_TRACKING_NUM; i++)
            {
                if(tracking[i].is_acti)
                {
                    Rect2d track_res;
                    tracking[i].tracker->update(mat_img, track_res);
                    if (track_res.x == 0 & track_res.y==0 & track_res.width == 0 & track_res.height == 0)
                    {
                        cout<<">>>>>> no trackings\n"<<endl;
                        //增加检测 找到与上一个跟踪框iou最大的box，作为本次的跟踪框
                        obj_result_vec = detector.detect(mat_img);

                        if (obj_result_vec.size() < 1) //也没有检测到，就删除吧。。。。。。
                        {
                            cout<<"can not detected obj\n"<<endl;
                        }else{
                            track_res = find_max_iou(obj_result_vec,res_trcking[i].cur_rect);
                            //draw_boxes(show_img, track_res, Scalar(0, 255, 0), "track_from_detect");
                            cout<<"detect box >> track box"<<endl;
                            cout<<track_res.x<<","<<track_res.y<<","<<track_res.width<<","<<track_res.height<<endl;
                        }


                    }

                    ////// 计算速度
                    track_speed_t speed = calc_speed(track_res,res_trcking[i].cur_rect);
                    res_trcking[i].vx = speed.vx;
                    res_trcking[i].vy = speed.vy;
                    res_trcking[i].direction = calc_direction(res_trcking[i].cur_rect);
                    res_trcking[i].cur_rect = track_res;
                    res_trcking[i].id = i;
                    res_trcking[i].l_tracking++;
                    //draw_boxes(show_img, track_res, Scalar(0, 255, 0), "track");
                    /*imshow("tracking", mat_img);
                    waitKey(1);*/
                    //destroyAllWindows;
                    //cvWaitKey(0);
                    }
            }
            if (count%n_interval == 0)
            {//重新更新跟踪结果
                if (obj_result_vec.size() == 0) //若前面己经detect过的，就不用再detect一次了
                {
                    obj_result_vec = detector.detect(mat_img);
                    //draw_detect_boxes(mat_img, obj_result_vec, obj_names);
                }
                //若某跟踪框与所有的检测框的iou<0.5,就删除该跟踪框；
                for (int j=0;j<MAX_TRACKING_NUM;j++)
                {
                    if(tracking[j].is_acti)
                    {
                        bool is_del = is_del_rect(obj_result_vec,res_trcking[j].cur_rect);
                        if(is_del)
                        {
                            tracking[j].is_acti = false;
                            //tracking[j].tracker = TrackerMOSSE::clear();
                            res_trcking[j].cur_rect = zero_rect();
                            res_trcking[j].init_rect = zero_rect();
                            res_trcking[j].l_tracking = 0;
                            res_trcking[j].id = -1;
                            res_trcking[j].vx = 0;
                            res_trcking[j].vy =0;
                            res_trcking[j].direction = 0;
                            res_trcking[j].is_acti = false;
                         }
                    }

                }
                //若某检测框与所有的跟踪框的iou<0.5,就添加一个跟踪框；
                for (int j=0;j<obj_result_vec.size();j++)
                {
                    bool is_add = is_add_rect(res_trcking,obj_result_vec[j]);
                    if(is_add)
                    {//加在哪？？？ 找第一个l_tracking = 0
                        Rect2d obj_rect =bboxt_2_rect(obj_result_vec[j]);
                        int idx = find_unusing_id(tracking);//idx = -1时，说明不止MAX_TRACKING_NUM个tracks
                        tracking[idx].is_acti = true;
                        tracking[idx].id = idx;
                        tracking[idx].init_rects = obj_rect;
                        tracking[idx].tracker = TrackerMOSSE::create();
                        tracking[idx].tracker->init(mat_img,obj_rect);
                        res_trcking[idx].cur_rect = obj_rect;
                        res_trcking[idx].init_rect = obj_rect;
                        res_trcking[idx].id = idx;
                        res_trcking[idx].l_tracking = 0;
                        res_trcking[idx].vx = 0;
                        res_trcking[idx].vy = 0;
                        res_trcking[idx].direction = 0;
                        res_trcking[idx].is_acti = true;
                    }
                }
            }
        }
        draw_tracking_res(show_img,res_trcking,Scalar(0, 255, 0),"track");

        outputs = convert_resTrack(res_trcking);
        print_output(outputs);
        outputs.clear();

        //draw_detect_boxes(show_img, obj_result_vec, obj_names);
        obj_result_vec.clear();
        imshow("res", show_img);
        waitKey(1);
        writer << show_img;
    }
    writer.release();
	return 1;
}
