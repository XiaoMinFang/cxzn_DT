#ifndef CXZH_TRACKING_HPP
#define CXZH_TRACKING_HPP


#include <opencv2/opencv.hpp>            // C++
#include <opencv2/tracking/tracking.hpp>

#include "yolo_v2_class.hpp"

#define MAX_TRACKING_NUM 10 // 最多tracking的个数

#define FRAME_TIME_INTERVAL 0.1 //帧时间间隔(s)
#define PIXEL2M 0.125//8 PIXEL = 1 M
#define SRC_POINT_X 240
#define SRC_POINT_Y 400
//#define HALF_PI 1.5707 //PI/2

struct track_speed_t{
    float vx;
    float vy;
};


//track_t 与 res_track_t 根据 id 对应
struct track_t {
	cv::Rect2d init_rects;                // 初始跟踪框
	int id;                        // 跟踪 id
	bool is_acti;                  //是否激活
	//cv::Rect2d update_rect;                  // 跟踪结果
	cv::Ptr<cv::TrackerMOSSE> tracker;       //tracker

};
struct res_track_t {
	bool is_acti;                  //是否激活
	cv::Rect2d init_rect;                // 初始跟踪框
	cv::Rect2d cur_rect;                // 当前跟踪结果
	int id;                        // 跟踪 id
	int l_tracking;                 //跟踪长度
	float vx;                   //横向速度
	float vy;                   //纵向速度
	float direction;             //方向角  以车体为中心的坐标系，以正前方为标准方向 = 0度，左边为负，右边为正 [-PI.+PI]
};
struct res_ouput_t{
//车体坐标系,最终返回值
    int loc_x; //x,以车体坐标系为主
    int loc_y;
    int loc_w;
    int loc_h;
    int l_tracking;                 //跟踪长度
	float vx;                   //横向速度
	float vy;                   //纵向速度
	float direction;
};

cv::Rect2d bboxt_2_rect(bbox_t b);
cv::Rect2d zero_rect();

int find_unusing_id(std::vector<track_t> tracking);

float box_iou(cv::Rect2d a, cv::Rect2d b);
track_speed_t calc_speed(cv::Rect2d a, cv::Rect2d b);
float calc_direction(cv::Rect2d obj);

cv::Rect2d find_max_iou(std::vector<bbox_t> obj_result_vec,cv::Rect2d pre_rect);
bool is_del_rect(std::vector<bbox_t> obj_result_vec,cv::Rect2d rect);
bool is_add_rect(std::vector<res_track_t> res_track, bbox_t obj_result);

//void init_res_tracking(std::vector<res_track_t> &res_tracking,std::vector<bbox_t> obj_rect);
//void init_tracking(std::vector<track_t> &tracking,std::vector<bbox_t> obj_rect,cv::Mat mat_img);
std::vector<res_track_t> init_res_tracking(std::vector<bbox_t> obj_rect,int n_tracks);
std::vector<track_t> init_tracking(std::vector<bbox_t> obj_rect,cv::Mat mat_img,int n_tracks);

std::vector<res_track_t> release_res_tracking(std::vector<bbox_t> obj_rect);
std::vector<track_t> release_tracking(std::vector<bbox_t> obj_rect,cv::Mat mat_img);

//void update_tracking(std::vector<track_t> &tracking,std::vector<res_track_t> &res_trcking,cv::Mat mat_img,int n_tracks);
std::vector<res_ouput_t> convert_resTrack(std::vector<res_track_t> src_track);
void print_output(std::vector<res_ouput_t> out);
//void release_tracking(std::vector<track_t> tracking);



#endif // CXZH_TRACKING_H
