#ifndef CXZH_TRACKING_HPP
#define CXZH_TRACKING_HPP


#include <opencv2/opencv.hpp>            // C++
#include <opencv2/tracking/tracking.hpp>

#include "yolo_v2_class.hpp"

#define MAX_TRACKING_NUM 10 // 最多tracking的个数

struct track_t {
	cv::Rect2d init_rects;                // 初始跟踪框
	int id;                        // 跟踪 id
	bool is_acti;                  //是否激活
	cv::Rect2d update_rect;                  // 跟踪结果
	cv::Ptr<cv::TrackerMOSSE> tracker;   //

};
struct res_track_t {
	cv::Rect2d res_track;                // 初始跟踪框
	int id;                        // 跟踪 id
	int l_tracking;                 //跟踪长度
};


//std::vector<res_track_t> init_res_tracking(std::vector<bbox_t> obj_rect);
void init_res_tracking(vector<res_track_t> &res_tracking,vector<bbox_t> obj_rect);
void init_tracking(vector<track_t> &tracking,vector<bbox_t> obj_rect,Mat mat_img);

//std::vector<track_t> init_tracking(std::vector<bbox_t> obj_rect,cv::Mat mat_img);

void update_tracking(vector<track_t> &tracking,vector<res_track_t> &res_trcking,Mat mat_img,int n_tracks)


void release_tracking(std::vector<track_t> tracking);



#endif // CXZH_TRACKING_H
