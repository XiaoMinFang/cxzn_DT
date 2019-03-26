#ifndef DTCOMMON_HPP
#define DTCOMMON_HPP

#include <opencv2/opencv.hpp>            // C++
#include <opencv2/tracking/tracking.hpp>
#include "yolo_v2_class.hpp"
#include "cxzn_tracking.hpp"

void draw_detect_boxes(cv::Mat mat_img, std::vector<bbox_t> result_vec, std::vector<std::string> obj_names,int current_det_fps, int current_cap_fps);
std::vector<std::string> get_objNames_fromFile(std::string const filename);
void draw_tracking_res(cv::Mat img,std::vector<res_track_t> res_track,cv::Scalar color,std::string text);
void draw_boxes(cv::Mat img, cv::Rect2d box,cv::Scalar color,std::string text);


#endif // DTCOMMON_HPP
