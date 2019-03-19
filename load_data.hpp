#ifndef LOAD_DATA_HPP
#define LOAD_DATA_HPP

#include<iostream>
#include<fstream>
#include<string>
#include<cmath>

//#include<boost/thread.hpp>
//#include<boost/timer.hpp>

#include <pcl/visualization/cloud_viewer.h>
#include <pcl/io/io.h>
#include <pcl/io/pcd_io.h>

#include <pcl/point_types.h>

//#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/passthrough.h>
//#include <pcl/filters/statistical_outlier_removal.h>
//#include <pcl/ModelCoefficients.h>
//#include <pcl/filters/project_inliers.h>
#include <pcl/kdtree/kdtree_flann.h>

//#include <pcl/common/transforms.h>
//#include <pcl/correspondence.h>

#include <opencv2/opencv.hpp>

//#define OVERLOP_NUM 3
////点云边
#define MINX -10.0
#define MAXX 50.0
#define MINY -30.0
#define MAXY 30.0
#define MINZ -1.6
#define MAXZ 2.4

#define GROUND_LIMIT_MIN 0.5
#define GROUND_LIMIT_MAX 5
#define OVERLOP_NUM 3


struct velo_data_t {
    int counts;
	float *x;
	float *y;
	float *z;
	int *r;
};
class LOAD_LIDAR_DATA{
    std::vector<velo_data_t> points_velo_list;
};



void get_img(cv::Mat& img_src,velo_data_t velo_points);

velo_data_t load_data(std::string velo_filename160,std::string velo_filename161);

pcl::PointCloud<pcl::PointXYZI>::Ptr passthrough_filter(pcl::PointCloud<pcl::PointXYZI>::Ptr point_cloud_velo);


int lidar_process(velo_data_t velo_points);


#endif // LOAD_DATA_HPP
