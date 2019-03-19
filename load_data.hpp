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


////点云边
#define MINX -10.0
#define MAXX 50.0
#define MINY -30.0
#define MAXY 30.0
#define MINZ -1.6
#define MAXZ 2.4



void load_data(cv::Mat& img_src,std::string velo_filename160,std::string velo_filename161);


#endif // LOAD_DATA_HPP
