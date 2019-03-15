#include<iostream>
#include<fstream>
#include<string>

#include<boost/thread.hpp>
#include<boost/timer.hpp>

#include <pcl/visualization/cloud_viewer.h>
#include <pcl/io/io.h>
#include <pcl/io/pcd_io.h>

#include <pcl/point_types.h>

#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/ModelCoefficients.h>
#include <pcl/filters/project_inliers.h>

#include <opencv2/opencv.hpp>


#define random(x1,x2) ((rand()%x2) - x1/2.0)

////点云边
#define MINX -10.0
#define MAXX 50.0
#define MINY -30.0
#define MAXY 30.0
#define MINZ -1.6
#define MAXZ 2.4

using namespace std;
using namespace cv;

//float MINX=-10.0,MAXX=50.0,MINY=-30.0,MAXY=30.0,MINZ=-1.6,MAXZ=2.4;

std::vector<float> normalize_0_255(vector<float> datas)
{
    float dMinValue, dMaxValue;
    int length;
//    dMinValue = *min_element(datas.begin(),datas.end());
//    dMaxValue = *max_element(datas.begin(),datas.end());
    dMinValue = MINZ;
    dMaxValue = MAXZ;
    length = datas.size();
    float ymax = 255; //归一化数据范围
    float ymin = 0;
    vector<float> features;
    float tmp;
    for (int d = 0; d < length; ++d)
    {
        tmp = (ymax-ymin)*(datas[d]-dMinValue)/(dMaxValue-dMinValue+1e-8)+ymin;
        //cout<<"> "<<tmp<<"\n"<<endl;
        features.push_back(tmp);
    }
    return features;
}




//采用C模式读二进制文件
int read_data()
{
    int user_data = 0;
	int arr_counts[1] = {0};
	int counts = 0;

	string velo_filename = "/home/kid/min/Annotations/LiDar/anno2/veloseq/1/VLP160/323.bin";
	ifstream fin(velo_filename, ios::binary);
	if(!fin)
	{
		cout << "读取文件失败" <<endl;
		return 0;
	}
	fin.read((char*)arr_counts,sizeof(int));
	if (arr_counts[0] != 0)
        counts = arr_counts[0];
	cout<<counts<<endl;


	float xx[counts];
    float yy[counts];
    float zz[counts];
    float rr[counts];
    fin.read((char*)xx,counts*sizeof(float));
    fin.read((char*)yy,counts*sizeof(float));
    fin.read((char*)zz,counts*sizeof(float));
    fin.read((char*)rr,counts*sizeof(float));
    fin.close();

//    pcl::PointCloud<pcl::PointXYZ> point_cloud_velo;
//
//    point_cloud_velo.width    = counts;
//    point_cloud_velo.height   = 1;
//    point_cloud_velo.is_dense = true;  //不是稠密型的
//	point_cloud_velo.points.resize(point_cloud_velo.width*point_cloud_velo.height);
//
//    for (int i=0;i < 5182;i++ )
//    {
//        point_cloud_velo.points[i].x = xx[i];
//        point_cloud_velo.points[i].y = yy[i];
//        point_cloud_velo.points[i].z = zz[i];
//
//    }
//
//    for (int i = 0; i < point_cloud_velo.points.size(); ++i)
//    {
//        std::cerr << ">>>>>" << xx[i]<<"<->"<<point_cloud_velo.points[i].x << "," <<yy[i]<<"<->"<< point_cloud_velo.points[i].y << "," << xx[i]<<"<->"<< point_cloud_velo.points[i].z << std::endl;
//    }


    pcl::PointCloud<pcl::PointXYZ>::Ptr point_cloud_velo(new pcl::PointCloud<pcl::PointXYZ>);

    //point_cloud_velo->width    = 5182;
    point_cloud_velo->width    = counts;
    point_cloud_velo->height   = 1;
    point_cloud_velo->is_dense = true;  //不是稠密型的
	point_cloud_velo->points.resize(point_cloud_velo->width*point_cloud_velo->height);

    for (int i=0;i < counts;i++ )
    {
        point_cloud_velo->points[i].x = yy[i];
        point_cloud_velo->points[i].y = xx[i];
        point_cloud_velo->points[i].z = zz[i]-MINZ;
    }

//    for (int i = 0; i < point_cloud_velo->points.size(); ++i)
//    {
//        std::cerr << ">>>>>" << xx[i]<<"<->"<<point_cloud_velo->points[i].x << "," <<yy[i]<<"<->"<< point_cloud_velo->points[i].y << "," << xx[i]<<"<->"<< point_cloud_velo->points[i].z << std::endl;
//    }

    // ********************
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered_x(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered_y(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered_z(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PassThrough<pcl::PointXYZ> pass;

	pass.setInputCloud(point_cloud_velo);
	pass.setFilterFieldName("x");
	pass.setFilterLimits(MINX,MAXX);
	pass.filter(*cloud_filtered_x);

	//pass.setFilterLimitsNegative (true);
	pass.setInputCloud(cloud_filtered_x);
	pass.setFilterFieldName("y");
	pass.setFilterLimits(MINY,MAXY);
	pass.filter(*cloud_filtered_y);

	//pass.setFilterLimitsNegative (true);
	pass.setInputCloud(cloud_filtered_y);
	pass.setFilterFieldName("z");
	pass.setFilterLimits(MINZ,MAXZ);
	//pass.setFilterLimitsNegative (true);
	pass.filter(*cloud_filtered);

	for (int i = 0; i < cloud_filtered->points.size(); ++i)
    {
        std::cerr << ">>>>>" <<i<<"::"<<cloud_filtered->points[i].x << "," << cloud_filtered->points[i].y << "," << cloud_filtered->points[i].z << std::endl;
    }
    cerr<<">>>>>>>>>>"<<cloud_filtered->points.size()<<"\n"<<endl;

	//*********************
    //# 转换为像素位置的值 - 基于分辨率
    int res = 480;
    float xp = (MAXY-MINY) / (float)res;
    float yp = (MAXX-MINX) / (float)res;
    cout<<"xp = "<<xp<<" , yp = "<<yp<<"\n"<<endl;
    int size_cloud_filtered = cloud_filtered->points.size();
    int ximg[size_cloud_filtered] = {0};
    int yimg[size_cloud_filtered] = {0};
    vector<float> z_points(size_cloud_filtered),nol_z_points;

    //vector<float> z_points(cloud_filtered->points.z);


    int px,py;//用来指向图像位置
    for(int j=0;j<size_cloud_filtered;j++)
    {
        ximg[j] = (int)((cloud_filtered->points[j].y - MINY)/ yp);
        yimg[j] = res - (int)((cloud_filtered->points[j].x - MINX)/ xp);
        px = ximg[j];
        py = yimg[j];
        if(px>=res||py>=res||px<0||py<0)
        {
            cout<<">> cloud_filtered->points[j].y="<<cloud_filtered->points[j].y<<", cloud_filtered->points[j].x="<<cloud_filtered->points[j].x<<"\n"<<endl;
            cout<<">> px="<<px<<", py="<<py<<"\n"<<endl;
        }
//        ximg[j] = (int)((cloud_filtered->points[j].y)/ yp);
//        yimg[j] = (int)((cloud_filtered->points[j].x)/ xp);
        z_points.push_back(cloud_filtered->points[j].z);
    }
    nol_z_points = normalize_0_255(z_points);
    Mat img(res, res, CV_8UC1);  //创建一个高200，宽100的灰度图

    for (int i=0;i<size_cloud_filtered;i++)
    {
        px = ximg[i];
        py = yimg[i];
        if(px>=res||py>=res||px<0||py<0)
        {
            cout<<">> px="<<px<<", py="<<py<<"\n"<<endl;
        }else
        {
            img.at<uchar>(py,px) = nol_z_points[i];
        }

    }
    cv::imshow("img",img);
    cv::waitKey(0);
    cv::imwrite("test.png",img);



//    boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer(new pcl::visualization::PCLVisualizer("3D Viewer"));
//	viewer->setBackgroundColor(0, 0, 0);
//	//viewer->addPointCloud<pcl::PointXYZ>(point_cloud_velo, "sample cloud");
//	viewer->addPointCloud<pcl::PointXYZ>(cloud_filtered, "sample cloud");
//	viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "sample cloud");
//	viewer->addCoordinateSystem(1.0);
//	viewer->initCameraParameters();
//
//	while (!viewer->wasStopped())
//	{
//		viewer->spinOnce(100);
//		boost::this_thread::sleep(boost::posix_time::microseconds(100000));
//	}
    return(1);

}

int creat_pcd()
{
//    pcl::PointCloud<pcl::PointXYZ> cloud;
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);

    // 创建点云  并设置适当的参数（width height is_dense）
    cloud->width    = 50000;
    cloud->height   = 1;
    cloud->is_dense = true;  //不是稠密型的
    cloud->points.resize (cloud->width * cloud->height);  //点云总数大小
    //cloud.points.resize (100);  //点云总数大小
    //用随机数的值填充PointCloud点云对象
    //for (size_t i = 0; i < cloud.points.size (); ++i)
    for (size_t i = 0; i < 50000; ++i)
    {
        cloud->points[i].x = 1024 * rand () / (RAND_MAX + 1.0f);
        cloud->points[i].y = 1024 * rand () / (RAND_MAX + 1.0f);
        cloud->points[i].z = 1024 * rand () / (RAND_MAX + 1.0f);
    }
    //把PointCloud对象数据存储在 test_pcd.pcd文件中
    pcl::io::savePCDFileASCII ("test_pcd.pcd", *cloud);

    //打印输出存储的点云数据
    std::cerr << "Saved " << cloud->points.size () << " data points to test_pcd.pcd." << std::endl;

    for (size_t i = 0; i < cloud->points.size (); ++i)
    std::cerr << ">>" << cloud->points[i].x << "," << cloud->points[i].y << "," << cloud->points[i].z << std::endl;

    return (1);
}

int main()
{
//	int sucess = creat_pcd();
    int sucess = read_data();

	return (sucess);
}

