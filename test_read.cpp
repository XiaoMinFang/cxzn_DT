//#include "stdafx.h"
#include<iostream>
#include<fstream>
#include<string>
#include<cmath>

#include <opencv2/opencv.hpp>            // C++

#include"load_data.hpp"

using namespace std;
using namespace cv;

//float MINX=-10.0,MAXX=50.0,MINY=-30.0,MAXY=30.0,MINZ=-1.6,MAXZ=2.4;

//std::vector<float> normalize_0_255(vector<float> datas,float dMinValue,float dMaxValue)
//{
//    int length;
////    dMinValue = *min_element(datas.begin(),datas.end());
////    dMaxValue = *max_element(datas.begin(),datas.end());
//    length = datas.size();
//    float ymax = 255; //归一化数据范围
//    float ymin = 0;
//    vector<float> features;
//    float tmp;
//    for (int d = 0; d < length; ++d)
//    {
//        tmp = (ymax-ymin)*(datas[d]-dMinValue)/(dMaxValue-dMinValue+1e-8)+ymin;
//        //cout<<"> "<<tmp<<"\n"<<endl;
//        features.push_back(tmp);
//    }
//    return features;
//}
////数组拼接
//float *SortArry(float *StrA,int lenA, float *StrB, int lenB)
//{
//    if (StrA == NULL || StrB == NULL)
//        return NULL;
//
//    float *StrC = new float[lenA + lenB+1];
//    int i, j, k;
//    i = j = k = 0;
////    while (i < lenA && j < lenB)
////    {
////        if (StrA[i] < StrB[j]) StrC[k++] = StrA[i++];
////        else StrC[k++] = StrB[j++];
////    }
//
//    while (i<lenA)
//    {
//        StrC[k++] = StrA[i++];
//    }
//
//    while (j<lenB)
//    {
//        StrC[k++] = StrB[j++];
//    }
//
//    return StrC;
//}
//
//
//
////采用C模式读二进制文件
//int read_data()
//{
//    int user_data = 0;
//	int arr_counts[1] = {0};
//	int counts160 = 0, counts161 = 0, counts = 0;
//
//
//
//
//	string velo_filename160 = "/home/kid/min/Annotations/LiDar/anno2/veloseq/1/VLP160/324.bin";
//	string velo_filename161 = "/home/kid/min/Annotations/LiDar/anno2/veloseq/1/VLP161/324.bin";
//	ifstream fin160(velo_filename160, ios::binary);
//	ifstream fin161(velo_filename161, ios::binary);
//	if(!fin160 && !fin161)
//	{
//		cout << "读取文件失败" <<endl;
//		return 0;
//	}
//	fin160.read((char*)arr_counts,sizeof(int));
//	if (arr_counts[0] != 0)
//        counts160 = arr_counts[0];
//	cout<<"points counts160 = "<<counts160<<endl;
//	fin161.read((char*)arr_counts,sizeof(int));
//	if (arr_counts[0] != 0)
//        counts161 = arr_counts[0];
//	cout<<"points counts161 = "<<counts161<<endl;
//
//    counts = counts160+counts161;
//
//	float xx160[counts160],xx161[counts161];
//    float yy160[counts160],yy161[counts161];
//    float zz160[counts160],zz161[counts161];
//    vector<int> rr_tmp;
//    char rr_tmp_char;
//    int rr_tmp_int;
//    int rr[counts];
//    fin160.read((char*)xx160,counts160*sizeof(float));
//    fin160.read((char*)yy160,counts160*sizeof(float));
//    fin160.read((char*)zz160,counts160*sizeof(float));
//    fin161.read((char*)xx161,counts161*sizeof(float));
//    fin161.read((char*)yy161,counts161*sizeof(float));
//    fin161.read((char*)zz161,counts161*sizeof(float));
//    for (int i=0;i<counts160;i++)
//    {
//        fin160.read((char*)&rr_tmp_char,sizeof(char));
//        rr_tmp_int = rr_tmp_char*1;
//        rr[i] = rr_tmp_int;
//    }
//    for (int i=0;i<counts161;i++)
//    {
//        fin161.read((char*)&rr_tmp_char,sizeof(char));
//        rr_tmp_int = rr_tmp_char*1;
//        rr[i+counts160] = rr_tmp_int;
//    }
//    fin160.close();fin161.close();
//    float *xx = SortArry(xx160,counts160,xx161,counts161);
//    float *yy = SortArry(yy160,counts160,yy161,counts161);
//    float *zz = SortArry(zz160,counts160,zz161,counts161);
//
//
//    pcl::PointCloud<pcl::PointXYZI>::Ptr point_cloud_velo(new pcl::PointCloud<pcl::PointXYZI>);
//
//    //point_cloud_velo->width    = 5182;
//    point_cloud_velo->width    = counts;
//    point_cloud_velo->height   = 1;
//    point_cloud_velo->is_dense = false;  //不是稠密型的
//	point_cloud_velo->points.resize(point_cloud_velo->width*point_cloud_velo->height);
//
//    for (int i=0;i < counts;i++ )
//    {
//        point_cloud_velo->points[i].x = yy[i];
//        point_cloud_velo->points[i].y = xx[i];
//        point_cloud_velo->points[i].z = zz[i]-MINZ;
//        point_cloud_velo->points[i].intensity = rr[i];
//    }
//    delete[] xx;
//    delete[] yy;
//    delete[] zz;
//
////    for (int i = 0; i < point_cloud_velo->points.size(); ++i)
////    {
////        std::cerr << ">>>>>" << xx[i]<<"<->"<<point_cloud_velo->points[i].x << "," <<yy[i]<<"<->"<< point_cloud_velo->points[i].y << "," << xx[i]<<"<->"<< point_cloud_velo->points[i].z << std::endl;
////    }
////    for (int i = 0; i < point_cloud_velo->points.size(); ++i)
////    {
////        std::cerr << ">>>>>" << rr[i]<<"<->"<<point_cloud_velo->points[i].intensity<<std::endl;
////    }
//
//    // ********************
//    pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_filtered(new pcl::PointCloud<pcl::PointXYZI>);
//    pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_filtered_x(new pcl::PointCloud<pcl::PointXYZI>);
//    pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_filtered_y(new pcl::PointCloud<pcl::PointXYZI>);
//    //pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_filtered_z(new pcl::PointCloud<pcl::PointXYZI>);
//    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_d(new pcl::PointCloud<pcl::PointXYZ>);
//    pcl::PassThrough<pcl::PointXYZI> pass;
//
//	pass.setInputCloud(point_cloud_velo);
//	pass.setFilterFieldName("x");
//	pass.setFilterLimits(MINX,MAXX);
//	pass.filter(*cloud_filtered_x);
//
//	//pass.setFilterLimitsNegative (true);
//	pass.setInputCloud(cloud_filtered_x);
//	pass.setFilterFieldName("y");
//	pass.setFilterLimits(MINY,MAXY);
////	pass.filter(*cloud_filtered_y);
//
//	//pass.setFilterLimitsNegative (true);
////	pass.setInputCloud(cloud_filtered_y);
////	pass.setFilterFieldName("z");
////	pass.setFilterLimits(MINZ,MAXZ);
//	//pass.setFilterLimitsNegative (true);
//	pass.filter(*cloud_filtered);
//
////	for (int i = 0; i < cloud_filtered->points.size(); ++i)
////    {
////        std::cerr << ">>>>>" <<i<<"::"<<cloud_filtered->points[i].x << "," << cloud_filtered->points[i].y << "," << cloud_filtered->points[i].z << std::endl;
////    }
////    cerr<<">>>>>>>>>>"<<cloud_filtered->points.size()<<"\n"<<endl;
//
//
////    pcl::copyPointCloud(*cloud_d, *cloud_filtered);
//    cout<<"remain points = "<<cloud_filtered->points.size()<<endl;
//
//    vector<float>densityMap,nol_densityMap;
//    int M = cloud_filtered->points.size();
//    for (int i = 0;i <M;i++)
//    {
//        pcl::PointXYZ p;
//        p.x = cloud_filtered->points[i].x*100.0;
//        p.y = cloud_filtered->points[i].y*100.0;
//        p.z = cloud_filtered->points[i].z*100.0;
//        cloud_d->points.push_back(p);
//    }
//    cloud_d->width = 1;
//    cloud_d->height = M;
//    //cout<<">>>"<<cloud_d->points.size()<<endl;
//    pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;  //创建一个快速k近邻查询,查询的时候若该点在点云中，则第一个近邻点是其本身
//    kdtree.setInputCloud(cloud_d);
//    float everagedistance =0;
//    for (int i =0; i < cloud_d->size();i++)
//    {
//        //cout<<">>>>> i = "<<i<<endl;
//        vector<int> nnh ;
//        vector<float> squaredistance;
//        pcl::PointXYZ p;
//        p = cloud_d->points[i];
//        kdtree.radiusSearch(p,1,nnh,squaredistance);
//        int tmp = nnh.size() ;
//        //if (tmp > 1) cout<<"i = "<<i<<", density = "<<tmp<<"\n"<<endl;
//
//        if (tmp > 1)
//        {
//            float tmp_ = min(1.0,log((float)tmp)/log(64));
//            densityMap.push_back(tmp_);
//        }
//        else densityMap.push_back(0.0);
//    }
//
//    //everagedistance = everagedistance/(cloud->size()/2);
//
//
//	//*********************
//    //# 转换为像素位置的值 - 基于分辨率
//    int res = 480;
//    float xp = (MAXY-MINY) / (float)res;
//    float yp = (MAXX-MINX) / (float)res;
//    cout<<"xp = "<<xp<<" , yp = "<<yp<<"\n"<<endl;
//    int size_cloud_filtered = cloud_filtered->points.size();
//    int ximg[size_cloud_filtered] = {0};
//    int yimg[size_cloud_filtered] = {0};
//    vector<float> z_points(size_cloud_filtered),nol_z_points;
//    vector<float> intensityMap(size_cloud_filtered),nol_intensityMap;
//    //vector<float> z_points(cloud_filtered->points.z);
//
//
//    int px,py;//用来指向图像位置
//    for(int j=0;j<size_cloud_filtered;j++)
//    {
//        ximg[j] = (int)((cloud_filtered->points[j].y - MINY)/ yp);
//        yimg[j] = res-1-(int)((cloud_filtered->points[j].x - MINX)/ xp);
//        px = ximg[j];
//        py = yimg[j];
//        if(px>=res||py>=res||px<0||py<0)
//        {
//            cout<<">> cloud_filtered->points[j].y="<<cloud_filtered->points[j].y<<", cloud_filtered->points[j].x="<<cloud_filtered->points[j].x<<"\n"<<endl;
//            cout<<">> px="<<px<<", py="<<py<<"\n"<<endl;
//        }
////        ximg[j] = (int)((cloud_filtered->points[j].y)/ yp);
////        yimg[j] = (int)((cloud_filtered->points[j].x)/ xp);
//        z_points.push_back(cloud_filtered->points[j].z);
//        intensityMap.push_back(cloud_filtered->points[j].intensity);
//    }
//    nol_z_points = normalize_0_255(z_points,MINZ,MAXZ);
//    float d_min = *min_element(intensityMap.begin(), intensityMap.end());
//    float d_max = *max_element(intensityMap.begin(), intensityMap.end());
//    nol_intensityMap = normalize_0_255(intensityMap,0,d_max);
//    d_min = *min_element(densityMap.begin(), densityMap.end());
//    d_max = *max_element(densityMap.begin(), densityMap.end());
//    nol_densityMap = normalize_0_255(densityMap,0,d_max);
//    //float density_max = *max_element(densityMap.begin(), densityMap.end());
//    Mat img(res, res, CV_8UC3,Scalar(0,0,0));  //
//
//    for (int i=0;i<size_cloud_filtered;i++)
//    {
//        px = ximg[i];
//        py = yimg[i];
//        if(px>=res||py>=res||px<0||py<0)
//        {
//            cout<<">> px="<<px<<", py="<<py<<"\n"<<endl;
//        }else //BGR
//        {
//            img.at<Vec3b>(py,px)[0] = nol_densityMap[i];
//            //img.at<Vec3b>(py,px)[0] = 0;
//            img.at<Vec3b>(py,px)[1] = nol_z_points[i];
//            img.at<Vec3b>(py,px)[2] = nol_intensityMap[i];
//            //img.at<Vec3b>(py,px)[2] = 0;
//            //img.at<uchar>(py,px) = nol_z_points[i];
//        }
//
//    }
//    cv::imshow("img",img);
//    cv::waitKey(0);
//    cv::imwrite("test.png",img);
//
//
//
////    boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer(new pcl::visualization::PCLVisualizer("3D Viewer"));
////	viewer->setBackgroundColor(0, 0, 0);
////	viewer->addPointCloud<pcl::PointXYZI>(point_cloud_velo, "sample cloud");
////	//viewer->addPointCloud<pcl::PointXYZ>(cloud_filtered, "sample cloud");
////	viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "sample cloud");
////	viewer->addCoordinateSystem(1.0);
////	viewer->initCameraParameters();
////
////	while (!viewer->wasStopped())
////	{
////		viewer->spinOnce(100);
////		boost::this_thread::sleep(boost::posix_time::microseconds(100000));
////	}
//
//    return(1);
//
//}

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

string&   replace_all(string&   str,const   string&   old_value,const   string&   new_value)
{
    while(true)   {
        string::size_type   pos(0);
        if(   (pos=str.find(old_value))!=string::npos   )
            str.replace(pos,old_value.length(),new_value);
        else   break;
    }
    return   str;
}


int test_read_main()
{
    //
//    float array1[] = { 1, 4, 6, 7, 9, 12 };
//	float array2[] = { 2, 3, 4, 5, 6, 7, 8, 10 };
//	int lenA = sizeof(array1) / sizeof(int);
//    int lenB = sizeof(array2) / sizeof(int);
//    float *ret = SortArry(array1, lenA, array2, lenB);
//    for (int i = 0; i < (lenA + lenB); i++)
//        cout << ret[i] <<endl;
//    delete[] ret; //清空堆内存
//
//

//	int sucess = creat_pcd();
    Mat img;
    ifstream fin160("/home/kid/min/Annotations/LiDar/anno3/vlp160.txt", ios::in);
    ifstream fin161("/home/kid/min/Annotations/LiDar/anno3/vlp161.txt", ios::in);
    string velo_path_160;
	string velo_path_161;
	int counts=0;

	string save_path_tmp1,save_path_tmp2,save_path;


	//while(!fin160.eof() && !fin161.eof())
    for(int i = 0;i<3;i++)
    {
        getline(fin160,velo_path_160);
        cout<<velo_path_160<<endl;
        getline(fin161,velo_path_161);
        cout<<velo_path_161<<endl;

        load_data(img,velo_path_160,velo_path_161);
        cv::imshow("img",img);
        cv::waitKey(0);

//        save_path_tmp1 = replace_all(velo_path_161,"veloseq","img_cpp_1c");
//        save_path_tmp2 = replace_all(save_path_tmp1,"VLP161/","");
//        save_path = replace_all(save_path_tmp2,".bin",".png");
//
//        cout<< save_path<<endl;
//        imwrite(save_path,img);
        counts++;
    }
    fin160.close()


//	string save_path = "/home/kid/min/Annotations/LiDar/anno3/";
//
//	int s_id = 97;int e_id = 603;
//
//	Mat mat_img;
//	VideoWriter writer;
//	string bin_s = ".bin";
//    string velo_160 = velo_path_160+to_string(s_id) + bin_s;
//    string velo_161 = velo_path_161+to_string(s_id) + bin_s;
//    cout<< velo_160<<endl;
////    load_data(img);
////    cv::imshow("img",img);
////    cv::waitKey(0);
//    cv::imwrite("test.png",img);
	return (1);
}

