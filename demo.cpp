#include<sys/select.h>
#include<unistd.h>
#include<sys/types.h>
#include<sys/socket.h>
#include<arpa/inet.h>
#include<netinet/in.h>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <pthread.h>
#include <vector>
#include <mutex>
#include <unistd.h>
#define _USE_MATH_DEFINES
#include <math.h>


#include "load_lidar_data.hpp"
#include "cxzn_tracking.hpp"
#include "DTCommon.hpp"


//static mutex lock_solvebyte;
//static mutex lock_info;
//
//static mutex lock_pointclouds;
#define port_in 9090
#define port_out 9090
#define Num_Thread 3

using namespace std;
using namespace cv;

int n_interval = 5;
string  names_file = "/home/kid/min/lidar_dl/data/names.list";
string  cfg_file = "/home/kid/min/detectAndtrack/lidar_3c_tiny_pcl.cfg";
string  weights_file = "/home/kid/min/detectAndtrack/lidar_3c_tiny_pcl_52000.weights";
float const thresh = 0.25;
Detector_YOLO detector(cfg_file, weights_file);
auto obj_names = get_objNames_fromFile(names_file);


Mat mat_img;
Mat show_img;
vector<track_t> tracking;
Rect2d rect_obj;
vector<bbox_t> obj_result_vec;
vector<res_track_t> res_trcking;
vector<res_ouput_t> outputs;
int n_tracks=0;
bool first_frame = true;
int counts=0;


struct Points
{
    double x;
    double y;
    double z;
    double distance;
    double intensity;
    double  verAngle;
    double horAngle;
    int c;
};
typedef char * BYTEARRAY;

typedef  Points Point3D;
double elevationAngle[] = { -16.5, -16, -14.5, -14, -12.5, -12, -10.5, -10,
                                     -8.4, -7.9, -6.4, -5.8, -4.6, -4, -3.5, -2.9,
                                     -2.4, -1.8, -1.26, -0.7, -0.14, 0.43, 0.99, 1.55,
                                     2.7, 3.3, 4.8, 5.4, 6.8, 7.4, 8.9, 9.4
                                 };

struct thread_senddata{
   int  sockfd;     //socket
   char *info;          //发送的数据
   char *addr;         //发送的地址
   int port;             //发送的端口
};

struct thread_recdata{
   int  sockfd;     //socket
   char *info;          //接受的数据
};

pthread_mutex_t lock_solvebyte;        //
pthread_mutex_t lock_info;
pthread_mutex_t lock_pointclouds;
pthread_mutex_t toReceiveLock;
pthread_mutex_t toSolveLock;

vector<BYTEARRAY> point;
int curAngle;
int oldAngle;
vector<BYTEARRAY>  toReceive;
vector<BYTEARRAY>  toSolve;
vector<Point3D> pointCloud;



double convertAngle(int ch)
{
    return ch + 15 * (ch % 2 - 1);
}

double convertAngle_HDL32(int ch)
{
    return elevationAngle[ch];
}

void parse_VLP16(vector<BYTEARRAY > *toSolve)
{
    vector<double> angle;
    vector<double> distance1;
    vector<double> distance2;
    vector<double> intensity1;
    vector<double> intensity2;

    double offsetx = 0, offsety = 0, offsetz = 0;
    double angleyaw = 0, angleheel = 0.0, anglepitch = 0.0; //横摆   侧倾   俯仰

    vector<BYTEARRAY> receiveBA;
//        vector<BYTEARRAY> *receiveBAPtr;

    pthread_mutex_lock(&lock_solvebyte);
    receiveBA = *toSolve;
    toSolve->clear();
    cout<<">>>"<<receiveBA.size()<<endl;;
    pthread_mutex_unlock(&lock_solvebyte);
    //receiveBA = *receiveBAPtr;
//        qDebug()<<"distance1------->"<<distance1;
//        qDebug()<<"distance2------->"<<distance2;

//    angle.clear();
//    distance1.clear();
//    intensity1.clear();
//    distance2.clear();
//    intensity2.clear();

    for(int i = 0;i < receiveBA.size();i++){
        for (int count1 = 0; count1 < 12; count1++)
        {
            //double curA =(int)((((((unsigned int)receiveBA[i][3 + 100 * count1])) *256) + (unsigned int)receiveBA[i][2 + 100 * count1])) * 0.01 + angleyaw;
            //double curA =(int)(((((receiveBA[i][3 + 100 * count1])) *256) + receiveBA[i][2 + 100 * count1])) * 0.01 + angleyaw;
            //double curA =(int)((((((unsigned char)receiveBA[i][3 + 100 * count1])) *256) + (unsigned char)receiveBA[i][2 + 100 * count1])) * 0.01 + angleyaw;

            double curA =(double)((receiveBA[i][3 + 100 * count1] << 8) + receiveBA[i][2 + 100 * count1]) * 0.01 + angleyaw;
           // double curA =(double)(((double)receiveBA[i][3 + 100 * count1] *256) + (double)receiveBA[i][2 + 100 * count1]) * 0.01 + angleyaw;


            if (curA < 0) curA += 360;
            if (curA > 360) curA -= 360;

            angle.push_back(curA);//依次导入角度值，每组一个角度

            for (int count2 = 0; count2 < 16; count2++)
            {
 //               distance1.append((recvBuf[5 + count2 * 3 + 100 * count1] * 256 + recvBuf[4 + count2 * 3 + 100 * count1]) * 2 * 0.001);
 //               intensity1.append(recvBuf[6 + count2 * 3 + 100 * count1]);
 //              distance2.append((recvBuf[5 + (count2 + 16) * 3 + 100 * count1] * 256 + recvBuf[4 + (count2 + 16) * 3 + 100 * count1]) * 2 * 0.001);
 //               intensity2.append(recvBuf[6 + (count2 + 16) * 3 + 100 * count1]);

                double dis1 = (receiveBA[i][5 + count2 * 3 + 100 * count1] * 256 + receiveBA[i][4 + count2 * 3 + 100 * count1])*2 * 0.001;
                double ins1 = receiveBA[i][6 + count2 * 3 + 100 * count1];

                distance1.push_back(dis1);
                intensity1.push_back(ins1);

                //distance1.push_back(((unsigned int)receiveBA[i][5 + count2 * 3 + 100 * count1] * 256 + (unsigned int)receiveBA[i][4 + count2 * 3 + 100 * count1]) * 0.01);
                //intensity1.push_back((unsigned int)receiveBA[i][6 + count2 * 3 + 100 * count1]);
            }

            for (int count2 = 0; count2 < 16; count2++)
            {
                double dis2 = (receiveBA[i][5 + (count2 + 16) * 3 + 100 * count1] * 256 + receiveBA[i][4 + (count2 + 16) * 3 + 100 * count1])*2 * 0.001;
                double ins2 = receiveBA[i][6 + (count2 + 16) * 3 + 100 * count1];

                distance2.push_back(dis2);
                intensity2.push_back(ins2);
            }

        }
    }

    //double xxx, yyy, zzz;

    vector<Point3D> pointCloud_tmp;

    double tmpAngle1 = 0, tmpAngle2 = 0, xxx = 0, yyy = 0, zzz = 0;
    for (int i = 0; i < angle.size(); i++)
    {
        tmpAngle1 = angle[i] * M_PI / 180;

        if (i < (angle.size() - 1))
        {
            if (angle[i] < angle[i + 1])
                tmpAngle2 = (angle[i] + angle[i + 1]) * M_PI / 360;
                //tmpAngle2 = (angle[i] + angle[i + 1]) * M_PI / 180;
            else
                tmpAngle2 = (angle[i] + angle[i + 1] + 360) * M_PI / 360;
                //tmpAngle2 = (angle[i] + angle[i + 1] + 180) * M_PI / 180;
        }
        else
            tmpAngle2 = tmpAngle1;

        for (int j = 0; j < 16; j++)
//        for (int j = 0; j < 32; j++)
        {
            int index = i * 16 + j;
//            int index = i * 32 + j;
            Point3D tempPoint1;

             tempPoint1.distance = distance1[index];
             tempPoint1.intensity = intensity1[index];

             //double verAngle = convertAngle(j) * M_PI / 180;
             //double horAngle = tmpAngle1;
             tempPoint1.verAngle = convertAngle(j) * M_PI / 180;
             tempPoint1.horAngle = tmpAngle1;


           // if(verAngle<0)
            {
                tempPoint1.z = tempPoint1.distance * sin(tempPoint1.verAngle);
                tempPoint1.y = tempPoint1.distance * cos(tempPoint1.verAngle) * cos(tempPoint1.horAngle);
                tempPoint1.x = tempPoint1.distance * cos(tempPoint1.verAngle) * sin(tempPoint1.horAngle);


                xxx = cos(angleheel) * tempPoint1.x - sin(angleheel) * tempPoint1.z;
                zzz = sin(angleheel) * tempPoint1.x + cos(angleheel) * tempPoint1.z;

                tempPoint1.x = xxx; tempPoint1.z = zzz;

                yyy = cos(anglepitch) * tempPoint1.y - sin(anglepitch) * tempPoint1.z;
                zzz = sin(anglepitch) * tempPoint1.y + cos(anglepitch) * tempPoint1.z;

                tempPoint1.y = yyy; tempPoint1.z = zzz;

                tempPoint1.x += offsetx; tempPoint1.y += offsety; tempPoint1.z += offsetz;
                //tempPoint->id = id++;

                if (tempPoint1.distance > 0.2  && tempPoint1.z > -1.8)
                    //if (!(tempPoint.x > -1.3 && tempPoint.x < +1.3 && tempPoint.y < 0 && tempPoint.y > -4))
                    //pointCloud_tmp.push_back(tempPoint1);
                    ;
            }

            Point3D tempPoint2;
            tempPoint2.distance = distance2[index];
            tempPoint2.intensity = intensity2[index];
            tempPoint2.verAngle = convertAngle(j) * M_PI / 180;
            tempPoint2.horAngle = tmpAngle2;


            {
                tempPoint2.z = tempPoint2.distance * sin(tempPoint2.verAngle);
                tempPoint2.y = tempPoint2.distance * cos(tempPoint2.verAngle) * cos(tempPoint2.horAngle);
                tempPoint2.x = tempPoint2.distance * cos(tempPoint2.verAngle) * sin(tempPoint2.horAngle);

                xxx = cos(angleheel) * tempPoint2.x - sin(angleheel) * tempPoint2.z;
                zzz = sin(angleheel) * tempPoint2.x + cos(angleheel) * tempPoint2.z;

                tempPoint2.x = xxx; tempPoint2.z = zzz;

                yyy = cos(anglepitch) * tempPoint2.y - sin(anglepitch) * tempPoint2.z;
                zzz = sin(anglepitch) * tempPoint2.y + cos(anglepitch) * tempPoint2.z;

                tempPoint2.y = yyy; tempPoint2.z = zzz;

                tempPoint2.x += offsetx; tempPoint2.y += offsety; tempPoint2.z += offsetz;
            }

            //tempPoint->id = id++;

            if (tempPoint2.distance > 0.2 && tempPoint2.z > -1.8)
                //if (!(tempPoint.x > -1.3 && tempPoint.x < +1.3 && tempPoint.y < 0 && tempPoint.y > -4))
                pointCloud_tmp.push_back(tempPoint2);

        }
    }

    receiveBA.clear();
    pthread_mutex_lock(&lock_pointclouds);

    cout<<pointCloud_tmp.size() << std::endl;


    pointCloud = pointCloud_tmp;
    pthread_mutex_unlock(&lock_pointclouds);
}

void process()
{
    counts++;
    if (mat_img.rows != 0 && mat_img.cols != 0)
    {
        show_img = mat_img.clone();
//        cv::imshow("img",show_img);
//        cv::waitKey(0);

        if (first_frame)
        {
            obj_result_vec = detector.detect(mat_img);
            //draw_detect_boxes(show_img, obj_result_vec, obj_names);
            n_tracks = obj_result_vec.size();
            if (n_tracks > 1)
            {
                tracking = init_tracking(obj_result_vec, mat_img,n_tracks);
                res_trcking = init_res_tracking(obj_result_vec,n_tracks);

                first_frame = false;
            }else{
                cout<<"can not detect obj"<<endl;
            }

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
            if (counts%n_interval == 0)
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
        if(res_trcking.size()>0)
        {
            draw_tracking_res(show_img,res_trcking,Scalar(0, 255, 0),"track");
            outputs = convert_resTrack(res_trcking);
            print_output(outputs);
            outputs.clear();
            obj_result_vec.clear();
        }


        //draw_detect_boxes(show_img, obj_result_vec, obj_names);

        imshow("res", show_img);
        waitKey(1);
    }

}

int initudp(void)
{
    int sockfd;
    /* Create Socket*/
    sockfd = socket(AF_INET, SOCK_DGRAM, 0);
    if(-1==sockfd){
        return 0;
        puts("Failed to create socket");
    }
    /*Config Socket Addr*/
    struct sockaddr_in addr;
    socklen_t          addr_len=sizeof(addr);
    memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;       // Use IPV4
    addr.sin_port   = htons(port_in);    //
    addr.sin_addr.s_addr = inet_addr("192.168.1.125");
    /* Time out*/
//    struct timeval tv;
//    tv.tv_sec  = 0;
//    tv.tv_usec = 200000;  // 200 ms
//    setsockopt(sockfd, SOL_SOCKET, SO_RCVTIMEO, (const char*)&tv, sizeof(struct timeval));
    /* Bind Socket*/
    if (bind(sockfd, (struct sockaddr*)&addr, addr_len) == -1){      //收数据才需要bind
        printf("Failed to bind socket on port %d\n", port_in);
        close(sockfd);
        return false;
    }
    return sockfd;
}

void sendinfo(thread_senddata *thread_info)     //发送数据调用
{
    int len;
    struct sockaddr_in dest;
    socklen_t dest_len = sizeof(dest);
    memset(&dest, 0, sizeof(dest));
    dest.sin_family = AF_INET;
    dest.sin_port   = htons(thread_info->port);
    dest.sin_addr.s_addr = inet_addr(thread_info->addr);


    if(strlen(thread_info->info)!=0)               //如果数组有数则发送
    {
        len = strlen(thread_info->info);
        sendto(thread_info->sockfd, thread_info->info, len, 0, (sockaddr*)&dest, dest_len);
    }
}

void *recinfo(void *rec_data)     //接受线程调用的函数
{
    struct thread_recdata *thread_info;
    thread_info = (struct thread_recdata *)rec_data;
    struct sockaddr_in src;
    socklen_t src_len = sizeof(src);
    memset(&src, 0, sizeof(src));

    while(1)
    {
        int sz = recvfrom(thread_info->sockfd, thread_info->info, 1206, 0, (sockaddr*)&src, &src_len);
        if (sz > 0)
        {
            thread_info->info[sz] = 0;

            if(sz)
            {
                pthread_mutex_lock(&toReceiveLock);
                char *a = (char*)malloc(sizeof(char)*1206);
                memcpy(a,thread_info->info,1206);
                //这里不能用strcpy
                //因为数据可能是0x00
                //strcpy复制到00就结束了
                //后面的就不复制了

                toReceive.push_back(a);
                for (int cou = 0; cou < 12; cou++)
                {
                    //int curAngle = (thread_info->info[3 + 100 * cou] << 8) + thread_info->info[2 + 100 * cou];
                    int curAngle = (a[3 + 100 * cou] << 8) + a[2 + 100 * cou];
                    if (oldAngle < 18000&& curAngle >= 18000)
                    {
                        pthread_mutex_lock(&toSolveLock);
                        toSolve.clear();
                        toSolve = toReceive;
                        toReceive.clear();
                        pthread_mutex_unlock(&toSolveLock);
                        break;
                    }
                    oldAngle = curAngle;
                }
                pthread_mutex_unlock(&toReceiveLock);
//                oldAngle = (thread_info->info[1103] << 8) + thread_info->info[1102];
                oldAngle = (a[1103] << 8) + a[1102];
            }
            //std::cout<<"size :"<<toSolve.size()<<endl;

        }
    }
}

void *get_info_rec(void *arg)    //显示线程调用的函数
{
    while(1)
    {
        //cout<<"point size:"<<point.size()<<endl;
        if (toSolve.size() > 70)
        {
            //pthread_mutex_lock(&rec_mutex);
            //cout<<"before:"<<toSolve.size()<<endl;
            parse_VLP16(&toSolve);

            //cout<<"after:"<<toSolve.size()<<endl;
            //cout<<"get point size:"<<pointCloud.size()<<endl;
            vector<Point3D> temp_p = pointCloud;
            pointCloud.clear();
            if(temp_p.size()>3000)
            {
                pcl::PointCloud<pcl::PointXYZI>::Ptr point_cloud_velo(new pcl::PointCloud<pcl::PointXYZI>);
                point_cloud_velo->width    = temp_p.size();
                point_cloud_velo->height   = 1;
                point_cloud_velo->is_dense = false;  //不是稠密型的
                point_cloud_velo->points.resize(point_cloud_velo->width*point_cloud_velo->height);
                for (int i=0;i < temp_p.size();i++ )
                {
                    // 这里没有points吧
//                    pcl::PointXYZI p;
//                    p.x = temp_p[i].x;
//                    p.y = temp_p[i].y;
//                    p.z = temp_p[i].z;
//                    p.intensity = temp_p[i].intensity;
//                    point_cloud_velo->points.push_back(p);
                    point_cloud_velo->points[i].x = temp_p[i].x;
                    point_cloud_velo->points[i].y = temp_p[i].y;
                    point_cloud_velo->points[i].z = temp_p[i].z;
                    point_cloud_velo->points[i].intensity = pointCloud[i].intensity;
                }
//                point_cloud_velo->width = temp_p.size();
//                point_cloud_velo->height = 1;
                //pointCloud.clear();
                passthrough_filter(point_cloud_velo,false);
                get_img(mat_img,point_cloud_velo);
//                cv::imshow("img",mat_img);
//                cv::waitKey(1);
                process();
            }


//            for(int j=0;j<pointCloud.size();j++)
//            {
//                std::cout<<"pointCloud:"<<pointCloud[j].x<<std::endl;
//            }
            //pthread_mutex_unlock(&rec_mutex);
        }
        else
        {
            usleep(1000);
            continue;
        }
    }
}




int main(void)
{
    //char info[1206];    //发送的
    char addr[20];    //发送目标地址
    thread_senddata thread_send_info;    //发送信息函数利用的信息结构体
    char buffer[1206];   //接受到的数据，要不要其实无所谓了，后面再修改吧
    int sockfd;
    thread_recdata thread_rec;
    //memset(info_rec,0,10*1206);

    /*线程所需变量*/
    pthread_t  thread[Num_Thread];     //存储线程的id 3个线程
    pthread_attr_t attr;       //线程属性设置
    int rc;      //创建线程的返回值，检查是否创建成功
    void *status;


    sockfd = initudp();/*创建UDP*/
    if (sockfd==0)
    {
        printf("socket error!");
    }

    /*设置接受线程参数*/
    thread_rec.sockfd = sockfd;
    thread_rec.info = buffer;

    /*创建线程*/
    pthread_attr_init(&attr);
    pthread_attr_setdetachstate(&attr,PTHREAD_CREATE_JOINABLE);

    rc = pthread_create(&thread[0], &attr ,recinfo,(void *)&thread_rec);   //收数据
    if (rc){
      printf("Error:unable to create thread");
     exit(-1);
    }
    rc = pthread_create(&thread[1], &attr ,get_info_rec,NULL);   //显示数据
    if (rc){
      printf("Error:unable to create thread");
     exit(-1);
    }



    //删除属性，并等待其他线程
    pthread_attr_destroy(&attr);
    for(int i = 0 ;i < Num_Thread;i++)
    {
          rc = pthread_join(thread[i], &status);
          if (rc){
              printf("Error:unable to join");
             exit(-1);
          }
    }
    pthread_exit(NULL);   //删除线程
    return 1;
}
