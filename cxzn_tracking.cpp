#include "cxzn_tracking.hpp"

using namespace std;
using namespace cv;


Rect2d bboxt_2_rect(bbox_t b)
{
    Rect2d t;
    t.x = b.x;
    t.y = b.y;
    t.width = b.w;
    t.height = b.h;
    return t;
}
Rect2d zero_rect()
{
    Rect2d t;
    t.x = 0;
    t.y= 0;
    t.width = 0;
    t.height = 0;
    return t;
}

float overlap(float x1, float w1, float x2, float w2)
{
    float l1 = x1 - w1/2;
    float l2 = x2 - w2/2;
    float left = l1 > l2 ? l1 : l2;
    float r1 = x1 + w1/2;
    float r2 = x2 + w2/2;
    float right = r1 < r2 ? r1 : r2;
    return right - left;
}

float box_intersection(Rect2d a, Rect2d b)
{
    float w = overlap(a.x, a.width, b.x, b.width);
    float h = overlap(a.y, a.height, b.y, b.height);
    if(w < 0 || h < 0) return 0;
    float area = w*h;
    return area;
}

float box_union(Rect2d a, Rect2d b)
{
    float i = box_intersection(a, b);
    float u = a.width*a.height + b.width*b.height - i;
    return u;
}

float box_iou(Rect2d a, Rect2d b)
{
    return box_intersection(a, b)/box_union(a, b);
}

Rect2d find_max_iou(vector<bbox_t> obj_result_vec,Rect2d pre_rect)
{
    float max_iou = -1;
    int max_iou_obj_id = 0;
    for (int i = 0;i<obj_result_vec.size();i++)
    {
        Rect2d obj_rect;
        obj_rect.x = obj_result_vec[i].x;
        obj_rect.y = obj_result_vec[i].y;
        obj_rect.width = obj_result_vec[i].w;
        obj_rect.height = obj_result_vec[i].h;
        float iou = box_iou(obj_rect,pre_rect);
        if (iou>max_iou)
        {
            max_iou = iou;
            max_iou_obj_id = i;
        }
    }
    Rect2d  cur_rect;
    if(obj_result_vec.size()>0)
    {
        cur_rect.x = obj_result_vec[0].x;
        cur_rect.y = obj_result_vec[0].y;
        cur_rect.width = obj_result_vec[0].w;
        cur_rect.height = obj_result_vec[0].h;
    }else{
        cur_rect.x = 0;
        cur_rect.y = 0;
        cur_rect.width = 0;
        cur_rect.height = 0;
    }

    return cur_rect;
}
bool is_del_rect(vector<bbox_t> obj_result_vec,Rect2d rect)
{//跟踪框与所有的检测框都不重叠，说明跟踪框丢了，则删除该跟踪框
    bool is_del = false;
    float max_iou = -1;
    int max_iou_obj_id = 0;
    for (int i = 0;i<obj_result_vec.size();i++)
    {
        Rect2d obj_rect;
        obj_rect.x = obj_result_vec[i].x;
        obj_rect.y = obj_result_vec[i].y;
        obj_rect.width = obj_result_vec[i].w;
        obj_rect.height = obj_result_vec[i].h;
        float iou = box_iou(obj_rect,rect);
        if (iou>max_iou)
        {
            max_iou = iou;
            max_iou_obj_id = i;
        }
    }
    if (max_iou < 0.5)
    {
        is_del = true;
    }
    return is_del;
}


bool is_add_rect(vector<res_track_t> res_track, bbox_t obj_result)
{//若某检测框与所有的跟踪框的iou<0.5,就添加一个跟踪框；
    float max_iou = -1;
    int max_iou_obj_id = 0;
    bool is_add = false;
    for(int i=0;i<MAX_TRACKING_NUM;i++)
    {
        if(res_track[i].is_acti)
        {
            Rect2d rect_track;
            rect_track = res_track[i].cur_rect;
            Rect2d rect_obj;
            rect_obj.x = obj_result.x;
            rect_obj.y = obj_result.y;
            rect_obj.width = obj_result.w;
            rect_obj.height = obj_result.h;

            float iou = box_iou(rect_obj,rect_track);
            if (iou>max_iou)
            {
                max_iou = iou;
                max_iou_obj_id = i;
            }
        }

    }
    if(max_iou < 0.5)
    {
        is_add = true;
    }
    return is_add;
}

int find_unusing_id(vector<track_t> tracking)
{//找没有用的tracking,加上
    for(int i=0;i<MAX_TRACKING_NUM;i++)
    {
        if(!tracking[i].is_acti)
            return i;
    }
    return -1;

}


track_speed_t calc_speed(Rect2d a, Rect2d b)
{
    //a当前的，b以前的
    //速度  的方向问题？？？
    track_speed_t v;

    if (a.width == 0 || a.height == 0||b.width == 0 && b.height == 0)
    {
        v.vx = 0;
        v.vy = 0;
        return v;
    }

    int x_c1 = (float)a.width*0.5 + a.x;
    int y_c1 = (float)a.height*0.5 + a.y;

    int x_c2 = (float)b.width*0.5+b.x;
    int y_c2 = (float)b.height*0.5+b.y;

    v.vx = (float)(x_c1 - x_c2)*PIXEL2M/FRAME_TIME_INTERVAL;
    v.vy = (float)(y_c1 - y_c2)*PIXEL2M/FRAME_TIME_INTERVAL;
    return v;


}

float calc_direction(Rect2d obj)
{
    if(obj.width == 0 || obj.height == 0)
        return 0;

    int cx = obj.x + 0.5*obj.width;
    int cy = obj.y + 0.5*obj.height;
    float x,y;

    y = SRC_POINT_Y-cy;
    x = cx - SRC_POINT_X;

    float di = atan2(-x,y);//atan2(y,x)以正右法线为原点方向

    return di;
}




//初始化tracking
//void init_tracking(vector<track_t> &tracking,vector<bbox_t> obj_rect,Mat mat_img)
//{
//	Rect2d rect_obj;
//	track_t tracking_;
//
//	//tracking(MAX_TRACKING_NUM);
//
//	int n = obj_rect.size();
//	if (n < MAX_TRACKING_NUM)
//	{
//
//		for (int i = 0; i < n; i++)
//		{
//			rect_obj.height = obj_rect[i].h; rect_obj.width = obj_rect[i].w;
//			rect_obj.x = obj_rect[i].x; rect_obj.y = obj_rect[i].y;
//
//			tracking_.tracker = TrackerMOSSE::create();
//			tracking_.tracker->init(mat_img, rect_obj);
//			tracking_.init_rects = rect_obj;
//			tracking_.id = i;
//			tracking_.is_acti = true;
//			tracking.push_back(tracking_);
//
//		}
//	}
//	else
//	{
//		for (int i = 0; i < MAX_TRACKING_NUM; i++)
//		{
//			rect_obj.height = obj_rect[i].h; rect_obj.width = obj_rect[i].w;
//			rect_obj.x = obj_rect[i].x; rect_obj.y = obj_rect[i].y;
//			tracking_.tracker = TrackerMOSSE::create();
//			tracking_.tracker->init(mat_img, rect_obj);
//			tracking_.init_rects = rect_obj;
//			tracking_.id = i;
//			tracking_.is_acti = true;
//			tracking.push_back(tracking_);
//		}
//	}
//
//
//
//	//return tracking;
//}
//
//void init_res_tracking(vector<res_track_t> &res_tracking,vector<bbox_t> obj_rect)
//{
//	//res_tracking(MAX_TRACKING_NUM);
//	Rect2d rect_obj;
//	res_track_t res_tracking_;
//
//	int n = obj_rect.size();
//	if (n < MAX_TRACKING_NUM)
//	{
//
//		for (int i = 0; i < n; i++)
//		{
//			rect_obj.height = obj_rect[i].h; rect_obj.width = obj_rect[i].w;
//			rect_obj.x = obj_rect[i].x; rect_obj.y = obj_rect[i].y;
//
//
//			res_tracking_.l_tracking = 0;
//			res_tracking_.id = i;
//			res_tracking_.res_track = rect_obj;
//			res_tracking.push_back(res_tracking_);
//
//		}
//	}
//	else
//	{
//		for (int i = 0; i < MAX_TRACKING_NUM; i++)
//		{
//			rect_obj.height = obj_rect[i].h; rect_obj.width = obj_rect[i].w;
//			rect_obj.x = obj_rect[i].x; rect_obj.y = obj_rect[i].y;
//
//			res_tracking_.l_tracking = 0;
//			res_tracking_.id = i;
//			res_tracking_.res_track = rect_obj;
//			res_tracking.push_back(res_tracking_);
//		}
//	}
//	//return res_tracking;
//}
//

vector<track_t> init_tracking(vector<bbox_t> obj_rect,Mat mat_img,int n_tracks)
{
	Rect2d rect_obj;


	vector<track_t> tracking(MAX_TRACKING_NUM);

	//int n = obj_rect.size();
	if (n_tracks < MAX_TRACKING_NUM)
	{
		for (int i = 0; i < n_tracks; i++)
		{
			rect_obj.height = obj_rect[i].h; rect_obj.width = obj_rect[i].w;
			rect_obj.x = obj_rect[i].x; rect_obj.y = obj_rect[i].y;

			tracking[i].tracker = TrackerMOSSE::create();
			tracking[i].tracker->init(mat_img, rect_obj);
			tracking[i].init_rects = rect_obj;
			tracking[i].id = i;
			tracking[i].is_acti = true;
		}
		for (int j = n_tracks; j < MAX_TRACKING_NUM; j++)
		{
			//tracking[i].tracker = TrackerMOSSE::create();
			//tracking[i].tracker->init(mat_img, rect_obj);
			//tracking[i].init_rects = rect_obj;
			tracking[j].id = j;
			tracking[j].is_acti = false;

		}

	}
	else //最多只初始化MAX_TRACKING_NUM个
	{
		for (int i = 0; i < MAX_TRACKING_NUM; i++)
		{
			rect_obj.height = obj_rect[i].h; rect_obj.width = obj_rect[i].w;
			rect_obj.x = obj_rect[i].x; rect_obj.y = obj_rect[i].y;
			tracking[i].tracker = TrackerMOSSE::create();
			tracking[i].tracker->init(mat_img, rect_obj);
			tracking[i].init_rects = rect_obj;
			tracking[i].id = i;
			tracking[i].is_acti = true;

		}
	}



	return tracking;
}

vector<res_track_t> init_res_tracking(vector<bbox_t> obj_rect,int n_tracks)
{
	vector<res_track_t> res_tracking(MAX_TRACKING_NUM);
	Rect2d rect_obj;
	res_track_t res_tracking_;

	//int n = obj_rect.size();
	if (n_tracks < MAX_TRACKING_NUM)
	{

		for (int i = 0; i < n_tracks; i++)
		{
			rect_obj.height = obj_rect[i].h; rect_obj.width = obj_rect[i].w;
			rect_obj.x = obj_rect[i].x; rect_obj.y = obj_rect[i].y;

			res_tracking[i].l_tracking = 0;
			res_tracking[i].id = i;
			res_tracking[i].init_rect = rect_obj;
			res_tracking[i].cur_rect = rect_obj;
			res_tracking[i].vx = 0;
			res_tracking[i].vy = 0;
			res_tracking[i].direction =0;
			res_tracking[i].is_acti = true;
		}
		for (int j = n_tracks; j < MAX_TRACKING_NUM; j++)
        {
            rect_obj = zero_rect();
//            rect_obj.height = 0; rect_obj.width = 0;
//			rect_obj.x = 0; rect_obj.y = 0;

			res_tracking[j].l_tracking = 0;
			res_tracking[j].id = j;
			res_tracking[j].init_rect = rect_obj;
			res_tracking[j].cur_rect = rect_obj;
			res_tracking[j].vx = 0;
			res_tracking[j].vy = 0;
			res_tracking[j].direction =0;
			res_tracking[j].is_acti = false;
        }
	}
	else
	{
		for (int i = 0; i < MAX_TRACKING_NUM; i++)
		{
			rect_obj.height = obj_rect[i].h; rect_obj.width = obj_rect[i].w;
			rect_obj.x = obj_rect[i].x; rect_obj.y = obj_rect[i].y;

			res_tracking[i].l_tracking = 0;
			res_tracking[i].id = i;
			res_tracking[i].init_rect = rect_obj;
			res_tracking[i].cur_rect = rect_obj;
			res_tracking[i].vx = 0;
			res_tracking[i].vy = 0;
			res_tracking[i].direction =0;
			res_tracking[i].is_acti = true;
		}
	}
	return res_tracking;
}

vector<res_ouput_t> convert_resTrack(vector<res_track_t> src_track)
{
    vector<res_ouput_t> dst_track;
    res_ouput_t dst_track_i;
    for(int i=0;i<MAX_TRACKING_NUM;i++)
    {
        if(src_track[i].is_acti)
        {
            dst_track_i.loc_x = (SRC_POINT_X-src_track[i].cur_rect.x)*PIXEL2M;
            dst_track_i.loc_y = (SRC_POINT_Y-src_track[i].cur_rect.y)*PIXEL2M;
            dst_track_i.loc_w = src_track[i].cur_rect.width*PIXEL2M;
            dst_track_i.loc_h = src_track[i].cur_rect.height*PIXEL2M;
            dst_track_i.l_tracking = src_track[i].l_tracking;
            dst_track_i.vx = src_track[i].vx;
            dst_track_i.vy = src_track[i].vy;
            dst_track_i.direction = src_track[i].direction;
            dst_track.push_back(dst_track_i);
        }
    }
    return dst_track;
}

void print_output(vector<res_ouput_t> out)
{
    cout<<"----------------------output:---------------------"<<endl;
    for(int i=0;i<out.size();i++)
    {
        cout<<i<<">>>>>"<<endl;
        cout<<"  x="<<out[i].loc_x<<", y="<<out[i].loc_y<<", w="<<out[i].loc_w<<", h="<<out[i].loc_h<<endl;
        cout<<"  l="<<out[i].l_tracking<<", vx="<<out[i].vx<<", vy="<<out[i].vy<<", direction="<<out[i].direction<<endl;
    }
    cout<<"--------------------------------------------------"<<endl;
}


