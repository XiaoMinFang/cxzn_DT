#include "cxzn_tracking.hpp"

using namespace std;
using namespace cv;
//初始化tracking
void init_tracking(vector<track_t> &tracking,vector<bbox_t> obj_rect,Mat mat_img)
{
	Rect2d rect_obj;
	tracking(MAX_TRACKING_NUM);

	int n = obj_rect.size();
	if (n < MAX_TRACKING_NUM)
	{

		for (int i = 0; i < n; i++)
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
	else
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

void init_res_tracking(vector<res_track_t> &res_tracking,vector<bbox_t> obj_rect)
{
	res_tracking(MAX_TRACKING_NUM);
	Rect2d rect_obj;

	int n = obj_rect.size();
	if (n < MAX_TRACKING_NUM)
	{

		for (int i = 0; i < n; i++)
		{
			rect_obj.height = obj_rect[i].h; rect_obj.width = obj_rect[i].w;
			rect_obj.x = obj_rect[i].x; rect_obj.y = obj_rect[i].y;

			res_tracking[i].l_tracking = 0;
			res_tracking[i].id = i;
			res_tracking[i].res_track = rect_obj;

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
			res_tracking[i].res_track = rect_obj;
		}
	}
	return res_tracking;
}


void update_tracking(vector<track_t> &tracking,vector<res_track_t> &res_trcking,Mat mat_img,int n_tracks)
{
    for (int i = 0; i < n_tracks; i++)
    {
        Rect2d track_res;
        tracking[i].tracker->update(mat_img, track_res);
        if (track_res.x == 0 & track_res.y==0 & track_res.width == 0 & track_res.height == 0)
        {
            cout<<">>>>>> no trackings\n"<<endl;
        }
        res_trcking[i].res_track = track_res;
        res_trcking[i].id = i;
        res_trcking[i].l_tracking++;
        //draw_boxes(mat_img, track_res, Scalar(0, 255, 0), "track");
        /*imshow("tracking", mat_img);
        waitKey(1);*/
        //destroyAllWindows;
        //cvWaitKey(0);
    }
}
