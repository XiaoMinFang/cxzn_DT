#include "DTCommon.hpp"
using namespace std;
using namespace cv;
void draw_boxes(Mat img, Rect2d box,Scalar color,string text)
{
	rectangle(img, box, color, 1);
	Size const text_size = getTextSize(text, FONT_HERSHEY_COMPLEX_SMALL, 1.2, 2, 0);
	int const max_width = (text_size.width > box.width + 2) ? text_size.width : (box.width + 2);
	/*rectangle(img, Point2f(max((int)box.x - 1, 0), max((int)box.y - 30, 0)),
		Point2f(min((int)box.x + max_width, img.cols - 1), min((int)box.y, img.rows - 1)),color, CV_FILLED, 8, 0);*/
	putText(img, text, Point2f(box.x, box.y - 10), FONT_HERSHEY_COMPLEX_SMALL, 1.2, color, 1);

}

void draw_tracking_res(Mat img,vector<res_track_t> res_track,Scalar color,string text)
{
    for(int i = 0;i<MAX_TRACKING_NUM;i++)
    {
        Rect2d b = res_track[i].cur_rect;
        if (b.width != 0)
        {
            rectangle(img, b, color, 1);
            Size const text_size = getTextSize(text, FONT_HERSHEY_COMPLEX_SMALL, 1.2, 2, 0);
            int const max_width = (text_size.width > b.width + 2) ? text_size.width : (b.width + 2);
            /*rectangle(img, Point2f(max((int)box.x - 1, 0), max((int)box.y - 30, 0)),
                Point2f(min((int)box.x + max_width, img.cols - 1), min((int)box.y, img.rows - 1)),color, CV_FILLED, 8, 0);*/
            putText(img, text, Point2f(b.x, b.y - 10), FONT_HERSHEY_COMPLEX_SMALL, 1.2, color, 1);
        }

    }

}


void draw_detect_boxes(Mat mat_img, vector<bbox_t> result_vec, vector<string> obj_names,int current_det_fps, int current_cap_fps)
{
	int const colors[6][3] = { { 1,0,1 },{ 0,0,1 },{ 0,1,1 },{ 0,1,0 },{ 1,1,0 },{ 1,0,0 } };

	for (auto &i : result_vec) {
		Scalar color = obj_id_to_color(i.obj_id);
		rectangle(mat_img, Rect(i.x, i.y, i.w, i.h), color, 1);
		if (obj_names.size() > i.obj_id) {
			string obj_name = obj_names[i.obj_id];
			if (i.track_id > 0) obj_name += " - " + to_string(i.track_id);
			Size const text_size = getTextSize(obj_name, FONT_HERSHEY_COMPLEX_SMALL, 1.2, 2, 0);
			int const max_width = (text_size.width > i.w + 2) ? text_size.width : (i.w + 2);
			/*rectangle(mat_img, Point2f(max((int)i.x - 1, 0), max((int)i.y - 30, 0)),
				Point2f(min((int)i.x + max_width, mat_img.cols - 1), min((int)i.y, mat_img.rows - 1)),
				color, CV_FILLED, 8, 0);*/
			putText(mat_img, obj_name, Point2f(i.x, i.y - 10), FONT_HERSHEY_COMPLEX_SMALL, 1.2, color, 1);
		}
	}
	if (current_det_fps >= 0 && current_cap_fps >= 0) {
		string fps_str = "FPS detection: " + to_string(current_det_fps) + "   FPS capture: " + to_string(current_cap_fps);
		putText(mat_img, fps_str, Point2f(10, 20), FONT_HERSHEY_COMPLEX_SMALL, 1.2, Scalar(50, 255, 0), 2);
	}
}

vector<string> get_objNames_fromFile(string const filename) {
	ifstream file(filename);
	vector<string> file_lines;
	if (!file.is_open()) return file_lines;
	for (string line; getline(file, line);) file_lines.push_back(line);
	cout << "object names loaded \n";
	return file_lines;
}

