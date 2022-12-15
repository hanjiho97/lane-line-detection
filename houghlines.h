#pragma once
#include "opencv2/opencv.hpp"

class Houghline
{
public:
	typedef std::vector<cv::Vec4i> Line;
	Houghline();
	~Houghline();
	void DivideLeftRight(const Line& all_lines, Line& left_lines, Line& right_lines);
	inline void GetLineParams(const Line& lines, float& m, float& b);
	void GetLinePosition(const Line& lines, bool is_left, float& line_x1, float& line_x2, int& line_pos);
	void ProcessImage(const cv::Mat& frame, int pos[]);
private:
	const int WIDTH = 640;
	const int HEIGHT = 480;
	const int OFFSET = 385;
	const int GAP = 30;
	int pre_left = 0, pre_right = WIDTH;
}; 