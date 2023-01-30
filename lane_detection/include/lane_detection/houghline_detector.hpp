#pragma once
#include "opencv2/opencv.hpp"

constexpr int WIDTH = 640;
constexpr int HEIGHT = 480;
constexpr int OFFSET = 385;
constexpr int GAP = 30;
typedef std::vector<cv::Vec4i> Line;

class Houghline
{
public:
	Houghline() {};
	~Houghline() {};
	void DivideLeftRight(const Line& all_lines, Line& left_lines, Line& right_lines);
	inline void GetLineParams(const Line& lines, float& m, float& b);
	void GetLinePosition(const Line& lines, bool is_left, float& line_x1, float& line_x2, int& line_pos);
	void ProcessImage(const cv::Mat& frame, int pos[]);
private:
	int pre_left = 0, pre_right = WIDTH;
};
