#ifndef LANE_DETECTION_HOUGHLINE_DETECTOR_HPP_
#define LANE_DETECTION_HOUGHLINE_DETECTOR_HPP_
#include <vector>
#include "opencv2/opencv.hpp"

namespace frame
{
constexpr uint16_t WIDTH = 640;
constexpr uint16_t HEIGHT = 480;
constexpr uint16_t OFFSET = 385;
constexpr uint16_t GAP = 30;
}

struct LinePositions
{
  uint16_t left_line_position = 0;
  uint16_t right_line_position = 640;
}

class Houghline_Detector
{
public:
	Houghline_Detector(uint16_t previous_left, uint16_t previous_right);
	virtual ~Houghline_Detector() = default;
	void divide_LeftRight(
		const std::vector<cv::Vec4i>& all_lines,
		std::vector<cv::Vec4i>& left_lines,
		std::vector<cv::Vec4i>& right_lines);
	void get_LineParams(
    const std::vector<cv::Vec4i>& lines,
    float slope, float y_intercept);
	cv::Mat preprocess_Image(cv::Mat& frame);
  LinePosition get_LinePositions(const cv::Mat& frame);
  void draw_Lines(cv::Mat& frame, LinePosition& line_positions);

private:
	void get_LinePosition(
		const std::vector<cv::Vec4i>& lines,
    bool is_left, float line_x1, float line_x2, int line_position);
	uint16_t previous_left = 0
  uint16_t previous_right = WIDTH;
};
#endif  //LANE_DETECTION_HOUGHLINE_DETECTOR_HPP_
