#ifndef LANE_DETECTION_HOUGHLINE_DETECTOR_HPP_
#define LANE_DETECTION_HOUGHLINE_DETECTOR_HPP_
#include <vector>
#include "opencv2/opencv.hpp"

namespace frame
{
constexpr uint16_t WIDTH = 640U;
constexpr uint16_t HEIGHT = 480U;
constexpr uint16_t OFFSET = 385U;
constexpr uint16_t GAP = 30U;
}

struct LinePositions
{
  uint16_t left_line_position = 0U;
  uint16_t right_line_position = 640U;
}

class Houghline_Detector
{
public:
	Houghline_Detector() = default;
	virtual ~Houghline_Detector() = default;
	void divide_LeftRight(
		const std::vector<cv::Vec4i>& all_lines,
		std::vector<cv::Vec4i>& left_lines,
		std::vector<cv::Vec4i>& right_lines);
  void filter_out_lines(
    const std::vector<cv::Vec4i>& all_lines,
    std::vector<float>& slopes,
    std::vector<cv::Vec4i>& new_lines);
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
  constexpr uint16_t MAX_LINE_SIZE = 64U;
	uint16_t previous_left_ = 0U;
  uint16_t previous_right_ = frame::WIDTH;
};
#endif  //LANE_DETECTION_HOUGHLINE_DETECTOR_HPP_
