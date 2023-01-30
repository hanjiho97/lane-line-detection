#ifndef LANE_DETECTION_HOUGHLINE_DETECTOR_HPP_
#define LANE_DETECTION_HOUGHLINE_DETECTOR_HPP_
#include <vector>
#include "opencv2/opencv.hpp"

namespace frame
{
constexpr uint16_t WIDTH = 640U;
constexpr uint16_t HALF_WIDTH = 320U;
constexpr uint16_t HEIGHT = 480U;
constexpr uint16_t HALF_HEIGHT = 240U;
constexpr uint16_t OFFSET = 385U;
constexpr uint16_t GAP = 30U;
constexpr uint16_t HALF_GAP = 15U;
constexpr uint16_t LANE_HEIGHT = 400U;
}

struct LinePositions
{
  uint16_t left_line_position = 0U;
  uint16_t right_line_position = 640U;
}

class Houghline_Detector
{
public:
	Houghline_Detector();
	virtual ~Houghline_Detector() = default;
	void divide_LeftRight(
		const std::vector<cv::Vec4i>& all_lines,
		std::vector<cv::Vec4i>& left_lines,
		std::vector<cv::Vec4i>& right_lines);
  void filter_out_lines(
    const std::vector<cv::Vec4i>& all_lines,
    std::vector<float>& slopes,
    std::vector<cv::Vec4i>& new_lines);
	bool get_LineParams(
    const std::vector<cv::Vec4i>& lines,
    float& slope, float& y_intercept);
	cv::Mat preprocess_Image(cv::Mat& input_frame);
  void get_LinePositions(const cv::Mat& output_frame);
  void draw_Points(cv::Mat& frame, LinePosition& line_positions);

private:
	void get_LinePosition(
		const std::vector<cv::Vec4i>& lines,
    bool is_left, float& line_x1, float& line_x2, uint16_t& line_position);
  constexpr uint16_t MAX_LINE_SIZE = 64U;
	uint16_t previous_left_ = 0U;
  uint16_t previous_right_ = frame::WIDTH;
  cv::Mat mask_image;
};
#endif  //LANE_DETECTION_HOUGHLINE_DETECTOR_HPP_
