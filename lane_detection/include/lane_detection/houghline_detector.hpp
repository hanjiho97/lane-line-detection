#ifndef LANE_DETECTION_HOUGHLINE_DETECTOR_HPP_
#define LANE_DETECTION_HOUGHLINE_DETECTOR_HPP_
#include <cmath>
#include <deque>
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

namespace size
{
	constexpr uint16_t MAX_LINE_SIZE = 128U;
	constexpr uint16_t MAX_LINE_POSITION_SIZE = 10000000U;
	constexpr uint16_t MAX_SAMPLING_SIZE = 10U;
}

struct LinePositions
{
	uint16_t left_line_position = 0U;
	uint16_t right_line_position = 640U;
};

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
	cv::Mat preprocess_Image(const cv::Mat& input_frame);
	void get_LinePositions(const cv::Mat& output_frame, LinePositions& lane);
	void add_left_sample(uint16_t new_sample);
	void add_right_sample(uint16_t new_sample);
	void get_left_weighted_mean();
	void get_right_weighted_mean();
	void draw_Points(cv::Mat& frame, LinePositions& line_positions);
private:
	void get_LinePosition(
		const std::vector<cv::Vec4i>& lines,
		bool is_left, uint16_t& line_position);
	std::deque<uint16_t> left_samples_;
	std::deque<uint16_t> right_samples_;
	std::vector<uint8_t> weights_;
	float left_mean_ = 0.0F;
	float right_mean_ = static_cast<float>(frame::WIDTH);
	cv::Mat mask_image_;
};
#endif  //LANE_DETECTION_HOUGHLINE_DETECTOR_HPP_
