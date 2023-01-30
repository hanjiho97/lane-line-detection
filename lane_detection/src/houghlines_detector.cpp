#include <cmath>
#include "lane_detection/houghlines_detector.hpp"


Houghline_Detector::Houghline_Detector()
{
  mask_image = cv::imread("mask.png", cv::IMREAD_GRAYSCALE);
  if (mask_img.empty())
  {
    std::cerr << "Mask image load failed!" << std::endl;
    exit(1);
  }
}

void Houghline_Detector::divide_LeftRight(
  const std::vector<cv::Vec4i>& all_lines,
  std::vector<cv::Vec4i>& left_lines,
  std::vector<cv::Vec4i>& right_lines)
{
  std::vector<float> slopes;
  slopes.reserve(MAX_LINE_SIZE);
  std::vector<cv::Vec4i> new_lines;
  new_lines.reserve(MAX_LINE_SIZE);
  uint16_t x1 = 0U;
  uint16_t x2 = 0U;
  uint16_t y1 = 0U;
  uint16_t y2 = 0U;
  float slope = 0.0F;
  filter_out_lines(all_lines, slopes, new_lines);
  //split left line and right line
  for (uint32_t i = 0U; i < slopes.size(); ++i)
  {
    cv::Vec4i new_line = new_lines[i];
    slope = slopes[i];
    x1 = new_line[0];
    x2 = new_line[2];
    y1 = new_line[1];
    y2 = new_line[3];
    float x_mean = static_cast<float>(x1 + x2) / 2.0F;
    if ((slope < 0.0F) && (x2 < frame::HALF_WIDTH) &&
      ((std::abs(previous_left_ - x_mean) < 20) || (previous_left_ == 0)))
    {
      left_lines.push_back(new_line);
    }
    else if ((slope > 0.0F) && (x1 > frame::HALF_WIDTH) &&
      ((std::abs(previous_right_ - x_mean) < 20) || (previous_right_ == frame::WIDTH)))
    {
      right_lines.push_back(new_line);
    }
  }
}

void Houghline_Detector::filter_out_lines(
  const std::vector<cv::Vec4i>& all_lines,
  std::vector<float> slopes,
  std::vector<cv::Vec4i> new_lines)
{
  uint16_t x1 = 0U;
  uint16_t x2 = 0U;
  uint16_t y1 = 0U;
  uint16_t y2 = 0U;
  float slope = 0.0F;
  //line filitering by slope
  for (auto& line : all_lines)
  {
    x1 = line[0];
    x2 = line[2];
    y1 = line[1];
    y2 = line[3];
    if ((x2 - x1) == 0)
    {
      slope = 0.0F;
    }
    else
    {
      slope = static_cast<float>(y2 - y1) / static_cast<float>(x2 - x1);
    }
    if (0.0 < std::abs(slope))
    {
      slopes.push_back(slope);
      new_lines.push_back(line);
    }
  }
}

bool Houghline_Detector::get_LineParams(
  const std::vector<cv::Vec4i>& lines,
  float& slope, float& y_intercept)
{
  float x_sum = 0.0F;
  float y_sum = 0.0F;
  float slope_sum = 0.0F;
  bool result = false;
  int32_t x1 = -1;
	int32_t y1 = -1;
	int32_t x2 = -1;
	int32_t y2 = -1;
  const auto line_size = lines.size();
  if (line_size == 0U) {
    slope = 0;
    y_intercept = 0;
  }
  else
  {
    for (auto& line : lines) {
      x1 = line[0U];
      x2 = line[2U];
      y1 = line[1U];
      y2 = line[3U];
      x_sum += static_cast<float>(x1 + x2);
      y_sum += static_cast<float>(y1 + y2);
      m_sum += static_cast<float>(y2 - y1) / static_cast<float>(x2 - x1);
    }
    float x_avg = x_sum / static_cast<float>(line_size * 2);
    float y_avg = y_sum / static_cast<float>(line_size * 2);
    slope = slope_sum / line_size;
    y_intercept = y_avg - slope * x_avg;
    result = true;
  }
  return result;
}

void Houghline_Detector::get_LinePosition(
  const std::vector<cv::Vec4i>& lines,
  bool is_left, int32_t& line_position)
{
    float slope = 0.0F;
    float y_intercept = 0.0F;
    if (!GetLineParams(lines, slope, y_intercept))
    {
      if (is_left)
      {
        line_position = 0U;
      }
      else
      {
        line_position = frame::WIDTH;
      }
    }
    else
    {
      line_position = static_cast<int32_t>((frame::HALF_GAP - y_intercept) / slope);
    }
}

cv::Mat Houghline_Detector::preprocess_Image(const cv::Mat& input_frame)
{
  double global_min = 255.0;
  double global_gmax = 0.0;
  cv::Mat gray_img;
  cvtColor(input_frame, gray_img, cv::COLOR_BGR2GRAY);
  cv::minMaxLoc(gray_img, &global_min, &global_gmax);
  cv::Mat strech_img;
  strech_img = (gray_img - global_min) * 255 / (global_gmax - global_min);
  cv::Mat blur_img;
  cv::GaussianBlur(strech_img, blur_img, cv::Size(5, 5), 1.5);
  cv::Mat edge_img;
  cv::Canny(blur_img, edge_img, 100, 200);
  cv::bitwise_and(edge_img, mask_img, masked_img);
  dilate(masked_img, masked_img, cv::Mat());
  cv::Mat roi;
  roi = masked_img(cv::Range(OFFSET, OFFSET + GAP), cv::Range(0, WIDTH));
  return roi;
}

void Houghline_Detector::get_LinePositions(const cv::Mat& output_frame, LinePositions& lane)
{
  std::vector<cv::Vec4i> all_lines;
  std::vector<cv::Vec4i> left_lines;
  std::vector<cv::Vec4i> right_lines;
  HoughLinesP(output_frame, all_lines, 1, (CV_PI / 180.0), 30, 12.5, 5);
  if (all_lines.size() == 0U)
  {
    lane.left_line_position = 0;
    lane.right_line_position = 640;
  }
  else
  {
    DivideLeftRight(all_lines, left_lines, right_lines);
    // get center of lines
    uint16_t left_position = 0U;
    uint16_t right_position = frame::WIDTH;
    get_LinePosition(left_lines, true, left_position);
    get_LinePosition(right_lines, false, right_position);
    lane.left_line_position = left_position;
    lane.right_line_position = right_position;
    previous_left_ = left_position;
    previous_right_ = right_position;
  }
}

void Houghline_Detector::draw_Points(cv::Mat& input_frame, LinePosition& lane)
{
  line(input_frame,
  cv::Point(lane.left_line_position, frame::LANE_HEIGHT),
  cv::Point(lane.left_line_position, frame::LANE_HEIGHT),
  cv::Scalar(255, 0, 0), 7, cv::LINE_AA);
  line(input_frame,
  cv::Point(lane.right_line_position, frame::LANE_HEIGHT),
  cv::Point(lane.right_line_position, frame::LANE_HEIGHT),
  cv::Scalar(255, 0, 0), 7, cv::LINE_AA);
}
