#include <cmath>
#include <vector>
#include "houghlines.h"


void Houghline::DivideLeftRight(const Line& all_lines, Line& left_lines, Line& right_lines)
{
    std::vector<float> slopes;
    Line new_lines;
    int x1 = 0, x2 = 0;
    int y1 = 0, y2 = 0;
    float slope = 0;
    //line filitering by slope
    for (auto& line : all_lines) {
        x1 = line[0], x2 = line[2];
        y1 = line[1], y2 = line[3];
        if (x2 - x1 == 0) {
            slope = 0;
        }
        else {
            slope = float(y2 - y1) / float(x2 - x1);
        }
        if (0 < abs(slope)) {
            slopes.push_back(slope);
            new_lines.push_back(line);
        }
    }
    //split left line and right line
    for (int i = 0; i < slopes.size(); i++) {
        cv::Vec4i new_line = new_lines[i];
        slope = slopes[i];
        x1 = new_line[0], x2 = new_line[2];
        y1 = new_line[1], y2 = new_line[3];
        float x_mean = float(x1 + x2) / 2;
        if (slope < 0 && x2 < WIDTH / 2 &&
            (abs(pre_left - x_mean) < 20 || pre_left == 0))
        {
            left_lines.push_back(new_line);
        }
        else if (slope > 0 && x1 > WIDTH / 2 &&
            (abs(pre_right - x_mean) < 20 || pre_right == WIDTH))
        {
            right_lines.push_back(new_line);
        }
    }
}


void Houghline::GetLineParams(const Line& lines, float& m, float& b) {
    float x_sum = 0.0, y_sum = 0.0, m_sum = 0.0;
    int x1, y1, x2, y2;
    int size = lines.size();
    if (!size) {
        m = 0; b = 0;
        return;
    }
    for (auto& line : lines) {
        x1 = line[0], x2 = line[2];
        y1 = line[1], y2 = line[3];
        x_sum += x1 + x2;
        y_sum += y1 + y2;
        m_sum += (float(y2 - y1) / float(x2 - x1));
    }
    float x_avg = float(x_sum) / float(size * 2);
    float y_avg = float(y_sum) / float(size * 2);
    m = m_sum / size;
    b = y_avg - m * x_avg;
}


void Houghline::GetLinePosition(const Line& lines, bool is_left, float& line_x1, float& line_x2, int& line_pos) {
    float m = 0, b = 0;
    int y = GAP / 2;
    GetLineParams(lines, m, b);
    line_x1 = 0, line_x2 = 0;
    if (m == 0 && b == 0) {
        if (is_left) {
            line_pos = 0;
        }
        else {
            line_pos = WIDTH;
        }
    }
    else {
        line_pos = (y - b) / m;
        b += OFFSET;
        line_x1 = (HEIGHT - b) / m;
        line_x2 = ((HEIGHT / 2) - b) / float(m);
    }
}


void Houghline::ProcessImage(const cv::Mat& frame, int pos[]) {
    double global_min = 255, global_gmax = 0;
    cv::Mat mask_img, gray_img, strech_img, blur_img, edge_img, masked_img;
    cv::Mat roi;
    //mask image load
    mask_img = cv::imread("mask.png", cv::IMREAD_GRAYSCALE);
    if (mask_img.empty()) {
        std::cerr << "Mask image load failed!\n";
        exit(1);
    }
    //covert to gray image
    cvtColor(frame, gray_img, cv::COLOR_BGR2GRAY);
    cv::minMaxLoc(gray_img, &global_min, &global_gmax);
    //histogram strech
    strech_img = (gray_img - global_min) * 255 / (global_gmax - global_min);
    //blur
    cv::GaussianBlur(strech_img, blur_img, cv::Size(5, 5), 1.5);
    //canny edge
    cv::Canny(blur_img, edge_img, 100, 200);
    //masking
    cv::bitwise_and(edge_img, mask_img, masked_img);
    dilate(masked_img, masked_img, cv::Mat());
    imshow("masked_img", masked_img);
    // roi
    roi = masked_img(cv::Range(OFFSET, OFFSET + GAP), cv::Range(0, WIDTH));
    // hough
    Line all_lines;
    Line left_lines, right_lines;
    HoughLinesP(roi, all_lines, 1, (CV_PI / 180), 30, 12.5, 5);
    // devide left, right lines
    if (!all_lines.size()) {
        pos[0] = 0; pos[1] = 640;
        return;
    }
    DivideLeftRight(all_lines, left_lines, right_lines);
    // get center of lines
    float left_x1 = 0, left_x2 = 0;
    float right_x1 = WIDTH, right_x2 = WIDTH;
    int left_pos = 0, right_pos = WIDTH;
    GetLinePosition(left_lines, true, left_x1, left_x2, left_pos);
    GetLinePosition(right_lines, false, right_x1, right_x2, right_pos);
    // ����
    line(frame,
        cv::Point(int(left_x1), HEIGHT),
        cv::Point(int(left_x2), (HEIGHT / 2)),
        cv::Scalar(255, 0, 0), 3);
    line(frame,
        cv::Point(int(right_x1), HEIGHT),
        cv::Point(int(right_x2), (HEIGHT / 2)),
        cv::Scalar(255, 0, 0), 3);
    pos[0] = left_pos, pos[1] = right_pos;
     //cout << "left_pos: " << left_pos << "right_pos: " << right_pos << endl;
    pre_left = left_pos, pre_right = right_pos;
}
