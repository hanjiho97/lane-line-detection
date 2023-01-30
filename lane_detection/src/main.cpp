#include <string>
#include <fstream>
#include "opencv2/opencv.hpp"
#include "houghlines.h"


int32_t main(int32_t argc, char** argv)
{
    Houghline_Detector houghline_detector;
    std::pair<int, int> current_line;
    std::vector<std::pair<int, int>> lines;
    cv::VideoCapture cap;
    cap.open("Sub_project.avi");
    if (!cap.isOpened())
    {
        std::cerr << "Video open failed!" << std::endl;
        exit(1);
    }
    cv::Mat input_frame;
    cv::Mat output_frame;
    LinePositions lane;
    cv::namedWindow("frame");
    while(true)
    {
        if (!cap.read(input_frame))
        {
            std::cout << "Video end!" << std::endl;
            break;
        }
        output_frame = houghline_detector.preprocess_Image(input_frame);
        houghline_detector.get_LinePositions(output_frame, lane);
        houghline_detector.draw_Points(input_frame, lane);
        current_line.first = lane.left_line_position;
        current_line.second = lane.right_line_position;
        lines.push_back(current_line);
        imshow("input_frame", input_frame);
        waitKey(1);
    }
    ofstream outfile;
    outfile.open("data.csv", ios::out);
    for (uint32_t j = 0; j < line.size(); ++j)
    {
        outfile << line[j].first << "," << line[j].second << std::endl;
    }
    outfile.close();
    return 0;
}
