#include <iostream>
#include <random>
#include <string>
#include <fstream>
#include "opencv2/opencv.hpp"


int main()
{
	cv::VideoCapture cap("Sub_project.avi");

	if (!cap.isOpened()) {
		std::cerr << "Video open failed!" << std::endl;
		return -1;
	}
	cv::Mat mask = cv::imread("mask.png", cv::IMREAD_GRAYSCALE);
	if (mask.empty()) {
		std::cerr << "Mask image load failed!" << std::endl;
		return -1;
	}

	cv::Mat frame, strech, gray, blur, edge, final;
	while (true) {
		cap >> frame;
		if (frame.empty())
			break;

		cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
		double gmin, gmax;
		cv::minMaxLoc(gray, &gmin, &gmax);
		strech = (gray - gmin) * 255 / (gmax - gmin);
		cv::GaussianBlur(strech, blur, cv::Size(5, 5), 1.5);
		cv::Canny(blur, edge, 100, 200);
		cv::bitwise_and(edge, mask, final);

		cv::imshow("mask", mask);
		cv::imshow("frame", frame);
		cv::imshow("equal", final);

		if (cv::waitKey(10) == 27)
			break;
	}
}
