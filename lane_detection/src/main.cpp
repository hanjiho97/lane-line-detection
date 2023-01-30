#include <string>
#include <fstream>
#include "opencv2/opencv.hpp"
#include "houghlines.h"


using namespace std;
using namespace cv;


int main(int argc, char** argv)
{
    Houghline hough;
    pair<int, int> temp;
    vector<pair<int, int>> line;
    VideoCapture cap;
    cap.open("Sub_project.avi");
    if (!cap.isOpened())
    {
        cerr << "Video open failed!\n";
        exit(1);
    }
    Mat frame;
    namedWindow("frame");
    while (true)
    {
        if (!cap.read(frame))
        {
            cout << "Video end!\n";
            break;
        }
        int pos[2] = { -1, -1 };
        hough.ProcessImage(frame, pos);
        temp.first = pos[0];
        temp.second = pos[1];
        line.push_back(temp);
        imshow("frame", frame);
        waitKey(1);
    }
    ofstream outfile;
    outfile.open("data.csv", ios::out);
    for (int j = 0; j < line.size(); j++)
    {
        outfile << line[j].first << "," << line[j].second << endl;
    }
    outfile.close();
    return 0;
}
