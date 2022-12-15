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
    //lpos rpos값을 저장할 csv파일 open
    outfile.open("test.csv", ios::out);
    //csv파일에 저장
    for (int j = 0; j < line.size(); j++)
    {
        outfile << line[j].first << "," << line[j].second << endl;
    }
    outfile.close();
    // 콘솔에서 csv에 입력된 값 출력하기(확인용)
    /*
    string str;
    ifstream input("test.csv");
    while (!input.eof()) {
        getline(input, str);
        cout << str << endl;
    }
    input.close();
    */
    return 0;
}