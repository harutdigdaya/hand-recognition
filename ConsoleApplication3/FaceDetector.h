#pragma once
#include<opencv\cv.h>
using namespace std;
class FaceDetector
{
public:
	~FaceDetector(void);
	cv::Mat images;
	cv::Rect detectFace(cv::Mat image);
	cv::Mat removeFace(cv::Rect faceRegion, cv::Mat frame);
};

