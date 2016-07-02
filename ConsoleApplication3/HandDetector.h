#pragma once
#include<opencv\cv.h>
using namespace std;
class HandDetector
{
public:
	HandDetector(void);
	~HandDetector(void);

	cv::Mat HandDetector::getSkin(cv::Mat input, bool isDark, bool isBright, bool isSkinSmoothing);
	cv::Mat getEdge(cv::Mat input);
	cv::Mat getMotion(cv::Mat firstMat, cv::Mat secondMat);
	cv::Mat getHand(cv::Mat skinMat, cv::Mat edgeMat, cv::Mat motionMat);
	cv::Mat hsvFilter(cv::Mat input, int satMin, int satMax, int valMin, int valMax);
	cv::Mat getClosingMat(cv::Mat input);
	cv::Mat skinSmoothing(cv::Mat input);
	cv::Point HandDetector::getHandCenterPoint(cv::Mat input,cv::Mat original);
	void SkinColorModel(cv::Mat frame, cv::Rect faceregion, int* ymax, int* ymin, int* crmax, int* crmin, int* cbmax, int* cbmin);
public:
	int hue1_min;
	int hue1_max;
	int hue2_min;
	int hue2_max;

	int darkSat_min;
	int darkSat_max;
	int darkVal_min;
	int darkVal_max;

	int normSat_min;
	int normSat_max;
	int normVal_min;
	int normVal_max;

	int brightSat_min;
	int brightSat_max;
	int brightVal_min;
	int brightVal_max;

	int width;
	int height;
	int widthScale;
	int heightScale;
	int ObjectSizeMin;
	int ObjectSizeMax;
};
