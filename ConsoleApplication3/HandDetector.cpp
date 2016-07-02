#include "stdafx.h"
#include "HandDetector.h"
#include"opencv2\opencv.hpp"
#include "opencv2\highgui\highgui.hpp"
#include "opencv2\imgproc\imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include <iostream>
#include <stdio.h>
#include <stdlib.h>


HandDetector::HandDetector(void)
{
	hue1_min = 0;
	hue1_max = 15;
	hue2_min = 135;
	hue2_max = 180;

	darkSat_min = 100;
	darkSat_max = 255;
	darkVal_min = 0;
	darkVal_max = 60;

	normSat_min = 51;
	normSat_max = 153;
	normVal_min = 102;
	normVal_max = 255;

	brightSat_min = 30;
	brightSat_max = 255;
	brightVal_min = 30;
	brightVal_max = 255;

}
HandDetector::~HandDetector(void)
{
}
cv::Mat HandDetector::getSkin(cv::Mat input, bool isDark, bool isBright, bool isSkinSmoothing)
{
	cv::Mat skin, conditionSkin,blur;
	cv::GaussianBlur(input, blur, cv::Size(5, 5), 0);
	skin = hsvFilter(blur, normSat_min, normSat_max, normVal_min, normVal_max);
	if (isBright) {
		conditionSkin = hsvFilter(blur, brightSat_min, brightSat_max, brightVal_min, brightVal_max);
		skin = skin | conditionSkin;
	}
	if (isDark) {
		conditionSkin = hsvFilter(blur, darkSat_min, darkSat_max, darkVal_min, darkVal_max);
		skin = skin | conditionSkin;
	}
	
	imshow("skinsss", skin);
	if (isSkinSmoothing) {
		skin = skinSmoothing(skin);
	}
	

	return skin;
}

cv::Mat HandDetector::hsvFilter(cv::Mat input, int satMin, int satMax, int valMin, int valMax)
{
	cv::Mat skin, skinHSVFilter1, skinHSVFilter2, result;
	cvtColor(input, skin, cv::COLOR_BGR2HSV);
	inRange(skin, cv::Scalar(hue1_min, satMin, valMin), cv::Scalar(hue1_max, satMax, valMax), skinHSVFilter1);
	inRange(skin, cv::Scalar(hue2_min, satMin, valMin), cv::Scalar(hue2_max, satMax, valMax), skinHSVFilter2);

	result = skinHSVFilter1 | skinHSVFilter2;
	return result;
}

cv::Mat HandDetector::skinSmoothing(cv::Mat input)
{
	cv::Mat result;

	//dispersing skin area to get more skin area
	cv::Mat structuringElement = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
	morphologyEx(input, result, cv::MORPH_OPEN, structuringElement);
	morphologyEx(result, result, cv::MORPH_CLOSE, structuringElement);
	for (int i = 0; i < 10; i++) {
		morphologyEx(result, result, cv::MORPH_DILATE, structuringElement);
		morphologyEx(result, result, cv::MORPH_DILATE, structuringElement);
		morphologyEx(result, result, cv::MORPH_ERODE, structuringElement);
	}
	return result;
}

cv::Mat HandDetector::getEdge(cv::Mat input)
{
	cv::Mat result, bwimage;
	int lowThreshold = 50;
	int highThreshold = 3 / 2 * lowThreshold;
	int kernelSize = 3;

	cv::cvtColor(input, bwimage, CV_BGR2GRAY);
	//cv::GaussianBlur(input, bwimage, cv::Size(3, 3), 0, 0, cv::BORDER_DEFAULT);
	cv::blur(bwimage, bwimage, cv::Size(3, 3));
	//cv::imshow("blur", bwimage);
	cv::threshold(bwimage, result, 30, 255, CV_THRESH_BINARY);
	cv::Canny(bwimage, result, 50, 75, kernelSize, true);
	return result;
}

cv::Mat HandDetector::getMotion(cv::Mat firstMat, cv::Mat secondMat)
{
	cv::Mat result;
	int minThreshold = 30;
	int maxThreshold = 255;
	cvtColor(firstMat, firstMat, cv::COLOR_BGR2GRAY);
	cvtColor(secondMat, secondMat, cv::COLOR_BGR2GRAY);

	result = abs(secondMat - firstMat);

	threshold(result, result, minThreshold, maxThreshold, CV_THRESH_BINARY);
	cv::Mat structuringElement = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
	morphologyEx(result, result, cv::MORPH_OPEN, structuringElement);

	return result;
}

cv::Mat HandDetector::getHand(cv::Mat skinMat, cv::Mat edgeMat, cv::Mat motionMat)
{
	cv::Mat result, blurred;
	/*//best result now
	skinEdge = skinMat & edgeMat;
	result = skinEdge | motionMat;
	//skinEdge = skinMat & motionMat;
	imshow("tess", skinEdge);
	//result = skinEdge & edgeMat;
//	result = getClosingMat(result);*/
	cv::blur(skinMat, blurred, cv::Size(3, 3));
	imshow("blur",blurred);
	blurred.copyTo(result, edgeMat);
	return result;
}

cv::Mat HandDetector::getClosingMat(cv::Mat input)
{
	cv::Mat result;
	cv::Mat structuringElement = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
	morphologyEx(input, result, cv::MORPH_CLOSE, structuringElement);
	for (int i = 0; i < 10; i++) {
		morphologyEx(result, result, cv::MORPH_DILATE, structuringElement);
		//morphologyEx(result, result, cv::MORPH_DILATE, structuringElement);
		morphologyEx(result, result, cv::MORPH_ERODE, structuringElement);
	}
	return result;
}

cv::Point HandDetector::getHandCenterPoint(cv::Mat handArea,cv::Mat originalImage)
{
	vector<vector<cv::Point> > contours,hulls;
	vector<cv::Vec4i> hierarchy;
	vector<cv::Vec4i> convexityDefect;
	vector<int> hullsI;
	vector<cv::Point> hull;
	cv::Point2f cp = cv::Point2f(0,0);
	double area,max_area = 0.0;
	int index;
	
	int x, y;
	int width = 320;
	int height = 320;
	int widthScale = width / 5;
	int heightScale = height / 5;
	ObjectSizeMax = 1200;
	ObjectSizeMin = 10;

	findContours(handArea, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE, cv::Point(0, 0));
	
	
	for (int i = 0; i < contours.size(); i++) {
		convexHull(contours[i], hull, false);
		hulls.push_back(hull);
		area = fabs(cv::contourArea(contours[i], false));
		if (area > max_area)
		{
			convexHull(contours[i], hullsI, false);
			max_area = area;
			index = i;
		}
	}
	drawContours(originalImage, contours, index, cv::Scalar(255, 0, 255), 5, 8);
	
	//for (int i = 0; i < contours.size(); i++) {
	drawContours(originalImage, hulls, index, cv::Scalar(255, 120, 255), 5, 8);
	if (!contours.empty() && !contours[index].empty())
	{
		if (!hulls.empty() && !hullsI.empty())
		{
			convexityDefects(contours[index], hullsI, convexityDefect);
			cv::Moments moment = moments((cv::Mat)contours[index]);
			double size = moment.m00;
			if ((size > ObjectSizeMin)) {
				x = moment.m10 / size;
				y = moment.m01 / size;
				cp = cv::Point2f(x, y);
			}
			else {
				cp = cv::Point2f(0, 0);
			}
		}
			
	}
	
		
	//}
	for (int j = 0; j < convexityDefect.size(); j++)
	{
		const cv::Vec4i v = convexityDefect[j];
		float depth = v[3] / 256;
		if (depth > 10)
		{
			int startidx = v[0]; cv::Point ptStart(contours[index][startidx]);
			int endidx = v[1]; cv::Point ptEnd(contours[index][endidx]);
			int faridx = v[2]; cv::Point ptFar(contours[index][faridx]);

			cv::line(originalImage, ptStart, ptEnd, cv::Scalar(0, 255, 0), 1);
			cv::line(originalImage, ptStart, ptFar, cv::Scalar(0, 255, 0), 1);
			cv::line(originalImage, ptEnd, ptFar, cv::Scalar(0, 255, 0), 1);
			cv::circle(originalImage, ptFar, 4, cv::Scalar(0, 255, 0), 2);
		}
	}

	imshow("Original Image", originalImage);
	/*if (hierarchy.size() > 0) {
		int numObjects = hierarchy.size();
		for (int index = 0; index >= 0; index = hierarchy[index][0]) {

			
			

			
		}
	}*/
	
	return cp;
}


void SkinColorModel(cv::Mat frame, cv::Rect faceregion, int* ymax, int* ymin, int* crmax, int* crmin, int* cbmax, int* cbmin)
{
	int y, cb, cr, r, b, g, gray;
	cv::Mat p;
	cv::cvtColor(frame, p, CV_BGR2YCrCb);
	*crmax = -1;
	*crmin = 295;
	*cbmax = -1;
	*cbmin = 295;
	*ymax = 295;
	*ymin = -1;
	if (faceregion.area() > 5)
	{

		for (int i = faceregion.x; i < faceregion.x + faceregion.width&& i < frame.cols; i++)
		{
			for (int j = faceregion.y; j < faceregion.y + faceregion.height&& j<frame.rows; j++)
			{

				b = frame.at<cv::Vec3b>(j, i)[0];
				g = frame.at<cv::Vec3b>(j, i)[1];
				r = frame.at<cv::Vec3b>(j, i)[2];
				y = p.at<cv::Vec3b>(j, i)[0];
				cr = p.at<cv::Vec3b>(j, i)[1];
				cb = p.at<cv::Vec3b>(j, i)[2];
				gray = (int)(0.2989 * r + 0.5870 * g + 0.1140 * b);
				if (gray<200 && gray>40 && r>g && r>b)
				{
					*ymax = (y > *ymax) ? y : *ymax;
					*ymin = (y < *ymin) ? y : *ymin;
					*crmax = (cr > *crmax) ? cr : *crmax;
					*crmin = (cr < *crmin) ? cr : *crmin;
					*cbmax = (cb > *cbmax) ? cb : *cbmax;
					*cbmin = (cb < *cbmin) ? cb : *cbmin;
				}
			}
		}
		/**ymin = *ymin - 10;
		*ymax = *ymax + 10;
		*crmin = *crmin - 10;
		*crmax = *crmax + 10;
		*cbmin = *cbmin - 10;
		*cbmax = *cbmax + 10;*/
	}
	else
	{
		*ymax = 255;//(*ymax>163) ? 163 : *ymax;
		*ymin = 0;// (*ymin < 54) ? 54 : *ymin;
		*crmax = 173;// (*crmax > 173) ? 173 : *crmax;
		*crmin = 133;// (*crmin < 133) ? 133 : *crmin;
		*cbmax = 127;// (*cbmax > 127) ? 127 : *cbmax;
		*cbmin = 77;// (*cbmin < 77) ? 77 : *cbmin;
	}
	/**crmax = (*crmax > 173) ? 173 : *crmax;
	*crmin = (*crmin < 133) ? 133 : *crmin;
	*cbmax = (*cbmax > 127) ? 127 : *cbmax;
	*cbmin = (*cbmin < 77) ? 77 : *cbmin;*/
}