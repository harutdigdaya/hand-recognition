#include "stdafx.h"
#include "FaceDetector.h"
#include "opencv2/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
using namespace cv;
using namespace std;

FaceDetector::~FaceDetector(void)
{
}
cv::Rect FaceDetector:: detectFace(cv::Mat image)
{
	Mat gray;
	int maxarea = -1;
	int maxareai = -1;
	vector<Rect> faces;
	Rect p = Rect(0, 0, 0, 0);
	CascadeClassifier face_cascade = CascadeClassifier("D:/JTK-POLBAN/PKM/opencv_git/data/haarcascades/haarcascade_frontalface_alt.xml");
	Mat Original = image;
	cvtColor(image, gray, CV_RGB2GRAY);
	equalizeHist(gray, gray);
	face_cascade.detectMultiScale(gray, faces, 1.1, 2, 0 | CV_HAAR_SCALE_IMAGE, cv::Size(30, 30));
	Rect ROI;
	for (int i = 0; i < faces.size(); i++)
	{
		Point center(faces[i].x + faces[i].width*0.5, faces[i].y + faces[i].height*0.5);
		maxareai = (faces[i].area()>maxarea) ? i : maxareai;
		maxarea = (faces[i].area()>maxarea) ? faces[i].area() : maxarea;
		ellipse(images, center, Size(faces[i].width*0.5, faces[i].height*0.5), 0, 0, 360, Scalar(255, 0, 255), 4, 8, 0);
	}
	if (faces.size() != 0) p = faces[maxareai];
	return p;
}

cv::Mat FaceDetector::removeFace(Rect faceRegion, Mat frame)
{
	int minThreshold = 30;
	int maxThreshold = 255;
	Mat faceRemove = frame.clone();
	faceRegion.width = faceRegion.width + faceRegion.width * 0.10;
	faceRegion.height = faceRegion.height + faceRegion.height * 0.10;
	for (int i = faceRegion.x; i < faceRegion.x + faceRegion.width && i < faceRemove.cols; i++)
	{
		for (int j = faceRegion.y; j < faceRegion.y + faceRegion.height && j < faceRemove.rows; j++)
		{
			faceRemove.at<Vec3b>(j, i)[0] = 0;
			faceRemove.at<Vec3b>(j, i)[1] = 0;
			faceRemove.at<Vec3b>(j, i)[2] = 0;
		}
	}

	//cvtColor(faceRemove, faceRemove, CV_BGR2GRAY);
	//GaussianBlur(faceRemove, faceRemove, Size(5, 5),0);
	//threshold(faceRemove, faceRemove, minThreshold, maxThreshold, CV_THRESH_BINARY_INV);
	return faceRemove;
}