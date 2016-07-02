#include "stdafx.h"
#include<opencv2\opencv.hpp>
#include "opencv2\objdetect.hpp"
#include"HandDetector.h"
#include"FaceDetector.h"
using namespace std;
using namespace cv;

Mat images;
int main()
{
	VideoCapture capture;
	//open capture object at location zero (default location for webcam)

	capture.open(0);

	//set height and width of capture frame
	capture.set(CV_CAP_PROP_FRAME_WIDTH, 480);
	capture.set(CV_CAP_PROP_FRAME_HEIGHT, 480);

	Mat cameraFeed, prevImage,image;

	HandDetector mySkinDetector;
	FaceDetector myFaceDetector;

	Point centerMat;
	Mat skinMat, edgeMat, motionMat, handMat,faceMat;
	Rect faceRegion;
	bool isFirstImage = true;

	while (1) {
		capture.read(cameraFeed);
		//flip(cameraFeed, cameraFeed, -1);
		if (isFirstImage != true) {
			if (!cameraFeed.empty()) {
				imshow("Original Image", cameraFeed);
			}
			myFaceDetector.images = cameraFeed.clone();
			image = cameraFeed.clone();
			faceRegion = myFaceDetector.detectFace(cameraFeed);
			imshow("kepala", myFaceDetector.images);

			faceMat = myFaceDetector.removeFace(faceRegion, cameraFeed);

			if (!faceMat.empty())
			{
				imshow("Remove Face", faceMat);
			}
			skinMat = mySkinDetector.getSkin(faceMat, false, true, true);
			if (!skinMat.empty()) {
				imshow("Skin Image", skinMat);
			}
			edgeMat = mySkinDetector.getEdge(cameraFeed);
			if (!edgeMat.empty()) {
				imshow("Edge Image", edgeMat);
			}
			motionMat = mySkinDetector.getMotion(cameraFeed, prevImage);
			if (!motionMat.empty()) {
				imshow("Motion Image", motionMat);
			}
			handMat = mySkinDetector.getHand(skinMat, edgeMat, motionMat);
			if (!handMat.empty()) {
				imshow("Hand Image", handMat);
			}
			centerMat = mySkinDetector.getHandCenterPoint(skinMat,image);
			//rectangle()
			circle(image, centerMat, 10, Scalar(255, 0, 0), -1, 0);

			imshow("center", image);

			waitKey(30);
		}
		else {
			isFirstImage = false;
		}
		prevImage = cameraFeed.clone();
		waitKey(10);
	}
	return 0;
}

