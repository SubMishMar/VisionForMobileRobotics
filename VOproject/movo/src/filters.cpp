#include <iostream>
#include "movo.h"

void movo::filterbyMask(cv::Mat mask,
					    std::vector<cv::Point2f> &corners1,
					    std::vector<cv::Point2f> &corners2) {
	int j = 0;
	for(int i = 0; i < corners1.size(); i++) {
		if(!mask.at<unsigned char>(i)) {
			continue;
		}
		corners1[j] = corners1[i];
		corners2[j] = corners2[i];
		j++;
	}
	corners1.resize(j);
	corners2.resize(j);	
}

void movo::filterbyMask(cv::Mat mask,
					    std::vector<cv::Point2f> &corners1,
					    std::vector<cv::Point2f> &corners2,
					    std::vector<cv::Point3f> &landmarks) {
	int j = 0;
	for(int i = 0; i < corners1.size(); i++) {
		if(!mask.at<unsigned char>(i)) {
			continue;
		}
		corners1[j] = corners1[i];
		corners2[j] = corners2[i];
		landmarks[j] = landmarks[i];
		j++;
	}
	corners1.resize(j);
	corners2.resize(j);	
	landmarks.resize(j);
}

void movo::filterbyStatus(std::vector<uchar> status,
					      std::vector<cv::Point2f> &corners1,
					      std::vector<cv::Point2f> &corners2) {
	int j = 0;
	for(int i = 0; i < status.size(); i++) {
		if(status[i] == 0 ||
		   corners2[i].x < 0 || corners2[i].y < 0 ||
		   corners2[i].x > img1.cols || corners2[i].y > img1.rows) continue;
		corners1[j] = corners1[i];
		corners2[j] = corners2[i];
		j++;
	}
	corners1.resize(j);
	corners2.resize(j);	
}

void movo::filterbyStatus(std::vector<uchar> status,
					      std::vector<cv::Point2f> &corners1,
					      std::vector<cv::Point2f> &corners2,
					      std::vector<cv::Point3f> &landmarks) {
	int j = 0;
	for(int i = 0; i < status.size(); i++) {
		if(status[i] == 0 ||
		   corners2[i].x < 0 || corners2[i].y < 0 ||
		   corners2[i].x > img1.cols || corners2[i].y > img1.rows) continue;
		corners1[j] = corners1[i];
		corners2[j] = corners2[i];
		landmarks[j] = landmarks[i];
		j++;
	}
	corners1.resize(j);
	corners2.resize(j);	
	landmarks.resize(j);
}

void movo::filterbyStatus(std::vector<uchar> status,
					std::vector<cv::Point2f> corners,
					std::vector<keypoint> &keypoints) {
	int j = 0;
	for(int i = 0; i < status.size(); i++) {
		if(status[i] == 0 ||
		   corners[i].x < 0 || corners[i].y < 0 ||
		   corners[i].x > img1.cols || corners[i].y > img1.rows) continue;
		keypoints[j] = keypoints[i];
		j++;
	}
	keypoints.resize(j);
}