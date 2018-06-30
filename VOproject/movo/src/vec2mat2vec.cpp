#include <iostream>
#include "movo.h"

cv::Mat movo::vector2mat(cv::Point2f pt2d) {
	cv::Mat pt2d_mat=cv::Mat::zeros(2, 1, CV_64FC1);
	pt2d_mat.at<double>(0) = pt2d.x;
	pt2d_mat.at<double>(1) = pt2d.y;
	return pt2d_mat;
}

cv::Mat movo::vector2mat(cv::Point3f pt3d) {
	cv::Mat pt3d_mat=cv::Mat::zeros(3, 1, CV_64FC1);
	pt3d_mat.at<double>(0) = pt3d.x;
	pt3d_mat.at<double>(1) = pt3d.y;
	pt3d_mat.at<double>(2) = pt3d.z;
	return pt3d_mat;
}