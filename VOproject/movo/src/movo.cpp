
#include <cmath>
#include <string>
#include <iostream>

#include <eigen3/Eigen/Eigen>

#include <opencv2/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/video/tracking.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/core/eigen.hpp>


// #include <pcl/common/common_headers.h>
// #include <pcl/io/pcd_io.h>
// #include <pcl/point_types.h>
// #include <pcl/point_cloud.h>
// #include <pcl/visualization/pcl_visualizer.h>

#include "movo.h"
movo::movo(int argc, char **argv){
	readParams(argc, argv);
}
void movo::readParams(int argc, char **argv) {
	std::vector<cv::String> filenames_left;
	cv::String folder_left = argv[1];
	cv::glob(folder_left, filenames_left);

	std::string config_file = argv[2];
	cv::FileStorage fsSettings(config_file, cv::FileStorage::READ);
	if(!fsSettings.isOpened()){
		std::cerr<<("Failed to open")<<std::endl;
	}
	else {
		fsSettings["P0"] >> P_L;

		fsSettings["maxCorners"] >> maxCorners;
		fsSettings["qualityLevel"] >> qualityLevel;
		fsSettings["minDistance"] >> minDistance;
		fsSettings["blockSize"] >> blockSize;
		fsSettings["useHarrisDetector"] >> useHarrisDetector;
		fsSettings["k"] >> k;
		fsSettings["winSizeGFTT"] >> winSizeGFTT;

		fsSettings["fast_threshold"] >> fast_threshold;
		fsSettings["nonmaxSuppression"] >> nonmaxSuppression;
		fsSettings["winSizeFAST"] >> winSizeFAST;

		std::cout << "Parameters Loaded Successfully" << std::endl << std::endl;
	}


}

void movo::detectGoodFeatures(cv::Mat img, 
							  std::vector<cv::Point2f> &corners) {
	cv::TermCriteria termcrit = 
				cv::TermCriteria(cv::TermCriteria::COUNT+cv::TermCriteria::EPS, 30, 0.01);
	goodFeaturesToTrack(img, corners, maxCorners, qualityLevel, minDistance, cv::Mat(),
						blockSize, useHarrisDetector, k);
	cornerSubPix(img, corners, cv::Size(winSizeGFTT, winSizeGFTT), 
				 cv::Size(-1, -1), termcrit);
}

void movo::detectFASTFeatures(cv::Mat img, 
							  std::vector<cv::Point2f> &corners) {
	cv::TermCriteria termcrit = 
				cv::TermCriteria(cv::TermCriteria::COUNT+cv::TermCriteria::EPS, 30, 0.01);
	std::vector<cv::KeyPoint> keypoints;
	cv::FAST(img, keypoints, fast_threshold, nonmaxSuppression);
	cv::KeyPoint::convert(keypoints, corners, std::vector<int>());
	cornerSubPix(img, corners, cv::Size(winSizeFAST, winSizeFAST),
				 cv::Size(-1, -1), termcrit);
}

std::vector<uchar> movo::calculateOpticalFlow(cv::Mat img1, cv::Mat img2, 
							  				  std::vector<cv::Point2f> &corners1,
							  				  std::vector<cv::Point2f> &corners2) {
	cv::TermCriteria termcrit = cv::TermCriteria(
			cv::TermCriteria::COUNT+cv::TermCriteria::EPS, 30, 0.01);
	std::vector<uchar> status;
	std::vector<float> err;
	calcOpticalFlowPyrLK(img1, img2, corners1, corners2, status, err, 
						 cv::Size(2*winSizeGFTT + 1, 2*winSizeGFTT + 1), 
						 3, termcrit, 0, 0.001); 	
	return status;
}

cv::Mat movo::poseEstimation(std::vector<cv::Point2f> corners1,
					         std::vector<cv::Point2f> corners2,
					         cv::Mat &R, cv::Mat &t) {
	std::vector<cv::Point2f> corners1_ud, corners2_ud;
	cv::Mat K = P_L(cv::Range(0,2), cv::Range(0, 2));
	undistortPoints(corners1, corners1_ud, K, cv::noArray(), cv::noArray(), cv::noArray());
	undistortPoints(corners2, corners2_ud, K, cv::noArray(), cv::noArray(), cv::noArray());
	cv::Mat mask;
	cv::Mat essMat = findEssentialMat(corners2_ud, corners1_ud, 1.0, cv::Point2d(0.0, 0.0), 
							  cv::RANSAC, 0.99, 
							  10.0/(P_L.at<double>(0, 0)+P_L.at<double>(1, 1)), mask);
	recoverPose(essMat, corners2, corners1, R, t, 1.0, cv::Point2d(0.0, 0.0), mask);
	return mask;
}

int main(int argc, char **argv) {
	movo vo(argc, argv);

	return 0;
}
