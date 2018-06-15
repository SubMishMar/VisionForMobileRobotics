#include <iostream>
#include "movo.h"

movo::movo(int argc, char **argv){
	readParams(argc, argv);
	K = P_L(cv::Range(0,3), cv::Range(0, 3));
}
void movo::readParams(int argc, char **argv) {
	folder_left = argv[1];
	cv::glob(folder_left, filenames_left);
    config_file = argv[2];
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

		fsSettings["useFAST"] >> useFAST;
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
	undistortPoints(corners1, corners1_ud, K, cv::noArray(), cv::noArray(), cv::noArray());
	undistortPoints(corners2, corners2_ud, K, cv::noArray(), cv::noArray(), cv::noArray());
	cv::Mat mask;
	cv::Mat essMat = findEssentialMat(corners1_ud, corners2_ud, 1.0, cv::Point2d(0.0, 0.0), 
							  cv::RANSAC, 0.99, 
							  10.0/(K.at<double>(0, 0)+K.at<double>(1, 1)), mask);
	recoverPose(essMat, corners1_ud, corners2_ud, R, t, 1.0, cv::Point2d(0.0, 0.0), mask);
	return mask;
}

void movo::filterbyMask(cv::Mat mask,
					    std::vector<cv::Point2f> &corners1,
					    std::vector<cv::Point2f> &corners2) {
	int j = 0;
	//mask.convertTo(mask, CV_64F);
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

void movo::drawmatches(cv::Mat img1, cv::Mat img2, 
				   	   std::vector<cv::Point2f> corners1,
					   std::vector<cv::Point2f> corners2) {
	
	cv::cvtColor(img1, img1_out, CV_GRAY2BGR);
	cv::cvtColor(img2, img2_out, CV_GRAY2BGR);
	for(int l = 0; l < corners1.size(); l++){
		cv::circle(img1_out, corners1[l], 4, CV_RGB(255, 0, 0), -1, 8, 0);
		cv::circle(img2_out, corners2[l], 4, CV_RGB(255, 0, 0), -1, 8, 0);
	}	
	imshow("img1", img1_out);
	cv::waitKey(0);
	imshow("img2", img2_out);
	cv::waitKey(0);
}

cv::Mat movo::convertFromHomogeneous(cv::Mat p3h) {
	cv::Mat p3d = cv::Mat::zeros(p3h.rows-1, p3h.cols, CV_64FC1);
	for(int i = 0; i < p3h.cols; i++) {
	  cv::Mat p3d_col_i;
	  cv::Mat p3h_col_i = p3h.col(i);
	  convertPointsFromHomogeneous(p3h_col_i.t(), p3d_col_i);
	  p3d.at<double>(0,i) = p3d_col_i.at<double>(0);
	  p3d.at<double>(1,i) = p3d_col_i.at<double>(1);
	  p3d.at<double>(2,i) = p3d_col_i.at<double>(2);
	}
	return p3d;
}
void movo::initialize(uint frame1, uint frame2) {
	img1 = imread(filenames_left[frame1], CV_8UC1);
	undistort(img1, img1_ud, K, cv::noArray(), K);
	img2 = imread(filenames_left[frame2], CV_8UC1);
	undistort(img2, img2_ud, K, cv::noArray(), K);

	std::vector<cv::Point2f> corners1, corners2;
	std::vector<uchar> status;
	if(false) {
		detectFASTFeatures(img1_ud, corners1);
	} else {
		detectGoodFeatures(img1_ud, corners1);
	}
	status = calculateOpticalFlow(img1_ud, img2_ud, corners1, corners2);
	filterbyStatus(status, corners1, corners2);
	mask = poseEstimation(corners1, corners2, R, t);
	filterbyMask(mask, corners1, corners2);
	drawmatches(img1_ud, img2_ud, corners1, corners2);
	std::vector<cv::Point2f> corners1_ud, corners2_ud;
	undistortPoints(corners1, corners1_ud, K, cv::noArray(), cv::noArray(), cv::noArray());
	undistortPoints(corners2, corners2_ud, K, cv::noArray(), cv::noArray(), cv::noArray());
	std::vector<cv::Point2d> triangulation_pts1, triangulation_pts2;
	for(int i = 0; i < corners1.size(); i++) {
		triangulation_pts1.push_back
							(cv::Point2d((double)corners1_ud[i].x, (double)corners1_ud[i].y));
		triangulation_pts2.push_back
							(cv::Point2d((double)corners2_ud[i].x, (double)corners2_ud[i].y));
	}

	cv::Mat M0 = cv::Mat::eye(3, 4, CV_64FC1);
	cv::Mat M1 = cv::Mat::eye(3, 4, CV_64FC1);
	R.copyTo(M1.rowRange(0, 3).colRange(0, 3));
	t.copyTo(M1.rowRange(0, 3).col(3));
	cv::triangulatePoints(M0, M1, triangulation_pts1, triangulation_pts2, point3d_homo);
	point3d_unhomo = convertFromHomogeneous(point3d_homo);
}



