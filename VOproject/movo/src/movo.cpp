#include <iostream>
#include <string>
#include <cmath>
#include <eigen3/Eigen/Eigen>
#include "opencv2/core.hpp"
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/video/tracking.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include <opencv2/core/eigen.hpp>


// Global Variables
cv::Mat P_L;
cv::Mat image_L_i, image_L_i_ud;
cv::Mat image_L_j, image_L_j_ud;
cv::Mat image_L_i_out, image_L_j_out;
double c_x, c_y, f, T_x;
cv::Mat essMat;

cv::Mat Rw, tw;
cv::Mat Rij, tij;


double findMatchedPoints(cv::Mat img_1, 
					   cv::Mat img_2,
					   std::vector<cv::Point2f> &corners_1, 
					   std::vector<cv::Point2f> &corners_2){
	cv::TermCriteria termcrit = cv::TermCriteria(cv::TermCriteria::COUNT+cv::TermCriteria::EPS, 
								30, 0.01);
    std::vector<cv::KeyPoint> keypoints_1;
    int fast_threshold = 20;
    bool nonmaxSuppression = true;
    cv::Size winSize = cv::Size(10,10);
    cv::FAST(img_1, keypoints_1, fast_threshold, nonmaxSuppression);
	cv::KeyPoint::convert(keypoints_1, corners_1, std::vector<int>());
	cornerSubPix(img_1, corners_1, winSize, cv::Size(-1,-1), termcrit);
	std::vector<uchar> status;
	std::vector<float> err;
	calcOpticalFlowPyrLK(img_1, img_2, corners_1, corners_2, status, err, 
						cv::Size(2*winSize.width+1, 2*winSize.height+1), 
						3, termcrit, 0, 0.001);
    size_t i, j;
    double diff = 0;
    for( i = 0; i < corners_1.size(); i++ ){
      if (!status[i]) continue;
      corners_1[j] = corners_1[i];
      corners_2[j] = corners_2[i];
      diff += sqrt((corners_1[j].x - corners_2[j].x)*(corners_1[j].x - corners_2[j].x)
      		+ (corners_1[j].y - corners_2[j].y)*(corners_1[j].y - corners_2[j].y));    
      j++;
    }
    diff = diff/j;
    std::cout << diff << std::endl;
    corners_1.resize(j);
    corners_2.resize(j);
    return diff;
}

double findTrackedPoints(cv::Mat img_1, 
					   cv::Mat img_2,
					   std::vector<cv::Point2f> &corners_1, 
					   std::vector<cv::Point2f> &corners_2){
	std::vector<float> err;
	std::vector<uchar> status;
  	
  	int maxCorners = 1000;
  	double qualityLevel = 0.001;
  	double minDistance = 20;
  	int blockSize = 3;
  	bool useHarrisDetector = false;
  	double k = 0.04;	

  	cv::Size winSize = cv::Size(10,10);
  	cv::TermCriteria termcrit = cv::TermCriteria(cv::TermCriteria::COUNT+cv::TermCriteria::EPS, 30, 0.01);
  	goodFeaturesToTrack( img_1, corners_1, maxCorners, qualityLevel, minDistance, cv::Mat(), 
  		blockSize, useHarrisDetector, k );
  	cornerSubPix( img_1, corners_1, winSize, cv::Size(-1,-1), termcrit);
  	calcOpticalFlowPyrLK( img_1, img_2, corners_1, corners_2, 
  		status, err, cv::Size(2*winSize.width+1, 2*winSize.height+1), 3, termcrit, 1, 0.001);
    size_t i, j;
    double diff = 0;
    for( i = j = 0; i < corners_1.size(); i++ ){
      if (!status[i]) continue;
      corners_1[j] = corners_1[i];
      corners_2[j] = corners_2[i];
      diff += sqrt((corners_1[j].x - corners_2[j].x)*(corners_1[j].x - corners_2[j].x)
      		+ (corners_1[j].y - corners_2[j].y)*(corners_1[j].y - corners_2[j].y));    
      j++;
    }
    diff = diff/j;
    std::cout << diff << std::endl;
    corners_1.resize(j);
    corners_2.resize(j);
    return diff;
}

int main(int argc, char **argv){
	std::vector<cv::String> filenames_left;
	cv::String folder_left = argv[1];
	cv::glob(folder_left, filenames_left);

	std::string config_file = argv[2];
	cv::FileStorage fsSettings(config_file, cv::FileStorage::READ);
	if(!fsSettings.isOpened()){
		std::cerr<<("Failed to open")<<std::endl;
	}
	
	fsSettings["P0"] >> P_L;

	c_x = P_L.at<double>(0, 2);
	c_y = P_L.at<double>(1, 2);
	f = P_L.at<double>(0, 0);

	cv::Mat K = cv::Mat::zeros(3, 3, CV_64F);
	K.at<double>(0, 0) = f;
	K.at<double>(0, 2) = c_x;
	K.at<double>(1, 1) = f;
	K.at<double>(1, 2) = c_y;
	K.at<double>(2, 2) = 1.0;
    
	cv::Scalar color= cv::Scalar( 0, 0, 255);

	cv::namedWindow("features_i");
	cv::namedWindow("features_j");
	cv::namedWindow("Trajectory");
	cv::Mat traj =  cv::Mat::zeros(1500, 1500, CV_8UC3);

	Rw = cv::Mat::eye(3, 3, CV_64F);
	tw = cv::Mat::zeros(3, 1, CV_64F);

	int x = 0, y = 0;	
	for(int i = 0; i < 2000/*filenames_left.size()-1*/; i++){
		image_L_i = imread(filenames_left[i], CV_8UC1);
		image_L_j = imread(filenames_left[i+1], CV_8UC1);
		undistort(image_L_i, image_L_i_ud, K, cv::noArray(), K);
		undistort(image_L_j, image_L_j_ud, K, cv::noArray(), K);

		std::vector<cv::Point2f> corners_i, corners_i_ud;
		std::vector<cv::Point2f> corners_j, corners_j_ud;
		
		cv::cvtColor(image_L_i, image_L_i_out, CV_GRAY2BGR);
		cv::cvtColor(image_L_j, image_L_j_out, CV_GRAY2BGR);

		double avg_error = findMatchedPoints(image_L_i, image_L_j, corners_i, corners_j);
		if(avg_error>=5){
			undistortPoints(corners_i, corners_i_ud, K, cv::noArray(), cv::noArray(), cv::noArray());
			undistortPoints(corners_j, corners_j_ud, K, cv::noArray(), cv::noArray(), cv::noArray());

			cv::Mat mask;
			essMat = findEssentialMat(corners_j_ud, corners_i_ud, 1.0, cv::Point2d(0.0, 0.0), cv::RANSAC,
									 0.99, 10.0/(P_L.at<double>(0, 0)+P_L.at<double>(1, 1)), mask);
			recoverPose(essMat, corners_j_ud, corners_i_ud, Rij, tij, 1.0, cv::Point2d(0.0, 0.0), mask);
		}
		else {
			tij = cv::Mat::zeros(3, 1, CV_64F);
			Rij = cv::Mat::eye(3, 3, CV_64F);
		}

		tw = Rw*tij + tw;
		Rw = Rw*Rij;

		x = int(tw.at<double>(0)) + 750;
		y = int(tw.at<double>(2)) + 750;	

		cv::circle(traj, cv::Point(y, x), 1, CV_RGB(255, 0, 0), 2);

/*		std::cout << tij << std::endl;
		std::cout << Rij << std::endl << std::endl;*/

		for(int l = 0; l < corners_i.size(); l++){
			cv::circle(image_L_i_out, corners_i[l], 4, color, -1, 8, 0);
			cv::circle(image_L_j_out, corners_j[l], 4, color, -1, 8, 0);
		}
		imshow("features_i", image_L_i_out);	
		cv::waitKey(30);
		imshow("features_j", image_L_j_out);
		cv::waitKey(30);	
		imshow("Trajectory", traj);
		cv::waitKey(30);	
		corners_i.clear();
		corners_j.clear();
	}
	imshow("features_i", image_L_i_out);	
	cv::waitKey(0);
	imshow("features_j", image_L_j_out);
	cv::waitKey(0);	
	imshow("Trajectory", traj);
	cv::waitKey(0);
	cv::destroyWindow("features_i");
	cv::destroyWindow("features_j");
	cv::destroyWindow("Trajectory");
	return 0;
}
