#include <iostream>
#include "movo.h"

movo::movo(int argc, char **argv){
	readParams(argc, argv);
	K = P_L(cv::Range(0,3), cv::Range(0, 3));
	useFAST = false;
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

		// fsSettings["useFAST"] >> useFAST;
		std::cout << "Parameters Loaded Successfully" << std::endl << std::endl;
	}
}

void movo::detectGoodFeatures(cv::Mat img, 
							  std::vector<cv::Point2f> &corners,
							  cv::Mat mask_mat) {
	cv::TermCriteria termcrit = 
				cv::TermCriteria(cv::TermCriteria::COUNT+cv::TermCriteria::EPS, 30, 0.01);
	goodFeaturesToTrack(img, corners, maxCorners, qualityLevel, minDistance, mask_mat,
						blockSize, useHarrisDetector, k);

	if(corners.size() > 0)
		cornerSubPix(img, corners, cv::Size(winSizeGFTT/2, winSizeGFTT/2), 
				 	cv::Size(-1, -1), termcrit);
}

void movo::detectFASTFeatures(cv::Mat img, 
							  std::vector<cv::Point2f> &corners) {
	cv::TermCriteria termcrit = 
				cv::TermCriteria(cv::TermCriteria::COUNT+cv::TermCriteria::EPS, 30, 0.01);
	std::vector<cv::KeyPoint> keypoints;
	cv::FAST(img, keypoints, fast_threshold, nonmaxSuppression);
	cv::KeyPoint::convert(keypoints, corners, std::vector<int>());
	if(corners.size()>0)
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
						 cv::Size(winSizeGFTT + 1, winSizeGFTT + 1), 
						 3, termcrit, 0, 0.001); 	
	return status;
}

cv::Mat movo::epipolarSearch(std::vector<cv::Point2f> corners1,
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

void movo::drawTrajectory(cv::Mat R, cv::Mat t) {

}

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

void movo::convertFromHomogeneous(cv::Mat p3h, std::vector<cv::Point3f> &p3uh) {
	for(int i = 0; i < p3h.cols; i++) {
	  cv::Mat p3d_col_i;
	  cv::Mat p3h_col_i = p3h.col(i);
	  convertPointsFromHomogeneous(p3h_col_i.t(), p3d_col_i);
	  float x = (float)p3d_col_i.at<double>(0);
	  float y = (float)p3d_col_i.at<double>(1);
	  float z = (float)p3d_col_i.at<double>(2);
	  p3uh.push_back(cv::Point3f(x, y, z));
	}
}

void movo::corners2keypoint(std::vector<cv::Point2f> corners,
					  std::vector<keypoint> &keypoints,
					  int frame,
					  cv::Mat M) {
	for(int i = 0; i < corners.size(); i++) {
		keypoint kpt;
		kpt.pt = corners[i];
		kpt.id0 = frame;
		kpt.M0 = M;
		keypoints.push_back(kpt); 
	}
}
void movo::initialize(uint frame1, uint frame2) {
	img1 = imread(filenames_left[frame1], CV_8UC1);
	undistort(img1, img1_ud, K, cv::noArray(), K);
	img2 = imread(filenames_left[frame2], CV_8UC1);
	undistort(img2, img2_ud, K, cv::noArray(), K);

	std::vector<cv::Point2f> corners1, corners2;
	std::vector<uchar> status;
	if(useFAST) {
		detectFASTFeatures(img1_ud, corners1);
	} else {
		detectGoodFeatures(img1_ud, corners1, cv::Mat::ones(img1_ud.rows, img1_ud.cols, CV_8UC1));
	}
	status = calculateOpticalFlow(img1_ud, img2_ud, corners1, corners2);
	filterbyStatus(status, corners1, corners2);
	mask = epipolarSearch(corners1, corners2, R, t);
	filterbyMask(mask, corners1, corners2);
	
	//drawmatches(img1_ud, img2_ud, corners1, corners2);
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
	corners2keypoint(corners1, candidate_kp, frame1, M0);
	cv::Mat M1 = cv::Mat::eye(3, 4, CV_64FC1);
	R.copyTo(M1.rowRange(0, 3).colRange(0, 3));
	t.copyTo(M1.rowRange(0, 3).col(3));
	cv::triangulatePoints(M0, M1, triangulation_pts1, triangulation_pts2, point3d_homo);
	convertFromHomogeneous(point3d_homo, point3d_unhomo);
	continousOperation(frame2, corners2, point3d_unhomo);
}

void movo::continousOperation(uint frame_id,
							  std::vector<cv::Point2f> corners,
							  std::vector<cv::Point3f> landmarks_3d) {
	uint database_id = frame_id;
	uint query_id = database_id + 1; 
	std::vector<cv::Point2f> database_corners = corners;
	std::vector<cv::Point2f> new_query_corners, new_database_corners;
	cv::Mat database_img, query_img;
	undistort(imread(filenames_left[database_id], CV_8UC1), 
				database_img, K, cv::noArray(), K);

	cv::Mat rvec, tvec;
	std::vector<cv::Point3f> rvecs, tvecs;
	std::vector<cv::Point3f> new_landmarks_3d;
	std::vector<cv::Point2f> query_corners;
	cv::Mat M_previous = cv::Mat::eye(3, 4, CV_64FC1);
	cv::Mat M_current = cv::Mat::eye(3, 4, CV_64FC1);
	int count = 0;
	bool new_triangulation = false;
	while(query_id < filenames_left.size()) {
		
		undistort(imread(filenames_left[query_id], CV_8UC1), 
					query_img, K, cv::noArray(), K);
		std::vector<uchar> status1, status2;
		// status1 = calculateOpticalFlow(database_img, query_img,
		// 							  new_database_corners, new_query_corners);
		// filterbyStatus(status1, new);
		status2 = calculateOpticalFlow(database_img,  query_img,
									  database_corners, query_corners);
		filterbyStatus(status2, database_corners, query_corners/*, landmarks_3d*/);
		
		// std::vector<double> diff(candidate_kp.size());
		// for(int i = 0; i < candidate_kp.size(); i++) {
		// 	diff.push_back(candidate_kp[i].id0-query_id);
		// 	std::cout << abs((int)candidate_kp[i].id0 - (int)query_id) << std::endl;
		// }

		// std::vector<int> inliers;
		// solvePnPRansac(landmarks_3d, query_corners, K, cv::noArray(), rvec, tvec, 
		// 			    false, 100, 8, 0.99, inliers, cv::SOLVEPNP_P3P);
		// Rodrigues(rvec, Rpnp, cv::noArray());
		//std::cout << (-Rpnp.inv()*tvec).t() << std::endl;
		//drawmatches(database_img, query_img, database_corners, query_corners);
		std::cout << database_corners.size() << " " << query_corners.size() << " " 
		/*<< candidate_kp.size() << */" " << query_id << std::endl;
		
		cv::Mat mask_mat(query_img.size(), CV_8UC1, cv::Scalar::all(255));
		cv::Mat mask_mat_color;
		cv::cvtColor(mask_mat, mask_mat_color, CV_GRAY2BGR);
		for(int i = 0; i < query_corners.size(); i++) {
			cv::circle(mask_mat_color, query_corners[i], 30, CV_RGB(0,0,0),-8,0);
		}
		cv::cvtColor(mask_mat_color, mask_mat, CV_BGR2GRAY);
		
		detectGoodFeatures(query_img, new_query_corners, mask_mat);
		corners2keypoint(new_query_corners, candidate_kp, 
							query_id, M_current);

		// query_corners.insert(query_corners.end(), new_query_corners.begin(), 
 	    // 					new_query_corners.end());

		database_corners = query_corners;
		new_database_corners = new_query_corners;
		query_img.copyTo(database_img);
		query_id++;
	}
	// cv::destroyWindow("img1");
	// cv::destroyWindow("img2");
}


