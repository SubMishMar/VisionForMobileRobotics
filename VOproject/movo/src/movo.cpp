#include <iostream>
#include "movo.h"

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

void movo::selectnewPts(std::vector<keypoint> &candidate_kp, 
						std::vector<cv::Point2f> &candidate_corners,
						int query_id,
						cv::Mat M_current,
						std::vector<cv::Point2f> &new_query_corners,
						std::vector<cv::Point3f> &new_landmarks_3d) {
	new_query_corners.clear();
	new_landmarks_3d.clear();
	size_t size = candidate_kp.size();
	new_query_corners = candidate_corners;
	std::vector<cv::Point2f> corners1_ud;
	std::vector<cv::Point2f> corners2_ud;
	std::vector<cv::Point2f> candidate_corners_0, diff;
	for(int i = 0; i < candidate_kp.size(); i++) {
		candidate_corners_0.push_back(candidate_kp[i].pt);
	}

	undistortPoints(candidate_corners_0, corners1_ud, K, 
		cv::noArray(), cv::noArray(), cv::noArray());
	undistortPoints(new_query_corners, corners2_ud, K, 
		cv::noArray(), cv::noArray(), cv::noArray());

	std::vector<cv::Point2d> triangulation_pts1, triangulation_pts2;
	for(int i = 0; i < candidate_corners_0.size(); i++) {
		triangulation_pts1.push_back
							(cv::Point2d((double)corners1_ud[i].x, 
								(double)corners1_ud[i].y));
		triangulation_pts2.push_back
							(cv::Point2d((double)corners2_ud[i].x, 
								(double)corners2_ud[i].y));
	}
	cv::Mat point3d_homo;
	cv::triangulatePoints(candidate_kp[0].M0 , M_current, 
		triangulation_pts1, triangulation_pts2, point3d_homo);
	
	convertFromHomogeneous(point3d_homo, new_landmarks_3d);

    int j = 0;
	for(int i = 0; i < new_landmarks_3d.size(); i++) {
		if(new_landmarks_3d[i].z < 0) {continue;}
		new_landmarks_3d[j] = new_landmarks_3d[i];
		new_query_corners[j] = new_query_corners[i];
		j++;
	}
	new_landmarks_3d.resize(j);
	new_query_corners.resize(j);

	candidate_kp.clear();
	candidate_corners.clear();
}

void movo::initialize(uint frame1, uint frame2) {
	cv::Mat point3d_homo;
    std::vector<cv::Point3f> point3d_unhomo;
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
	cv::Mat database_img, query_img;
	undistort(imread(filenames_left[database_id], CV_8UC1), 
				database_img, K, cv::noArray(), K);

    cv::Mat point3d_homo;
    std::vector<cv::Point3f> point3d_unhomo;
    std::vector<keypoint> candidate_kp;
    std::vector<cv::Point2f> candidate_corners;
	cv::Mat rvec, tvec;
	std::vector<cv::Point3f> rvecs, tvecs;
	std::vector<cv::Point3f> new_landmarks_3d;
	std::vector<cv::Point2f> query_corners, new_query_corners;
	std::vector<cv::Point2f> new_candidate_corners;
	std::vector<cv::Point2f> candidate_corners_j;

	
	int count = 0;
	bool new_triangulation = false;
	cv::Mat traj = cv::Mat::zeros(1500, 1500, CV_8UC3);
	while(query_id < filenames_left.size()) {
		cv::Mat M_current = cv::Mat::eye(3, 4, CV_64FC1);
		undistort(imread(filenames_left[query_id], CV_8UC1), 
					query_img, K, cv::noArray(), K);
		std::vector<uchar> status1, status2;
		status2 = calculateOpticalFlow(database_img,  query_img,
									  database_corners, query_corners);

		filterbyStatus(status2, database_corners, query_corners, landmarks_3d);

		drawmatches(database_img, query_img, database_corners, query_corners);

		std::vector<int> inliers;
		solvePnPRansac(landmarks_3d, query_corners, K, cv::noArray(), rvec, tvec, 
					    false, 100, 8, 0.99, inliers, cv::SOLVEPNP_P3P);
		Rodrigues(rvec, Rpnp, cv::noArray());

		Rpnp.copyTo(M_current.rowRange(0, 3).colRange(0, 3));
		tvec.copyTo(M_current.rowRange(0, 3).col(3));
		
		std::cout << database_corners.size() << "\t" 
		          << query_corners.size() << "\t"
		          << candidate_corners.size() << "\t"
		          << landmarks_3d.size() << std::endl;
		          
		std::cout << (-Rpnp.inv()*tvec).t() << std::endl;
		//drawTrajectory((-Rpnp.inv()*tvec), traj);
		// if(candidate_corners.size()>10) {
		// 	status1 = calculateOpticalFlow(database_img,  query_img,
		// 								  candidate_corners, candidate_corners_j);
		// 	filterbyStatus(status1, candidate_corners_j, candidate_kp);		
		// 	filterbyStatus(status1, candidate_corners, candidate_corners_j);
		// 	candidate_corners = candidate_corners_j;
		// 	selectnewPts(candidate_kp, candidate_corners, query_id, M_current, 
		// 				new_query_corners, new_landmarks_3d);
		// 	query_corners.insert(query_corners.end(), new_query_corners.begin(), 
 	//     							new_query_corners.end());
		// 	landmarks_3d.insert(landmarks_3d.end(), new_landmarks_3d.begin(), 
		// 					new_landmarks_3d.end());
		// }	

		drawLandmarks(landmarks_3d);
		// cv::Mat mask_mat(query_img.size(), CV_8UC1, cv::Scalar::all(255));
		// cv::Mat mask_mat_color;
		// cv::cvtColor(mask_mat, mask_mat_color, CV_GRAY2BGR);
		// std::vector<cv::Point2f> queryPlusCandidateCorners(query_corners.size()
		// 								+candidate_corners.size());


		// queryPlusCandidateCorners.insert(queryPlusCandidateCorners.end(),
		// 						query_corners.begin(), query_corners.end());
		// queryPlusCandidateCorners.insert(queryPlusCandidateCorners.end(),
		// 						candidate_corners.begin(), candidate_corners.end());

		// for(int i = 0; i < queryPlusCandidateCorners.size(); i++) {
		// 	cv::circle(mask_mat_color, queryPlusCandidateCorners[i], 
		// 		15, CV_RGB(0,0,0), -8, 0);
		// }
		// cv::cvtColor(mask_mat_color, mask_mat, CV_BGR2GRAY);
		
		// detectGoodFeatures(query_img, new_candidate_corners, mask_mat);
       
		// corners2keypoint(new_candidate_corners, candidate_kp, 
		// 					query_id, M_current);

		// candidate_corners.insert(candidate_corners.end(), new_candidate_corners.begin(), 
 	//     					new_candidate_corners.end());


		database_corners = query_corners;
		query_img.copyTo(database_img);
		query_id++;
	}
	// cv::destroyWindow("img1");
	// cv::destroyWindow("img2");
}


