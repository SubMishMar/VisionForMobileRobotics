#include <iostream>
#include "movo.h"

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

void movo::drawTrajectory(cv::Mat t, cv::Mat &traj) {
    int x = int(t.at<double>(0)) + 750;
	int y = int(t.at<double>(2)) + 750;
	circle(traj, cv::Point(y, x), 1, CV_RGB(255, 0, 0), 2);
	imshow( "Trajectory", traj);
	cv::waitKey(30);
}


void movo::drawLandmarks(std::vector<cv::Point3f> landmarks) {
  	viewer.removeAllShapes();
    viewer.removeAllPointClouds();
	viewer.setBackgroundColor (255, 255, 255);
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZRGB>);
	cloud->points.resize(landmarks.size());


	for(int i = 0; i < landmarks.size(); i++) {
	    pcl::PointXYZRGB &point = cloud->points[i];
	    point.x = landmarks[i].x;
	    point.y = landmarks[i].y;
	    point.z = landmarks[i].z;
	    point.r = 0;
	    point.g = 0;
	    point.b = 255;
	}
  	viewer.addPointCloud(cloud, "Triangulated Point Cloud");
  	viewer.setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE,
                                            3,
											"Triangulated Point Cloud");
  	viewer.addCoordinateSystem (1.0);
  	//viewer.setCameraPosition(0,0,0,-1,0,0,0,0,0);
  	viewer.spinOnce(100);
}