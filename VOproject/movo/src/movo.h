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
struct keypoint
{
	cv::Point2f pt;
	uint id0;
	cv::Mat M0;
};

class movo{
private:
	// for GFTT 
	int maxCorners;
  	double qualityLevel;
  	double minDistance;
  	int blockSize;
  	bool useHarrisDetector;
	double k ;
	int winSizeGFTT;

	// Calibration Matrix
	cv::Mat P_L, K;

	// for FAST
    int fast_threshold;
    bool nonmaxSuppression;
    int winSizeFAST;


    // folder and file names
    std::vector<cv::String> filenames_left;
    cv::String folder_left;
    std::string config_file; 

    // Image containers and related Mats and Vectors
    cv::Mat img1, img2;
    cv::Mat img1_out, img2_out;
    cv::Mat img1_ud, img2_ud;
    cv::Mat mask;
    //
    bool useFAST;

    // pose
    cv::Mat R, t, Rpnp;
public:
	//construtor
	movo(int, char**);

	//reads Params related to all functionalities
	void readParams(int, char**);

	// detects gftt
	void detectGoodFeatures(cv::Mat img, 
							std::vector<cv::Point2f> &corners,
							cv::Mat mask_mat);

	//detects FAST features
	void detectFASTFeatures(cv::Mat img, 
							std::vector<cv::Point2f> &corners);

	//Caluclates optical flow and returns a status to represent points
	//for which a valied tracked point has been found
	std::vector<uchar> calculateOpticalFlow(cv::Mat img1, cv::Mat img2, 
							  				std::vector<cv::Point2f> &corners1,
							  				std::vector<cv::Point2f> &corners2);

	//Estimates R, t from tracked feature points and returns an inlier mask
	cv::Mat epipolarSearch(std::vector<cv::Point2f> corners1,
					       std::vector<cv::Point2f> corners2,
					       cv::Mat &R, cv::Mat &t);

	//Initialization Procedure
	void initialize(uint, uint);

	// Filter corners 
	void filterbyMask(cv::Mat mask,
					  std::vector<cv::Point2f> &corners1,
					  std::vector<cv::Point2f> &corners2);

	void filterbyMask(cv::Mat mask,
					  std::vector<cv::Point2f> &corners1,
					  std::vector<cv::Point2f> &corners2,
					  std::vector<cv::Point3f> &landmarks);

	void filterbyStatus(std::vector<uchar> status,
					    std::vector<cv::Point2f> &corners1,
					    std::vector<cv::Point2f> &corners2);

	void filterbyStatus(std::vector<uchar> status,
					    std::vector<cv::Point2f> &corners1,
					    std::vector<cv::Point2f> &corners2,
					    std::vector<cv::Point3f> &landmarks);

	void filterbyStatus(std::vector<uchar> status,
						std::vector<cv::Point2f> corners,
						std::vector<keypoint> &keypoints);

	void filterbyStatus(cv::Mat mask,
						std::vector<keypoint> &keypoints);

	void filterbyLmkZ(std::vector<cv::Point2f> &corners1,
					   std::vector<cv::Point2f> &corners2,
					   std::vector<cv::Point3f> &landmarks); 
					    //Filters points which result 
						 //in negative Z in landmarks
	//Drawmatches
	void drawmatches(cv::Mat img1, cv::Mat img2, 
					 std::vector<cv::Point2f> corners1,
					 std::vector<cv::Point2f> corners2);

	//Convert Homogenous 3d pts to non-homogenous
	void convertFromHomogeneous(cv::Mat p3h, std::vector<cv::Point3f> &p3uh);

	//Continous VO operation
	void continousOperation(uint, std::vector<cv::Point2f>, std::vector<cv::Point3f>);

	//vector2mat
	cv::Mat vector2mat(cv::Point3f);
	cv::Mat vector2mat(cv::Point2f);

	//Draw trajectory;
	void drawTrajectory(cv::Mat, cv::Mat);

	//
	void corners2keypoint(std::vector<cv::Point2f> src,
						  std::vector<keypoint> &dst,
						  int,
						  cv::Mat);

	void selectnewPts(std::vector<keypoint> &candidate_kp, 
					  std::vector<cv::Point2f> &candidate_corners,
					  int query_id,
					  cv::Mat M_current,
					  std::vector<cv::Point2f> &new_query_corners,
					  std::vector<cv::Point3f> &new_landmarks_3d);
};