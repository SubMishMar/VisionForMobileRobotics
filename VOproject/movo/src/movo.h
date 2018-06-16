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

    // Image containers and related Mats
    cv::Mat img1, img2;
    cv::Mat img1_out, img2_out;
    cv::Mat img1_ud, img2_ud;
    cv::Mat mask;
    cv::Mat point3d_homo;
    cv::Mat point3d_unhomo;

    //
    bool useFAST;

    // pose
    cv::Mat R, t;
public:
	//construtor
	movo(int, char**);

	//reads Params related to all functionalities
	void readParams(int, char**);

	// detects gftt
	void detectGoodFeatures(cv::Mat img, 
							std::vector<cv::Point2f> &corners);

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

	void filterbyStatus(std::vector<uchar> status,
					    std::vector<cv::Point2f> &corners1,
					    std::vector<cv::Point2f> &corners2);
	//Drawmatches
	void drawmatches(cv::Mat img1, cv::Mat img2, 
					 std::vector<cv::Point2f> corners1,
					 std::vector<cv::Point2f> corners2);

	//Convert Homogenous 3d pts to non-homogenous
	cv::Mat convertFromHomogeneous(cv::Mat p3h);

	//Continous VO operation
	void continousOperation(uint, std::vector<cv::Point2f>, cv::Mat);
};