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
	cv::Mat P_L;

	// for FAST
    int fast_threshold;
    bool nonmaxSuppression;
    int winSizeFAST;

public:
	movo();
	~movo();
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
	cv::Mat poseEstimation(std::vector<cv::Point2f> corners1,
					       std::vector<cv::Point2f> corners2,
					       cv::Mat &R, cv::Mat &t);
};