/* #ifdef OPENCV */
#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/imgproc/imgproc.hpp>

class Blob {
	public:
		// member variables
		std::vector<cv::Point> currentContour;
		cv::Rect currentBoundingRect;
		std::vector<cv::Point> centerPositions;
		double dblCurrentDiagonalSize;
		double dblCurrentAspectRatio;
		bool blnCurrentMatchFoundOrNewBlob;
		bool blnStillBeingTracked;
		int intNumOfConsecutiveFramesWithoutAMatch;
    // TODO just one object in blob???
    char * current_blob_single_name;
		cv::Point predictedNextPosition;

		// function prototypes
		Blob(std::vector<cv::Point> _contour);
		void predictNextPosition(void);
};

/* #endif  */
