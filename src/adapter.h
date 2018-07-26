#include <stdlib.h>
#include <stdio.h>
#include <float.h>
#include <string.h>
#include <math.h>
#include "image.h"

/* #ifndef OPENCV */
#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/imgproc/imgproc.hpp>

image crop_mat_to_image(cv::Mat mat_crop);

image convert_ipl_to_image(IplImage* src);
/* #endif */
