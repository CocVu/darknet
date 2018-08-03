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

/* void convert_image_to_ipl(image src, cv::Mat big_picture, int x , int y); */
void convert_image_to_cvMat(image src, cv::Mat big_picture, int x , int y);

/* image convert_image_to_ipl(Image src, cv:Mat big_picture, int x , int y){ */
/* #endif */
