/* #ifndef OPENCV */
extern "C"
{
#include "image.h"
}
#include "adapter.h"

#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/imgproc/imgproc.hpp>

#ifdef OPENCV
using namespace cv;

image convert_ipl_to_image(IplImage* src){
  unsigned char *data = (unsigned char *)src->imageData;
  int h = src->height;
  int w = src->width;
  int c = src->nChannels;
  int step = src->widthStep;
  image out = make_image(w, h, c);
  int i, j, k, count=0;;

  for(k= 0; k < c; ++k){
    for(i = 0; i < h; ++i){
      for(j = 0; j < w; ++j){
        out.data[count++] = data[i*step + j*c + k]/255.;
      }
    }
  }
  return out;
}

void convert_image_to_cvMat(image src, cv::Mat big_picture, int x , int y){
  // unsigned char *data = (unsigned char *)src->imageData;
  int h = src.h;
  int w = src.w;
  int c = src.c;
  // int step = src->widthStep;
  image out = make_image(w, h, c);
  int i, j, k, count=0;;

  for(k= 0; k < c; ++k){
    for(i = 0; i < h; ++i){
      for(j = 0; j < w; ++j){
        // TODO: exept for 256 or -1
        // -big_picture.at<int>( i*step + j*c + k ) = (int)(src.data[count++]*255);
        // big_picture.at<int>(y + i , x + j*c + k ) =0;
        big_picture.at<float>(y + i , x + k*c + j) =0;
        // (int)(src.data[count++]*255);
      }
    }
  }
  // return data;
}

image crop_mat_to_image(cv::Mat mat_crop)
{
  IplImage old = mat_crop;
  IplImage * src  = &old;
  image out = convert_ipl_to_image(src);
  if (out.c > 1)
    rgbgr_image(out);
  return out;
}

#endif
