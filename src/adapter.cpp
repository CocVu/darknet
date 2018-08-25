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

// image ipl_to_image(IplImage* src) // image.c:937
void draw_image_to_cvMat(image src, cv::Mat big_picture, int x , int y){
  int h = src.h;
  int w = src.w;
  int c = src.c;
  image out = make_image(w, h, c);
  int i, j, k, count=0;;
  unsigned char tmp_val;
  for(k= 0; k < c; ++k){
    for(i = 0; i < h; ++i){
      for(j = 0; j < w; ++j){
        // DONE: Convert RGB to gray
        tmp_val = (unsigned char)(src.data[count++]*255);

        // for 3 channels
        big_picture.at<unsigned char>(y + i ,(x + k+ j)*c ) = tmp_val;
        big_picture.at<unsigned char>(y + i ,(x + k+ j)*c +1 ) = tmp_val;
        big_picture.at<unsigned char>(y + i ,(x + k+ j)*c +2 ) = tmp_val;
      }
    }
  }
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
