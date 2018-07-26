// opencv 2 lightweight -> dont have tracking
extern "C"
{
#include "network.h"
#include "region_layer.h"
#include "cost_layer.h"
#include "utils.h"
#include "parser.h"
#include "box.h"
#include "demo.h"
#include "option_list.h"

}
// #include "opencv2/highgui/highgui_c.h"
// #include "opencv2/core/core_c.h"
// #include "opencv2/imgproc/imgproc_c.h"
//#include "opencv2/core/core.hpp"
// #include "opencv2/core/version.hpp"

using namespace std;
#include "Blob.h"
#include "adapter.h"
#include <fstream>
#include <cstdlib>
#include <string>
#include <stdio.h>
#include <iomanip>
#pragma warning(disable : 4996)
#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/imgproc/imgproc.hpp>
#include<iostream>
#define SHOW_STEPS


// const global variables
const cv::Scalar SCALAR_BLACK = cv::Scalar(0.0, 0.0, 0.0);
const cv::Scalar SCALAR_WHITE = cv::Scalar(255.0, 255.0, 255.0);
const cv::Scalar SCALAR_YELLOW = cv::Scalar(0.0, 255.0, 255.0);
const cv::Scalar SCALAR_GREEN = cv::Scalar(0.0, 200.0, 0.0);
const cv::Scalar SCALAR_RED = cv::Scalar(0.0, 0.0, 255.0);


# define M_PI           3.14159265358979323846
std::stringstream date;
int carCountLeft, intVerticalLinePosition, carCountRight = 0;
// int left1_x, left1_y, left2_x, left2_y, right1_x, right1_y, right2_x, right2_y;

int left1_x = 100;
int left1_y = 500;
int left2_x = 500;
int left2_y = 500;
double left_alpha = M_PI/4;


int right1_x = 500;
int right1_y = 400;
int right2_x = 900;
int right2_y = 400;
double right_alpha = -M_PI/4;

double left_slope = (left2_y - left1_y) / (left2_x - left1_x);
double right_slope = (right2_y - right1_y) / (right2_x - right1_x);
int min_blob_area = 400;
// function prototypes
// void matchCurrentFrameBlobsToExistingBlobs(std::vector<Blob> &existingBlobs, std::vector<Blob> &currentFrameBlobs);
// void addBlobToExistingBlobs(Blob &currentFrameBlob, std::vector<Blob> &existingBlobs, int &intIndex);
// void addNewBlob(Blob &currentFrameBlob, std::vector<Blob> &existingBlobs);
// double distanceBetweenPoints(cv::Point point1, cv::Point point2);

// void drawAndShowContours(cv::Size imageSize, std::vector<std::vector<cv::Point> > contours, std::string strImageName);
// void drawAndShowContours(cv::Size imageSize, std::vector<Blob> blobs, std::string strImageName);
// bool checkIfBlobsCrossedTheLineRight(std::vector<Blob> &blobs, int &intHorizontalLinePosition, int &carCountRight);
// bool checkIfBlobsCrossedTheLineLeft(std::vector<Blob> &blobs, int &intHorizontalLinePositionLeft, int &carCountLeft);
// void drawBlobInfoOnImage(std::vector<Blob> &blobs, cv::Mat &imgFrame2Copy);
// void drawCarCountOnImage(int &carCountRight, cv::Mat &imgFrame2Copy);

void addBlobToExistingBlobs(Blob &currentFrameBlob, std::vector<Blob> &existingBlobs, int &intIndex) {
  existingBlobs[intIndex].currentContour = currentFrameBlob.currentContour;
  existingBlobs[intIndex].currentBoundingRect = currentFrameBlob.currentBoundingRect;
  existingBlobs[intIndex].centerPositions.push_back(currentFrameBlob.centerPositions.back());
  existingBlobs[intIndex].dblCurrentDiagonalSize = currentFrameBlob.dblCurrentDiagonalSize;
  existingBlobs[intIndex].dblCurrentAspectRatio = currentFrameBlob.dblCurrentAspectRatio;
  existingBlobs[intIndex].blnStillBeingTracked = true;
  existingBlobs[intIndex].blnCurrentMatchFoundOrNewBlob = true;
}


void addNewBlob(Blob &currentFrameBlob, std::vector<Blob> &existingBlobs) {
  currentFrameBlob.blnCurrentMatchFoundOrNewBlob = true;
  existingBlobs.push_back(currentFrameBlob);
}



double distanceBetweenPoints(cv::Point point1, cv::Point point2) {
  int intX = abs(point1.x - point2.x);
  int intY = abs(point1.y - point2.y);

  return(sqrt(pow(intX, 2) + pow(intY, 2)));
}
void matchCurrentFrameBlobsToExistingBlobs(std::vector<Blob> &existingBlobs, std::vector<Blob> &currentFrameBlobs) {
  for (auto &existingBlob : existingBlobs) {
    existingBlob.blnCurrentMatchFoundOrNewBlob = false;
    existingBlob.predictNextPosition();
  }

  for (auto &currentFrameBlob : currentFrameBlobs) {
    int intIndexOfLeastDistance = 0;
    double dblLeastDistance = 100000.0;

    for (unsigned int i = 0; i < existingBlobs.size(); i++) {

      if (existingBlobs[i].blnStillBeingTracked == true) {
        double dblDistance = distanceBetweenPoints(currentFrameBlob.centerPositions.back(), existingBlobs[i].predictedNextPosition);

        if (dblDistance < dblLeastDistance) {
          dblLeastDistance = dblDistance;
          intIndexOfLeastDistance = i;
        }
      }
    }

    if (dblLeastDistance < currentFrameBlob.dblCurrentDiagonalSize * 0.5) {
      addBlobToExistingBlobs(currentFrameBlob, existingBlobs, intIndexOfLeastDistance);
    }
    else {
      addNewBlob(currentFrameBlob, existingBlobs);
    }

  }

  for (auto &existingBlob : existingBlobs) {
    if (existingBlob.blnCurrentMatchFoundOrNewBlob == false) {
      existingBlob.intNumOfConsecutiveFramesWithoutAMatch++;
    }
    if (existingBlob.intNumOfConsecutiveFramesWithoutAMatch >= 5) {
      existingBlob.blnStillBeingTracked = false;
    }
  }
}



void drawAndShowContours(cv::Size imageSize, std::vector<std::vector<cv::Point> > contours, std::string strImageName) {
  cv::Mat image(imageSize, CV_8UC3, SCALAR_BLACK);
  cv::drawContours(image, contours, -1, SCALAR_WHITE, -1);
  cv::imshow(strImageName, image);
}


void drawAndShowContours(cv::Size imageSize, std::vector<Blob> blobs, std::string strImageName) {
  cv::Mat image(imageSize, CV_8UC3, SCALAR_BLACK);
  std::vector<std::vector<cv::Point> > contours;

  for (auto &blob : blobs) {
    if (blob.blnStillBeingTracked == true) {
      contours.push_back(blob.currentContour);
    }
  }

  cv::drawContours(image, contours, -1, SCALAR_WHITE, -1);
  cv::imshow(strImageName, image);
}


bool checkIfBlobsCrossedTheLineRight(std::vector<Blob> &blobs, int &intHorizontalLinePosition, int &carCountRight) {
  bool blnAtLeastOneBlobCrossedTheLine = false;

  for (auto blob : blobs) {
    if (blob.blnStillBeingTracked == true && blob.centerPositions.size() >= 2) {
      int prevFrameIndex = (int)blob.centerPositions.size() - 2;
      int currFrameIndex = (int)blob.centerPositions.size() - 1;

      // Left way
      if (blob.centerPositions[prevFrameIndex].y > left2_y && blob.centerPositions[currFrameIndex].y <= intHorizontalLinePosition && blob.centerPositions[currFrameIndex].x > left2_x) {
        carCountRight++;
        blnAtLeastOneBlobCrossedTheLine = true;
      }
    }
  }

  return blnAtLeastOneBlobCrossedTheLine;
}


bool checkIfBlobsCrossedTheLineLeft(std::vector<Blob> &blobs, int &intHorizontalLinePosition, int &carCountLeft) {

  bool blnAtLeastOneBlobCrossedTheLineLeft = false;

  for (auto blob : blobs) {
    if (blob.blnStillBeingTracked == true && blob.centerPositions.size() >= 2) {
      int prevFrameIndex = (int)blob.centerPositions.size() - 2;
      int currFrameIndex = (int)blob.centerPositions.size() - 1;

      // Left way
      if (blob.centerPositions[prevFrameIndex].y <= intHorizontalLinePosition && blob.centerPositions[currFrameIndex].y > intHorizontalLinePosition && blob.centerPositions[currFrameIndex].x < left2_x && blob.centerPositions[currFrameIndex].x > left1_x) {
        carCountLeft++;
        blnAtLeastOneBlobCrossedTheLineLeft = true;
      }
    }
  }

  return blnAtLeastOneBlobCrossedTheLineLeft;
}


void drawBlobInfoOnImage(std::vector<Blob> &blobs, cv::Mat &imgFrame2Copy) {
  for (unsigned int i = 0; i < blobs.size(); i++) {
    if (blobs[i].blnStillBeingTracked == true) {
      cv::rectangle(imgFrame2Copy, blobs[i].currentBoundingRect, SCALAR_RED, 2);

      int intFontFace = CV_FONT_HERSHEY_SIMPLEX;
      double dblFontScale = (imgFrame2Copy.rows * imgFrame2Copy.cols) / 300000.0;
      int intFontThickness = (int)::round(dblFontScale * 1.0);

      cv::putText(imgFrame2Copy, std::to_string(i), blobs[i].centerPositions.back(), intFontFace, dblFontScale, SCALAR_GREEN, intFontThickness);
    }
  }
}


void drawCarCountOnImage(int &carCountRight, cv::Mat &imgFrame2Copy) {
  int intFontFace = CV_FONT_HERSHEY_SIMPLEX;
  double dblFontScale = (imgFrame2Copy.rows * imgFrame2Copy.cols) / 450000.0;
  int intFontThickness = (int)::round(dblFontScale * 2.5);

  // Right way
  cv::Size textSize = cv::getTextSize(std::to_string(carCountRight), intFontFace, dblFontScale, intFontThickness, 0);
  cv::putText(imgFrame2Copy,"right" + std::to_string(carCountRight),cv::Point(right1_x,right1_y), intFontFace, dblFontScale, SCALAR_RED, intFontThickness);

  // //Left way
  cv::Size textSize1 = cv::getTextSize(std::to_string(carCountLeft), intFontFace, dblFontScale, intFontThickness, 0);
  cv::putText(imgFrame2Copy,"left" + std::to_string(carCountLeft), cv::Point(left1_x, left1_y), intFontFace, dblFontScale, SCALAR_YELLOW, intFontThickness);
}


image crop_blob(cv::Mat image_origin, cv::Rect rect){
  std::cout<< "____________________________________________________________________" << endl;
  // cv::Rect * new_rect = new cv::Rect;

  cv::Rect roi;
  int edge_square_max = rect.width;
  if(rect.width < rect.height)
    edge_square_max = rect.height;

  //DONE: square scale rect.x + max(width ...) <= 1280 and height <= 720
  roi.x = rect.x;
  roi.y = rect.y;
  roi.width = rect.width;
  roi.height = rect.height;
  if(rect.width < edge_square_max && (rect.x + edge_square_max) <= 1280)
    roi.height = edge_square_max;
  else if(rect.height < edge_square_max && (rect.y + edge_square_max) <= 720)
    roi.width= edge_square_max;

  cout << roi.height << "\t" << roi.width<< endl;
  cout << roi.x << "\t" << roi.y<< endl;
  cout << roi.height + roi.x<< "\t" << roi.width + roi.y<< endl;
  // cv::Mat mat_crop = image_origin(rect); //crop in matrix

  cv::Mat mat_crop = image_origin(roi); //crop in matrix
  image image_crop = crop_mat_to_image(mat_crop);


  // delete &roi;
  return image_crop;
}

int main(int argc, char *argv[]) {

  string video;
  char *datacfg = "list/obj.data";
  char *cfgfile = "list/cpu_2_yolo.cfg";
  char *weightfile = "list/cpu_2_yolo_50000.weights";
  float thresh = .5;
  float hier_thresh = 1.0;
  cv::VideoCapture capVideo;
  cv::Mat imgFrame1;
  cv::Mat imgFrame2;
  std::vector<Blob> blobs;
  cv::Point crossingLine[2];
  cv::Point crossingLineLeft[2];

  switch(argc){
  case 1:
    video = "/home/nam/Videos/TruongKimDong2.mp4";
    break;
  case 2:
    video = argv[1];
    break;
  case 3:
    video = argv[1];
    min_blob_area = atoi(argv[2]);
    break;
  default :
    break;
  }

  capVideo.open(video);

  if (!capVideo.isOpened()) {
    std::cout << "error reading video file" << std::endl << std::endl;
    return(0);
  }

  capVideo.read(imgFrame1);
  capVideo.read(imgFrame2);

  // using ::round instead std::round
  int intHorizontalLinePosition = (int)::round((double)imgFrame1.rows * 0.35);
  intHorizontalLinePosition = intHorizontalLinePosition*1.40;
  intVerticalLinePosition = (int)::round((double)imgFrame1.cols * 0.35);


  crossingLine[0].x = right1_x;
  crossingLine[0].y = right1_y;

  crossingLine[1].x = right2_x;
  crossingLine[1].y = right2_y;

  crossingLineLeft[0].x = left1_x;
  crossingLineLeft[0].y = left1_y;

  crossingLineLeft[1].x = left2_x;
  crossingLineLeft[1].y = left2_y;

  char chCheckForEscKey = 0;
  bool blnFirstFrame = true;
  int frameCount = 2;

  list *options = read_data_cfg(datacfg);
  char *name_list = option_find_str(options, "names", "data/names.list");
  int names_size = 0;
  char **names = get_labels_custom(name_list, &names_size); //get_labels(name_list);
  image **alphabet = load_alphabet();

  network net = parse_network_cfg_custom(cfgfile, 1); // set batch=1
  if(weightfile){
    load_weights(&net, weightfile);
  }
  fuse_conv_batchnorm(net);

  if (net.layers[net.n - 1].classes != names_size) {
    printf(" Error: in the file %s number of names %d that isn't equal to classes=%d in the file %s \n",
           name_list, names_size, net.layers[net.n - 1].classes, cfgfile);
    if(net.layers[net.n - 1].classes > names_size) getchar();
  }

  while (capVideo.isOpened() && chCheckForEscKey != 27 && chCheckForEscKey !='q') {

    int letterbox = 0;
    std::vector<Blob> currentFrameBlobs;
    cv::Mat imgFrame1Copy = imgFrame1.clone();
    cv::Mat imgFrame2Copy = imgFrame2.clone();


    cv::Mat imgDifference;
    cv::Mat imgThresh;
    cv::cvtColor(imgFrame1Copy, imgFrame1Copy, CV_BGR2GRAY);
    cv::cvtColor(imgFrame2Copy, imgFrame2Copy, CV_BGR2GRAY);
    cv::GaussianBlur(imgFrame1Copy, imgFrame1Copy, cv::Size(5, 5), 0);
    cv::GaussianBlur(imgFrame2Copy, imgFrame2Copy, cv::Size(5, 5), 0);
    cv::absdiff(imgFrame1Copy, imgFrame2Copy, imgDifference);
    cv::threshold(imgDifference, imgThresh, 30, 255.0, CV_THRESH_BINARY);

    cv::imshow("imgThresh", imgThresh);
    cv::Mat structuringElement3x3 = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
    cv::Mat structuringElement5x5 = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));
    cv::Mat structuringElement7x7 = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(7, 7));
    cv::Mat structuringElement15x15 = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(15, 15));

    for (unsigned int i = 0; i < 2; i++) {
      cv::dilate(imgThresh, imgThresh, structuringElement5x5);
      cv::dilate(imgThresh, imgThresh, structuringElement5x5);
      cv::erode(imgThresh, imgThresh, structuringElement5x5);
    }

    cv::Mat imgThreshCopy = imgThresh.clone();
    std::vector<std::vector<cv::Point> > contours;
    cv::findContours(imgThreshCopy, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    drawAndShowContours(imgThresh.size(), contours, "imgContours");

    std::vector<std::vector<cv::Point> > convexHulls(contours.size());

    for (unsigned int i = 0; i < contours.size(); i++) {
      cv::convexHull(contours[i], convexHulls[i]);
    }

    drawAndShowContours(imgThresh.size(), convexHulls, "imgConvexHulls");

    // std::cout << convexHulls.size();
    // printf("%d", convexHulls.size());

    for (auto &convexHull : convexHulls) {
      Blob possibleBlob(convexHull);

      if (possibleBlob.currentBoundingRect.area() > min_blob_area &&
          possibleBlob.dblCurrentAspectRatio > 0.2 &&
          possibleBlob.dblCurrentAspectRatio < 4.0 &&
          possibleBlob.currentBoundingRect.width > 30 &&
          possibleBlob.currentBoundingRect.height > 30 &&
          possibleBlob.dblCurrentDiagonalSize > 60.0 &&
          (cv::contourArea(possibleBlob.currentContour) / (double)possibleBlob.currentBoundingRect.area()) > 0.50) {
        currentFrameBlobs.push_back(possibleBlob);
      }
    }

    drawAndShowContours(imgThresh.size(), currentFrameBlobs, "imgCurrentFrameBlobs");

    if (blnFirstFrame == true) {
      for (auto &currentFrameBlob : currentFrameBlobs) {
        blobs.push_back(currentFrameBlob);
      }
    }

    else {
      matchCurrentFrameBlobsToExistingBlobs(blobs, currentFrameBlobs);
    }

    drawAndShowContours(imgThresh.size(), blobs, "imgBlobs");

    // cout << blobs.size();
    // std::cout << blobs[1].currentBoundingRect<< endl;


    imgFrame2Copy = imgFrame2.clone();	// get another copy of frame 2 since we changed the previous frame 2 copy in the processing above

    // draw red rectangle on imgFrame2Copy

    // drawBlobInfoOnImage(blobs, imgFrame2Copy);

    // Check the rightWay
    bool blnAtLeastOneBlobCrossedTheLine = checkIfBlobsCrossedTheLineRight(blobs, intHorizontalLinePosition, carCountRight);
    // Check the leftWay
    bool blnAtLeastOneBlobCrossedTheLineLeft = checkIfBlobsCrossedTheLineLeft(blobs, intHorizontalLinePosition, carCountLeft);

    //rightWay
    if (blnAtLeastOneBlobCrossedTheLine == true) {
      cv::line(imgFrame2Copy, crossingLine[0], crossingLine[1], SCALAR_GREEN, 2);
    }
    else if (blnAtLeastOneBlobCrossedTheLine == false) {
      cv::line(imgFrame2Copy, crossingLine[0], crossingLine[1], SCALAR_RED, 2);
    }

    //leftway
    if (blnAtLeastOneBlobCrossedTheLineLeft == true) {
      cv::line(imgFrame2Copy, crossingLineLeft[0], crossingLineLeft[1], SCALAR_WHITE, 2);
    }
    else if (blnAtLeastOneBlobCrossedTheLineLeft == false) {
      cv::line(imgFrame2Copy, crossingLineLeft[0], crossingLineLeft[1], SCALAR_YELLOW, 2);
    }

    // if(blobs.back().currentBoundingRect.width < 100)
    //   continue;
    //DONE: convert cv::Mat frame crop to image
    image crop_image = crop_blob(imgFrame2Copy, blobs.back().currentBoundingRect);
    cout << crop_image.w << "  "<< crop_image.h<< endl;
    image sized = resize_image(crop_image, net.w, net.h);
    // last layer in config file
    layer l = net.layers[net.n-1];

    float nms=.45;	// 0.4F

    float *X = sized.data;

    int nboxes = 0;

    // l.classes = 5 total classes in layer;

    // draw_detections_v3(im, dets, nboxes, thresh, names, alphabet, l.classes, ext_output);

    // cout << names[0] <<endl;
    network_predict(net, X);

    // detection *dets = get_network_boxes(&net, crop_image.w, crop_image.h, thresh, hier_thresh, 0, 1, &nboxes, letterbox);
    detection *dets = get_network_boxes(&net, crop_image.w, crop_image.h, .8, hier_thresh, 0, 1, &nboxes, letterbox);
    if (nms) do_nms_sort(dets, nboxes, l.classes, nms);
    int ext_output;
    draw_detections_v3(crop_image, dets, nboxes, thresh, names, alphabet, l.classes, ext_output);
    // save_image(crop_image, "output_image/predictions");

    show_image(crop_image, "predictions");

    for (int i = 0; i < 10; ++i)
      {
        cout << X[i] << endl;
      }

    drawCarCountOnImage(carCountRight, imgFrame2Copy);

    cv::imshow("imgFrame2Copy", imgFrame2Copy);

    currentFrameBlobs.clear();

    imgFrame1 = imgFrame2.clone();	// move frame 1 up to where frame 2 is

    if ((capVideo.get(CV_CAP_PROP_POS_FRAMES) + 1) < capVideo.get(CV_CAP_PROP_FRAME_COUNT)) {
      capVideo.read(imgFrame2);
    }

    else {
      std::cout << "end of video\n";
      break;
    }

    blnFirstFrame = false;
    frameCount++;
    chCheckForEscKey = cv::waitKey(1);

    //free image crop
    free_image(crop_image);
    free_image(sized);
  }

  if (chCheckForEscKey != 27 && chCheckForEscKey != 'q') {
    cv::waitKey(0);
  }

  //free network and alphabet
  free_ptrs((void **)names, (int)net.layers[net.n - 1].classes);
  free_list_contents_kvp(options);
  free_list(options);
  const int nsize = 8;
  for (int j = 0; j < nsize; ++j) {
    for (int i = 32; i < 127; ++i) {
      free_image(alphabet[j][i]);
    }
    free(alphabet[j]);
  }
  free(alphabet);
  free_network(net);
  return(0);
}
