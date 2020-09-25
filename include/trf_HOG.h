#ifndef HOG_EXTRACT_
#define HOG_EXTRACT_

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <iostream>
#include <fstream>

void computeMagAngle(cv::InputArray src, cv::OutputArray mag, cv::OutputArray ang);

void computeHOG(cv::InputArray mag, cv::InputArray ang, cv::OutputArray dst, int dims, bool isWeighted);

#endif //HOG_EXTRACT_
