#ifndef TRF_CORE_H
#define TRF_CORE_H

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/features2d.hpp>

void prehandlingMatTo3channels(const cv::Mat &src, cv::Mat &dst);

/** Function tah finds distanse between 2 points*/
float find_distance(cv::Point2f point1, cv::Point2f point2);

/** Function that finds perimeter*/
float find_perimeter(std::vector<std::vector<cv::Point2f>> features);

/** function that calculates center of features.*/
cv::Point2f facefeaturescenter(std:: vector<cv::Point2f> features);

/** function that calculates rotation angle for features.*/
float getfeaturerotationangle(std:: vector<cv::Point2f> features);

/** function that gives rotated features.*/
std:: vector<cv::Point2f> getrotatedfeatures(std:: vector<cv::Point2f> features, float rotationangle);

/** function that finds perimeter*/
float find_perimeter(std::vector<cv::Point2f>  features);

/** function that gives scaled features.*/
std::vector<cv::Point2f> getscaledfeatures(std::vector<cv::Point2f> features, float scale);

/** function that returns hardcoded reference face features.*/
std::vector<cv::Point2f> fillreferencefeatures();

/** function that shifts points in FFP vector.*/
std::vector<cv::Point2f> move_features(std:: vector<cv::Point2f> features, float incx, float incy);

/** function that divide points values by the perimeter.*/
std::vector<cv::Point2f> normalizepoints(std:: vector<cv::Point2f> orig_features);

/** function that prehandles input features(avoid exceding -59 - 59 limit of codng coordinates).*/
std::vector<cv::Point2f> preHandlingfeatures(std:: vector<cv::Point2f> features);

/** function to cropp any rectandle even with negative or */
cv::Mat croppAnyRect(const cv::Mat &src, cv::Rect &ROI);

#endif // TRF_CORE_H
