#ifndef FACIALFEATUREPOINTS_H
#define FACIALFEATUREPOINTS_H

#include <opencv2/opencv.hpp>
#include <opencv2/face.hpp>

class FacialFeaturePoints
{
public:

    FacialFeaturePoints(std::string face_detector_path = "../data/data_face_proc/haarcascade_frontalface_alt.xml",
                        std::string face_landmark_detector_path= "../data/data_face_proc/lbfmodel.yaml");

    std::vector<cv::Point2f> detect_landmarks(cv::Mat &frame);

    cv::CascadeClassifier faceDetector() const;
    void setFaceDetector(const cv::CascadeClassifier &faceDetector);

private:
    cv::CascadeClassifier _faceDetector;
    // Facemark instance
    cv::Ptr<cv::face::Facemark> _facemark;
};

#endif // FACIALFEATUREPOINTS_H
