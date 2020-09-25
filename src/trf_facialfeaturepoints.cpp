#include "trf_facialfeaturepoints.h"

using namespace std;
using namespace cv;

FacialFeaturePoints::FacialFeaturePoints(std::string face_detector_path,std::string face_landmark_detector_path)
{

    cv::CascadeClassifier faceDetector(face_detector_path);

    _faceDetector = faceDetector;
    // Create an instance of Facemark
    cv::Ptr<cv::face::Facemark> facemark = cv::face::FacemarkLBF::create();
    // Load landmark detector
    facemark->loadModel(face_landmark_detector_path);
    _facemark = facemark;
}

vector <cv::Point2f> FacialFeaturePoints::detect_landmarks(cv::Mat & frame)
{

    cv::Mat gray;
    vector<cv::Rect> faces;
    // Convert frame to grayscale because
    // faceDetector requires grayscale image.
    if(frame.channels() == 3)
    {
        cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
    }
    else {
      gray = frame.clone();
    }
    // Detect faces
    _faceDetector.detectMultiScale(gray, faces);

    // all faces landmarks.
    vector< vector<cv::Point2f> > arr_ffps;

    // main faces landmarks.
    vector<cv::Point2f> main_ffps;

    // Run landmark detector
    bool success = _facemark->fit(frame,faces,arr_ffps);

    //Main landmarks
    if(arr_ffps.size()>0)
    {
        main_ffps = arr_ffps.at(0);
    }

    return main_ffps;
}

cv::CascadeClassifier FacialFeaturePoints::faceDetector() const
{
    return _faceDetector;
}

void FacialFeaturePoints::setFaceDetector(const cv::CascadeClassifier &faceDetector)
{
    _faceDetector = faceDetector;
}







