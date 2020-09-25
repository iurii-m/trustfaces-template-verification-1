#ifndef TRF_FACIAL_BIOMETRIC_TEMPLATE_H
#define TRF_FACIAL_BIOMETRIC_TEMPLATE_H

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/objdetect.hpp>

#include <trf_core.h>
#include <trf_facialfeaturepoints.h>
#include <trf_HOG.h>

class trf_facial_biometric_template
{
public:
    trf_facial_biometric_template(std::string datapath = "");

    ~trf_facial_biometric_template();

    void calculate_geometric_template(cv::Mat face);

    void calculate_texture_template(cv::Mat face);

    std::vector<float> getFace_texture_template() const;
    void setFace_texture_template(const std::vector<float> &face_texture_template);

    std::vector<float> getFace_geometric_template() const;
    void setFace_geometric_template(const std::vector<float> &face_geometric_template);

private:

    //Geometric descriptor
    std::vector<float> _face_geometric_template;

    // Texture descriptor
    std::vector<cv::Mat> adjustAndDivideFaces(cv::Mat image);

    cv::Mat perspectiveCorretion(cv::Mat &im_src, cv::Mat &im_dst, std::vector<cv::Point2f> pointsSrc, std::vector<cv::Point2f> pointsDst);

    std::vector<float> _face_texture_template;

    FacialFeaturePoints _ffpDetector;

    cv::Mat _ref_face;

    std::vector <cv::Point2f> _ref_ffps;

    double _minX;
    double _minY;
    double _maxX;
    double _maxY;

    std::vector<cv::Mat> splitColorChannels(cv:: Mat & bgrImage);

    void getMinMaxFFPs(std::vector<cv::Point2f> &landmarks);

    std::vector<cv::Mat> getFaceRegionsMat(cv::Mat &im, std::vector<cv::Point2f> &landmarks);

    int pnpoly(std::vector<cv::Point2f> &verts, cv::Point2f point);

    cv::Mat getAllFaceRegionMat(cv::Mat &im, std::vector<cv::Point2f> &landmarks);

    std::vector<cv::Mat> calculatesHOGofFacesRois(std::vector<cv::Mat> rois_of_faces);

};

#endif // TRF_FACIAL_BIOMETRIC_TEMPLATE_H
