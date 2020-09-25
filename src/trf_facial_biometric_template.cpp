#include "trf_facial_biometric_template.h"

using namespace  cv;
using namespace  std;

trf_facial_biometric_template::trf_facial_biometric_template(std::string datapath)
{
    _ffpDetector = FacialFeaturePoints(datapath+"haarcascade_frontalface_alt.xml",datapath+"lbfmodel.yaml");
    _ref_face = cv::imread(datapath+"ref_image.png",cv::IMREAD_COLOR);

    //detecting landmarks on reference face
    _ref_ffps = _ffpDetector.detect_landmarks(_ref_face);

}

trf_facial_biometric_template::~trf_facial_biometric_template()
{

}

void trf_facial_biometric_template::calculate_geometric_template(cv::Mat face)
{
     _face_geometric_template.clear();

     //Preprocessing image
     cv::Mat framecopy;
     prehandlingMatTo3channels(face,framecopy);
     Mat framegray;
     cv::cvtColor(framecopy, framegray,COLOR_BGR2GRAY);

     //Detecting faces
     std::vector<cv::Rect> frontalfaces;

     unsigned int frontalfaces_mainpicture=0;

     //detecting landmarks
      vector<Point2f> face_features = _ffpDetector.detect_landmarks(framegray);

      // preHandling input features
      std:: vector<Point2f> handled_points = preHandlingfeatures(face_features);
      handled_points = normalizepoints(handled_points);

      //write vector of differnces to string
      std::vector<float> descriptor;

      for (unsigned long i = 0; i < handled_points.size(); ++i) {
           descriptor.push_back(float(handled_points[i].x));
           descriptor.push_back(float(handled_points[i].y));

      }

      _face_geometric_template = descriptor;

      string difstring = "";

      for (unsigned long i = 0; i < handled_points.size(); ++i) {
           std::string differencex = to_string(handled_points[i].x);
           std::string differencey = to_string(handled_points[i].y);
           difstring.append(differencex);
           difstring.append(" ");
           difstring.append(differencey);
           difstring.append(" ");
      }

      return;
}



void trf_facial_biometric_template::calculate_texture_template(cv::Mat face)
{

    _face_texture_template.clear();
     //Adjust and divide the face gets the parts of the face
     vector<Mat> parts_of_faces = adjustAndDivideFaces(face);

     //Calculating the HOG
     vector<Mat> wHogFeatures = calculatesHOGofFacesRois(parts_of_faces);

     //cout<<" hog features size "<<wHogFeatures.size()<<" cols "<<wHogFeatures[0].cols<<" rows "<<wHogFeatures[0].rows<<" hog features mat type -  "<<wHogFeatures[0].type()<<endl;

     vector<float> HogFeatures;

     for (int k = 0; k < wHogFeatures.size(); k++)
         {
         for (int j = 0; j < wHogFeatures[k].cols; j++)
             {

             HogFeatures.push_back(wHogFeatures[k].at<float>(0,j));

             //cout<<wHogFeatures[k].at<double>(j,0)<< " ";
             }
         }

     _face_texture_template = HogFeatures;
     return;

}

vector<Mat> trf_facial_biometric_template::adjustAndDivideFaces(cv::Mat image){

     Mat rgb_tgt = image;
     Mat rbg_src = _ref_face.clone();

     // Its detecting the FFPs from two faces
     Mat new_rgb_tgt = rgb_tgt.clone();

     //to do - do thip operation in constructor
     vector<Point2f> ffPs_src = _ref_ffps;

     vector<Point2f> ffPs_tgt = _ffpDetector.detect_landmarks(rgb_tgt);

     //resizing and the perspective Correction
     Mat resized_rgb_tgt = perspectiveCorretion(rgb_tgt,rbg_src,ffPs_tgt,ffPs_src);

     Mat bgr_src = rbg_src.clone();
     Mat resized_bgr_tgt = resized_rgb_tgt.clone();

     //get the min and max ffps of src
     getMinMaxFFPs(ffPs_src);
     //get the three channels of color image of the source
     vector<Mat> colorChannelsSCR = splitColorChannels(bgr_src);

     //convert to grayscale the image
     cvtColor(bgr_src, bgr_src, cv::COLOR_RGB2GRAY);

     //gets the  regions of the src
     //vector<Mat> face_part_src = getFaceRegionsMat(bgr_src, ffPs_src);

     //create the roi
     cv::Rect myROI_src(_minX, _minY, _maxX-_minX,_maxY-_minY);

     //recalculate the ffps of target
     ffPs_tgt = _ffpDetector.detect_landmarks(resized_rgb_tgt);

     //get the min and max ffps of target
     getMinMaxFFPs(ffPs_tgt);

     //get the three channels of color image of the source
     //vector<Mat> colorChannelsTGT = splitColorChannels(resized_bgr_tgt);

     cvtColor(resized_bgr_tgt, resized_rgb_tgt, cv::COLOR_RGB2GRAY);

     //gets the 10 regions of the tgt
     vector<Mat> face_part_tgt = getFaceRegionsMat(resized_rgb_tgt, ffPs_tgt);

     //get all face
     Mat all_Face_tgt = getAllFaceRegionMat(resized_rgb_tgt, ffPs_tgt);
     //get the three channels of color image of the target
     //vector<Mat> all_Face_Color_tgt = getAllFaceRegionMat(colorChannelsTGT, ffPs_src);

     //roi to crop the region
     cv::Rect myROI_tgt(_minX, _minY, _maxX-_minX,_maxY-_minY);
     for(int i = 0; i < face_part_tgt.size(); i++){
         face_part_tgt.at(i) = face_part_tgt.at(i)(myROI_tgt);
     }


     //    for(int i = 0; i < new_faces_parts.at(0).size(); i++){
     //                imshow("new_faces_parts0", new_faces_parts.at(0).at(i));
     //                imshow("new_faces_parts1", new_faces_parts.at(1).at(i));
     //                waitKey(0);
     //    }
     return face_part_tgt;
 }

cv::Mat trf_facial_biometric_template::perspectiveCorretion(Mat &im_src,Mat &im_dst,vector<Point2f> pointsSrc, vector<Point2f> pointsDst)
{
    Mat h = findHomography(pointsSrc, pointsDst);
    Mat im_out;
    // Warp source image to destination based on homography
    warpPerspective(im_src, im_out, h, im_dst.size());
    // imshow("Source Image", im_src);
    // imshow("Destination Image", im_dst);
    // imshow("Warped Source Image", im_out);
    return im_out;
}

std::vector<float> trf_facial_biometric_template::getFace_texture_template() const
{
    return _face_texture_template;
}

void trf_facial_biometric_template::setFace_texture_template(const std::vector<float> &face_texture_template)
{
    _face_texture_template = face_texture_template;
}

std::vector<float> trf_facial_biometric_template::getFace_geometric_template() const
{
    return _face_geometric_template;
}

void trf_facial_biometric_template::setFace_geometric_template(const std::vector<float> &face_geometric_template)
{
    _face_geometric_template = face_geometric_template;
}

std::vector<cv::Mat> trf_facial_biometric_template::splitColorChannels(Mat& bgrImage)
{
    vector<Mat> bgr_channels;
    split( bgrImage, bgr_channels );
    return bgr_channels;
}

std::vector<cv::Mat> trf_facial_biometric_template::getFaceRegionsMat(Mat &im, vector<Point2f> &landmarks)
{
    /**********************************************************************
REGION 0 */

    vector<Point2f> regionLeftEye;
    for(int i = 36; i < 42; i++)
        regionLeftEye.push_back(landmarks.at(i));


    /**********************************************************************
REGION 1 */

    vector<Point2f> regionRightEye;
    for(int i = 42; i < 48; i++)
        regionRightEye.push_back(landmarks.at(i));

    /***********************************************************************
REGION 2 */

    vector<Point2f> regionNose;
    for(int i = 31; i < 36; i++)
        regionNose.push_back(landmarks.at(i));
    regionNose.push_back(landmarks.at(22));
    regionNose.push_back(landmarks.at(21));

    /***********************************************************************
REGION 3 */

    vector<Point2f> regionLeftCheek;
    regionLeftCheek.push_back(landmarks.at(1));
    regionLeftCheek.push_back(landmarks.at(36));
    regionLeftCheek.push_back(landmarks.at(41));
    regionLeftCheek.push_back(landmarks.at(40));
    regionLeftCheek.push_back(landmarks.at(39));
    regionLeftCheek.push_back(landmarks.at(28));
    regionLeftCheek.push_back(landmarks.at(29));
    regionLeftCheek.push_back(landmarks.at(30));
    regionLeftCheek.push_back(landmarks.at(31));
    regionLeftCheek.push_back(landmarks.at(49));
    regionLeftCheek.push_back(landmarks.at(48));
    regionLeftCheek.push_back(landmarks.at(4));
    regionLeftCheek.push_back(landmarks.at(3));
    regionLeftCheek.push_back(landmarks.at(2));

    /***********************************************************************
REGION 4 */

    vector<Point2f> regionRightCheek;
    regionRightCheek.push_back(landmarks.at(16));
    regionRightCheek.push_back(landmarks.at(45));
    regionRightCheek.push_back(landmarks.at(46));
    regionRightCheek.push_back(landmarks.at(47));
    regionRightCheek.push_back(landmarks.at(42));
    regionRightCheek.push_back(landmarks.at(28));
    regionRightCheek.push_back(landmarks.at(29));
    regionRightCheek.push_back(landmarks.at(30));
    regionRightCheek.push_back(landmarks.at(35));
    regionRightCheek.push_back(landmarks.at(53));
    regionRightCheek.push_back(landmarks.at(54));
    regionRightCheek.push_back(landmarks.at(12));
    regionRightCheek.push_back(landmarks.at(13));
    regionRightCheek.push_back(landmarks.at(14));

    /***********************************************************************
REGION 5 */

    vector<Point2f> regionLeftChin;
    for(int i = 2; i < 9; i++)
        regionLeftChin.push_back(landmarks.at(i));
    regionLeftChin.push_back(landmarks.at(31));

    /***********************************************************************
REGION 6 */

    vector<Point2f> regionRightChin;
    for(int i = 14; i > 7; i--)
        regionRightChin.push_back(landmarks.at(i));
    regionRightChin.push_back(landmarks.at(35));

    /***********************************************************************
REGION 7 */

    vector<Point2f> regionMoustache;
    regionMoustache.push_back(landmarks.at(31));
    regionMoustache.push_back(landmarks.at(35));
    regionMoustache.push_back(landmarks.at(54));
    regionMoustache.push_back(landmarks.at(48));

    /***********************************************************************
REGION 8 */

    vector<Point2f> regionLeftEyebrow;
    for(int i = 17; i < 22; i++)
        regionLeftEyebrow.push_back(landmarks.at(i));
    for(int i = 39; i > 35; i--)
        regionLeftEyebrow.push_back(landmarks.at(i));

    /***********************************************************************
REGION 9 */

    vector<Point2f> regionRightEyebrow;
    for(int i = 22; i < 27; i++)
        regionRightEyebrow.push_back(landmarks.at(i));
    for(int i = 45; i > 42; i--)
        regionRightEyebrow.push_back(landmarks.at(i));

    /***********************************************************************/

    vector<Mat> regions;
//    vector<Mat> regionsFace;

    for (int i = 0; i < 10; i++)
    {
        regions.push_back(cv::Mat::zeros(im.rows,im.cols, CV_8UC1));
    }

    for (int x = _minX; x < _maxX; x++)
    {
        for (int y = _minY; y < _maxY; y++)
        {
            Point2f point = cv::Point(x,y);
            if(pnpoly(regionLeftEye,point)){
                regions.at(0).at<uchar>(y,x) = im.at<uchar>(y,x);
            }
            if(pnpoly(regionRightEye,point)){
                regions.at(1).at<uchar>(y,x) = im.at<uchar>(y,x);
            }
            if(pnpoly(regionNose,point)){
                regions.at(2).at<uchar>(y,x) = im.at<uchar>(y,x);
            }
            if(pnpoly(regionLeftCheek,point)){
                regions.at(3).at<uchar>(y,x) = im.at<uchar>(y,x);
            }
            if(pnpoly(regionRightCheek,point)){
                regions.at(4).at<uchar>(y,x) = im.at<uchar>(y,x);
            }
            if(pnpoly(regionLeftChin,point)){
                regions.at(5).at<uchar>(y,x) = im.at<uchar>(y,x);
            }
            if(pnpoly(regionRightChin,point)){
                regions.at(6).at<uchar>(y,x) = im.at<uchar>(y,x);
            }
            if(pnpoly(regionMoustache,point)){
                regions.at(7).at<uchar>(y,x) = im.at<uchar>(y,x);
            }
            if(pnpoly(regionLeftEyebrow,point)){
                regions.at(8).at<uchar>(y,x) = im.at<uchar>(y,x);
            }
            if(pnpoly(regionRightEyebrow,point)){
                regions.at(9).at<uchar>(y,x) = im.at<uchar>(y,x);
            }
        }
    }

//    for (int i = 0; i < 10; i++)
//    {
//        regionsFace.push_back(regions.at(i));
//    }

//        imshow( "regions 0", regions.at(0));
//        imshow( "regions 1", regions.at(1));
//        imshow( "regions 2", regions.at(2));
//        imshow( "regions 3", regions.at(3));
//        imshow( "regions 4", regions.at(4));
//        imshow( "regions 5", regions.at(5));
//        imshow( "regions 6", regions.at(6));
//        imshow( "regions 7", regions.at(7));
//        imshow( "regions 8", regions.at(8));
//        imshow( "regions 9", regions.at(9));
//        waitKey(0);

    return  regions;
}

/*Method to get all face as unique region*/
cv::Mat trf_facial_biometric_template::getAllFaceRegionMat(Mat &im, vector<Point2f> &landmarks)
{

    vector<Point2f> allFace;
    for(int i = 0; i < 16; i++)
        allFace.push_back(landmarks.at(i));
    for(int i = 26; i > 16; i--)
        allFace.push_back(landmarks.at(i));

    Mat region = cv::Mat::zeros(im.rows,im.cols, CV_8UC1);
    for (int x = _minX; x < _maxX; x++)
    {
        for (int y = _minY; y < _maxY; y++)
        {
            Point2f point = cv::Point(x,y);
            if(pnpoly(allFace,point)){
                region.at<uchar>(y,x) = im.at<uchar>(y,x);
            }
        }
    }
//    imshow( "allFace", region);
//    waitKey(0);
    return region;
}


/*Method that receives the rois of the faces and returns the Hog features of each roi*/
vector<Mat> trf_facial_biometric_template::calculatesHOGofFacesRois(vector<Mat> rois_of_faces){

        vector<Mat> mags_of_face;
        vector<Mat> angs_of_face;
        for(int j = 0; j < rois_of_faces.size();j++){
            Mat mag;
            Mat ang;
            computeMagAngle(rois_of_faces.at(j),mag,ang);
            //            computeMagAngle(lbps_faces.at(i).at(j),mag,ang);
            //            imshow("mag", mag);
            //            imshow("ang", ang);
            //            waitKey(0);
            mags_of_face.push_back(mag);
            angs_of_face.push_back(ang);
        }

        vector<Mat> wHogFeature;
        for(int j = 0; j < mags_of_face.size();j++){
            Mat wHogFeat;
            computeHOG(mags_of_face.at(j), angs_of_face.at(j), wHogFeat, 8, true);

            Mat wHogFeatnorm;// = normalizeHistograma(wHogFeat);
            normalize(wHogFeat, wHogFeatnorm, 0, 1,NORM_MINMAX);
            wHogFeature.push_back(wHogFeatnorm);
        }


    return wHogFeature;
}


//gets the minimum and maximum values of the ffps
void trf_facial_biometric_template::getMinMaxFFPs(vector<Point2f> &landmarks)
{
    _minX = landmarks.at(0).x;
    _minY = landmarks.at(0).y;
    _maxX = landmarks.at(0).x;
    _maxY = landmarks.at(0).y;
    for (int i = 1; i < landmarks.size(); i++){
        if(landmarks.at(i).x < _minX)
            _minX = landmarks.at(i).x;
        if(landmarks.at(i).x > _maxX)
            _maxX = landmarks.at(i).x;
        if(landmarks.at(i).y < _minY)
            _minY = landmarks.at(i).y;
        if(landmarks.at(i).y > _maxY)
            _maxY = landmarks.at(i).y;
    }
}

//verify if a point is inside of a roi
int trf_facial_biometric_template::pnpoly(vector<Point2f> &verts, Point2f point)
{
    int i, j, c = 0;
    for (i = 0, j = verts.size()-1; i < verts.size(); j = i++) {
        if ( ((verts[i].y>point.y) != (verts[j].y>point.y)) &&
             (point.x < (verts[j].x-verts[i].x) * (point.y-verts[i].y) / (verts[j].y-verts[i].y) + verts[i].x) )
            c = !c;
    }
    return c;
}
