#include "trf_core.h"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>

using namespace  std;
using namespace  cv;

const int coeficient = 4000; //coefficient for UniQode coding. Transform difference to a convinient integer range
const int max_range = 59; //alphabet has 120 symbols so we have 59 positives and negatives, 1 zero and 1 extra

void prehandlingMatTo3channels(const cv::Mat &src, cv::Mat &dst)
{
    CV_Assert((src.type() == CV_8UC1) || (src.type() == CV_8UC3) || (src.type() == CV_8UC4));

    Mat src_work;
    if (src.type() == CV_8UC4) {
        cvtColor(src, src_work, COLOR_BGRA2BGR);
    }else if(src.type() == CV_8UC3){
        src_work = src.clone();
    }else {
        cvtColor(src, src_work, COLOR_GRAY2BGR);
    }
    dst = src_work;
    return;
}


/** Function tah finds distanse between 2 points*/
float find_distance(Point2f point1, Point2f point2)
{
        float distance = std::sqrt(((point1.x- point2.x)*(point1.x - point2.x))+((point1.y- (point2.y))*(point1.y- point2.y)));
        return (distance);
}

/** Function that finds perimeter*/
float find_perimeter(std::vector< vector<Point2f> > features)
{
    float perimeter =0;
    unsigned long first_point =-1;
    unsigned long final_point =0;

    for(unsigned long i=0;i<features[0].size();i++)
    {
     //if ((i>0)&&(i<27)){
     if ((i>=0)&&(i<27)){
         if (first_point==-1){first_point =i;}
         else {perimeter=perimeter+find_distance(features[0][i],features[0][final_point]);}

         final_point =i;
                }
    }
    perimeter=perimeter+find_distance(features[0][first_point],features[0][final_point]);
    return (perimeter);
}



Point2f facefeaturescenter(std:: vector<Point2f> features)
{
  float x=0; float y=0;
  int dec = 0;
  for(unsigned long k=0;k<features.size();k++)
  {
    x = x + features[k].x;
    y = y + features[k].y;
    dec++;
  }
  Point2f center = Point2f(x/dec,y/dec);
  return  center;
}

float getfeaturerotationangle(std:: vector<Point2f> features)
{
    float angle = 0; //in radians
    std:: vector<float> angles;
    angles.push_back(std::atan((features[16].y-features[0].y)/(features[16].x-features[0].x)));
    angles.push_back(std::atan((features[15].y-features[1].y)/(features[15].x-features[1].x)));
    angles.push_back(std::atan((features[14].y-features[2].y)/(features[14].x-features[2].x)));
    angles.push_back(std::atan((features[13].y-features[3].y)/(features[13].x-features[3].x)));
    angles.push_back(std::atan((features[12].y-features[4].y)/(features[12].x-features[4].x)));
    angles.push_back(std::atan((features[11].y-features[5].y)/(features[11].x-features[5].x)));
    angles.push_back(std::atan((features[10].y-features[6].y)/(features[10].x-features[6].x)));
    angles.push_back(std::atan((features[9].y-features[7].y)/(features[9].x-features[7].x)));
    angles.push_back(std::atan((features[26].y-features[17].y)/(features[26].x-features[17].x)));
    angles.push_back(std::atan((features[25].y-features[18].y)/(features[25].x-features[18].x)));
    angles.push_back(std::atan((features[24].y-features[19].y)/(features[24].x-features[19].x)));
    angles.push_back(std::atan((features[23].y-features[20].y)/(features[23].x-features[20].x)));
    angles.push_back(std::atan((features[22].y-features[21].y)/(features[22].x-features[21].x)));
    for (int i = 0; i < angles.size(); ++i) {
        //cout <<angles[i]<<" "<<endl;
        angle = angle +angles[i];
    }
    angle = angle/angles.size();
    return angle;
}

std:: vector<Point2f> getrotatedfeatures(std:: vector<Point2f> features, float rotationangle)
{
   std:: vector<Point2f> rotatedfeatures;
   for (int i = 0; i < features.size(); ++i) {
       Point2f featurepoint = Point2f(features[i].x*cos(rotationangle)-features[i].y*sin(rotationangle),features[i].y*cos(rotationangle)+features[i].x*sin(rotationangle));
       rotatedfeatures.push_back(featurepoint);
   }
   return rotatedfeatures;
}

float find_perimeter(std:: vector<Point2f>  features)
{
    float perimeter =0;
    unsigned long first_point =-1;
    unsigned long final_point =0;

    for(unsigned long i=0;i<features.size();i++)
    {
         //if ((i>0)&&(i<27)){
         if ((i>=0)&&(i<27))
         {
             if (first_point==-1){first_point =i;}
             else {perimeter=perimeter+find_distance(features[i],features[final_point]);}

             final_point =i;
        }
    }
    perimeter=perimeter+find_distance(features[first_point],features[final_point]);
    return (perimeter);
}


std:: vector<Point2f> getscaledfeatures(std:: vector<Point2f> features, float scale)
{
    std:: vector<Point2f> scaledfeatures;
    for (int i = 0; i < features.size(); ++i) {
        //cout <<angles[i]<<" "<<endl;
        Point2f featurepoint = Point2f(features[i].x*scale,features[i].y*scale);
        scaledfeatures.push_back(featurepoint);
    }
    return scaledfeatures;
}


std::vector<Point2f> fillreferencefeatures()
{
    std::vector<Point2f> features;
    features.push_back(Point2f(float(43.8947), float(526.118)));
    features.push_back(Point2f(float(47.7388), float(695.876)));
    features.push_back(Point2f(float(71.3329), float(861.667)));
    features.push_back(Point2f(float(100.978), float(1019.91)));
    features.push_back(Point2f(float(145.943), float(1172.82)));
    features.push_back(Point2f(float(225.271), float(1304.71)));
    features.push_back(Point2f(float(335.479), float(1416.35)));
    features.push_back(Point2f(float(470.257), float(1492.47)));
    features.push_back(Point2f(float(623.79), float(1515.08)));
    features.push_back(Point2f(float(781.003), float(1490.4)));
    features.push_back(Point2f(float(908.815), float(1403.13)));
    features.push_back(Point2f(float(1020.92), float(1294.84)));
    features.push_back(Point2f(float(1101.47), float(1159.81)));
    features.push_back(Point2f(float(1143.27), float(1011.04)));
    features.push_back(Point2f(float(1169.27), float(857.621)));
    features.push_back(Point2f(float(1189.51), float(705.624)));
    features.push_back(Point2f(float(1203.55), float(550.342)));
    features.push_back(Point2f(float(124.565), float(479.165)));
    features.push_back(Point2f(float(189.876), float(391.042)));
    features.push_back(Point2f(float(307.451), float(367.489)));
    features.push_back(Point2f(float(430.095), float(383.666)));
    features.push_back(Point2f(float(543.372), float(431.271)));
    features.push_back(Point2f(float(717.487), float(433.763)));
    features.push_back(Point2f(float(830.127), float(386.732)));
    features.push_back(Point2f(float(951.012), float(374.09)));
    features.push_back(Point2f(float(1070.18), float(392.506)));
    features.push_back(Point2f(float(1136.33), float(483.816)));
    features.push_back(Point2f(float(631.245), float(525.665)));
    features.push_back(Point2f(float(631.024), float(638.1)));
    features.push_back(Point2f(float(630.764), float(748.83)));
    features.push_back(Point2f(float(631.268), float(860.931)));
    features.push_back(Point2f(float(503.559), float(928.939)));
    features.push_back(Point2f(float(565.357), float(949.819)));
    features.push_back(Point2f(float(629.712), float(967.628)));
    features.push_back(Point2f(float(695.964), float(950.796)));
    features.push_back(Point2f(float(757.489), float(929.809)));
    features.push_back(Point2f(float(257.719), float(545.177)));
    features.push_back(Point2f(float(323.096), float(502.712)));
    features.push_back(Point2f(float(407.53), float(502.896)));
    features.push_back(Point2f(float(484.449), float(561.304)));
    features.push_back(Point2f(float(403.747), float(576.793)));
    features.push_back(Point2f(float(318.093), float(578.049)));
    features.push_back(Point2f(float(782.509), float(565.621)));
    features.push_back(Point2f(float(858.365), float(508.377)));
    features.push_back(Point2f(float(942.323), float(508.629)));
    features.push_back(Point2f(float(1003.71), float(552.693)));
    features.push_back(Point2f(float(943.726), float(585.201)));
    features.push_back(Point2f(float(859.405), float(581.729)));
    features.push_back(Point2f(float(399.378), float(1128.48)));
    features.push_back(Point2f(float(494.341), float(1107.76)));
    features.push_back(Point2f(float(569.346), float(1090.67)));
    features.push_back(Point2f(float(627.271), float(1110.57)));
    features.push_back(Point2f(float(684.726), float(1093.87)));
    features.push_back(Point2f(float(754.356), float(1109.04)));
    features.push_back(Point2f(float(845.558), float(1130.61)));
    features.push_back(Point2f(float(754.923), float(1177.88)));
    features.push_back(Point2f(float(684.736), float(1191.96)));
    features.push_back(Point2f(float(626.307), float(1194.34)));
    features.push_back(Point2f(float(565.258), float(1189.71)));
    features.push_back(Point2f(float(490.363), float(1173.37)));
    features.push_back(Point2f(float(434.397), float(1135.02)));
    features.push_back(Point2f(float(568.707), float(1135.3)));
    features.push_back(Point2f(float(627.757), float(1146.16)));
    features.push_back(Point2f(float(685.554), float(1137.88)));
    features.push_back(Point2f(float(812.408), float(1137.69)));
    features.push_back(Point2f(float(683.193), float(1122.4)));
    features.push_back(Point2f(float(626.84), float(1129.28)));
    features.push_back(Point2f(float(568.04), float(1120.7)));
    return features;

}

std::vector<Point2f> move_features(std:: vector<Point2f> features, float incx, float incy)
{
    std:: vector<Point2f> features_mod;
    for(unsigned long i=0;i<features.size();i++)
    {
        Point2f featurepoint = Point2f(features[i].x+incx,features[i].y+incy);
        features_mod.push_back(featurepoint);
    }
 return (features_mod);
}


std::vector<Point2f> normalizepoints(std:: vector<Point2f> orig_features)
{

    std:: vector<Point2f> normalized_points;
    float float_org_perimeter = find_perimeter(orig_features);

    for (unsigned long i = 0; i < orig_features.size(); ++i) {
      Point2f feature = Point2f((orig_features[i].x)/float_org_perimeter*1,(orig_features[i].y)/float_org_perimeter*1);
      normalized_points.push_back(feature);
    }

    return  normalized_points;

}

std:: vector<Point2f> preHandlingfeatures(std:: vector<Point2f> features)
{
    std:: vector<Point2f> referencefeatures = fillreferencefeatures();
    //rotation begin. making features horizontal
        //cout <<"featurescenter"<<featurescenter.x<<" "<< featurescenter.y <<endl;
        float rotation_angle = -1*getfeaturerotationangle(features); // -1 !!!
        std:: vector<Point2f> rotated_original_img_features;
        rotated_original_img_features = getrotatedfeatures(features,rotation_angle);
    //rotation end

    //scale begin making same scale as the reference features
        float float_original_perimeter = find_perimeter(rotated_original_img_features);
        float float_reference_perimeter = find_perimeter(referencefeatures);

        float scale = float_reference_perimeter/float_original_perimeter;
        std:: vector<Point2f> scaled_original_features;
        scaled_original_features = getscaledfeatures(rotated_original_img_features,scale);
        float float_scaled_original_perimeter = find_perimeter(scaled_original_features);
    //scale end

    //move begin. align input featurs and reference features.
        Point2f originalcenter = facefeaturescenter(scaled_original_features);
        Point2f referencecenter = facefeaturescenter(referencefeatures);
        float movex = referencecenter.x - originalcenter.x;
        float movey = referencecenter.y - originalcenter.y;
        //cout <<"movex" <<movex <<"movey" <<movey <<endl;
        std:: vector<Point2f> moved_features = move_features(scaled_original_features,movex,movey);
    //move end




    //handling features begin



     for (unsigned long i = 0; i < moved_features.size(); ++i) {
      float x = (moved_features[i].x-referencefeatures[i].x)/float_reference_perimeter*float(coeficient);
      float y = (moved_features[i].y-referencefeatures[i].y)/float_reference_perimeter*float(coeficient);
      if (x<float(-max_range)){
          moved_features[i].x = -float(max_range)*float_reference_perimeter/float(coeficient)+referencefeatures[i].x;
      }
      if (x>float(max_range)){
          moved_features[i].x = float(max_range)*float_reference_perimeter/float(coeficient)+referencefeatures[i].x;
      }
      if (y<float(-max_range)){
          moved_features[i].y = -float(max_range)*float_reference_perimeter/float(coeficient)+referencefeatures[i].y;
      }
      if (y>float(max_range)){
          moved_features[i].y = float(max_range)*float_reference_perimeter/float(coeficient)+referencefeatures[i].y;
      }

     }

    return moved_features;
}


cv::Mat croppAnyRect(const cv::Mat &src, cv::Rect &ROI)
{
    // Define margins for bottom top left and right
    int BtMrg = 0;
    int TpMrg = 0;
    int LftMrg = 0;
    int RgtMrg = 0;
    // Check left and top margins.
    if(ROI.x<0)
    {LftMrg = abs(ROI.x);}
    if(ROI.y<0) {TpMrg = abs(ROI.y);}
    // Check right and bottom margins
    int gpCols = src.cols - ROI.x - ROI.width;
    int gpRows = src.rows - ROI.y - ROI.height;
    if(gpCols<0)
    {RgtMrg = abs(gpCols);}
    if(gpRows<0)
    {BtMrg = abs(gpRows);}
    cv::Mat dst;
    //Decrease borders.
    cv::copyMakeBorder(src, dst, TpMrg, BtMrg, LftMrg, RgtMrg, cv::BORDER_REPLICATE);
    cv::Rect newROI = cv::Rect(ROI.x+LftMrg,ROI.y+TpMrg,ROI.width,ROI.height);
    cv::Mat anyCropp = dst(newROI);
    return anyCropp;
}



