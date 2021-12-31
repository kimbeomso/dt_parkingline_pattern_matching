
// double M_result=0;
// clock_t start, end;start = clock();
// end = clock();
// M_result = (double)(end - start);
// printf("sec : %f\n", M_result/CLOCKS_PER_SEC);

#include <iostream>
#include <string>
#include <exception>

#include <tf/transform_broadcaster.h>
#include <ros/ros.h>
#include <ros/package.h>
#include <tf/transform_broadcaster.h>
#include <nav_msgs/OccupancyGrid.h>
#include <geometry_msgs/PoseArray.h>
#include <visualization_msgs/Marker.h>

#include <cv_bridge/cv_bridge.h>
#include <cv.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/opencv.hpp"

#include <std_msgs/Float32MultiArray.h>
#include <algorithm>
#include <vector>
#include <stack>

using namespace cv;
using namespace std;

int img_width=400;
int img_height=400;
int real_occupancy_size_x = 25;
int real_occupancy_size_y = real_occupancy_size_x;

bool   is_mod=false;
double occ_ratio=1.0;                 // ocam의 occupancy grid맵 크기변화를 template 크기에 적용하기 위함
double img_ratio=1.0;
double pixel_per_meter_ratio = 1.0;

#define PIXEL_PER_METER  double(img_width) / double(real_occupancy_size_x)           //400PIX / REAL_OCCUPANCY_SIZE_XM 
#define SIDE 0
#define CENTER 1
#define DEG2RAD 3.14/180.0
bool matching_fail = true;

geometry_msgs::PoseArray posearray;
int g_direction = SIDE;                 // SIDE / CENTER 
double dis_thresh =1.3;                 // distance cost threshold
double searching_range = 1.5;           // meter
int shape_discrete_dt_threshold = 5;

double parking_long_line_len = 4.7/2.0;
double arrow_line_len = 0.7/2.0; 

ros::Publisher pub_pose;
ros::Subscriber sub_raw_cen ,sub_raw_side ,sub_seg_cen ,sub_seg_side ,Sub_phantom_DR_Path, sub_bounding_box;
ros::Publisher vis_pattern_pub;

//center, side RAW, SEG image
cv::Mat srcImg_cen_seg = cv::Mat::zeros(img_width, img_height, CV_8UC3);
cv::Mat srcImg_side_seg = cv::Mat::zeros(img_width, img_height, CV_8UC3);

cv::Mat temp_skel= cv::Mat::zeros(img_width, img_height, CV_8UC1);
cv::Mat temp_skel2= cv::Mat::zeros(img_width, img_height, CV_8UC1);

cv::Mat click_img= cv::Mat::zeros(img_width, img_height, CV_8UC1);

cv::Mat temp_src= cv::Mat::zeros(img_width, img_height, CV_8UC3);
cv::Mat temp_src2= cv::Mat::zeros(img_width, img_height, CV_8UC3);
cv::Mat temp_dist= cv::Mat::zeros(img_width, img_height, CV_32FC1);

cv::Mat skel(img_width, img_height, CV_8UC1, cv::Scalar(0));
cv::Mat skel2(img_width, img_height, CV_8UC1, cv::Scalar(0));
cv::Mat dist  = cv::Mat::zeros(img_width*img_ratio, img_height*img_ratio, CV_32FC1);
cv::Mat dist2 = cv::Mat::zeros(img_width*img_ratio, img_height*img_ratio, CV_32FC1);

vector<pair<int,int>> m_shape_points;
typedef struct LINE_STRUCT{
    int x;
    int y;
}skeleton_pixel;

stack<skeleton_pixel> m_parking_line_side;
stack<skeleton_pixel> m_parking_line_cen;

typedef struct BOX_POINT{
    float x;
    float y;
};
vector<BOX_POINT> box_point_vector;
typedef struct POSE{
    double x;
    double y;
    double th;
};
vector<POSE> pose_vec_side;
vector<POSE> pose_vec_cen;
Point g_tmp_point;

int clicked_x=0, clicked_y=0;
double corner1_x= -18.0, corner1_y= 0.0;
double corner2_x= 18.0, corner2_y= 0.0;

void skeletonize_parkingline(cv::Mat src, cv::Mat dst){
    if(!src.empty() && !dst.empty() && (src.size()==dst.size()))
    {
        cv::Mat img_gray =  cv::Mat::zeros(img_width*img_ratio, img_height*img_ratio, CV_8UC1);
        cv::Mat eroded;
        cv::Mat temp(img_width*img_ratio, img_height*img_ratio, CV_8UC1);
        dst = cv::Mat::zeros(img_width*img_ratio, img_height*img_ratio, CV_8UC1);
        
        cv::Mat element = cv::getStructuringElement(cv::MORPH_CROSS, cv::Size(3, 3));
        cvtColor(src, img_gray, COLOR_BGR2GRAY);
        cv::threshold(img_gray, img_gray, 70, 255, cv::THRESH_BINARY);
        bool done;
        do
        {
            cv::erode(img_gray, eroded, element);
            cv::dilate(eroded, temp, element);   // temp = open(img)
            cv::subtract(img_gray, temp, temp);
            cv::bitwise_or(dst, temp, dst);
            eroded.copyTo(img_gray);   
            done = (cv::countNonZero(img_gray) == 0);
        } while (!done);
    }
}

void preprocessing (cv::Mat input_img1, cv::Mat input_img2){
    if(!input_img1.empty() && !input_img2.empty() && (input_img1.size() == input_img2.size()))
    for(int ii=0;ii < input_img1.size().height; ii++)
        for(int jj = 0; jj< input_img1.size().width; jj++){
            //remove vehicle
            if((input_img2.at<Vec3b>(jj,ii)[0]==0) && (input_img2.at<Vec3b>(jj, ii)[1]==125) && (input_img2.at<Vec3b>(jj,ii)[2]==255))
                {input_img1.at<Vec3b>(jj,ii)[0] =0;  input_img1.at<Vec3b>(jj,ii)[1] =0;  input_img1.at<Vec3b>(jj,ii)[2] =0; }
            if((input_img2.at<Vec3b>(jj,ii)[0]==255) && (input_img2.at<Vec3b>(jj, ii)[1]==255) && (input_img2.at<Vec3b>(jj,ii)[2]==255))
                {input_img1.at<Vec3b>(jj,ii)[0] =0;  input_img1.at<Vec3b>(jj,ii)[1] =0;  input_img1.at<Vec3b>(jj,ii)[2] =0; }
            if((input_img2.at<Vec3b>(jj,ii)[0]==255) && (input_img2.at<Vec3b>(jj, ii)[1]==0) && (input_img2.at<Vec3b>(jj,ii)[2]==0))
                {input_img1.at<Vec3b>(jj,ii)[0] =0;  input_img1.at<Vec3b>(jj,ii)[1] =0;  input_img1.at<Vec3b>(jj,ii)[2] =0; }
            //remove background
            if((input_img1.at<cv::Vec3b>(jj, ii)[2] < 150) || (input_img1.at<cv::Vec3b>(jj, ii)[1] > 125)  || (input_img1.at<cv::Vec3b>(jj, ii)[0] < 175)) 
                {input_img1.at<Vec3b>(jj,ii)[0] =0;  input_img1.at<Vec3b>(jj,ii)[1] =0;  input_img1.at<Vec3b>(jj,ii)[2] =0; }
        }
}
void reverse_image(cv::Mat src, cv::Mat dst){
    for(int ii=0;ii < src.size().height; ii++)
        for(int jj = 0; jj< src.size().width; jj++)
            dst.at<uchar>(jj,ii) = 255 - src.at<uchar>(jj,ii);
}
//mouse click하면 근처 픽셀 저장
void stack_skeleton(cv::Mat src, int direction){
    skeleton_pixel parking_line_;

    if(!src.empty())
    for(int ii=0;ii < src.size().height; ii++)
        for(int jj = 0; jj< src.size().width; jj++)
        {
            double distance = sqrt((jj - clicked_x)*(jj - clicked_x) + (ii - clicked_y)*(ii - clicked_y));

            if ( src.at<uchar>(ii, jj) !=0 && (distance < searching_range * double(PIXEL_PER_METER)*pixel_per_meter_ratio) )               // line이 검은색 (reverse상태)   //3m이내의 점들
            {
                // skeleton line 에 해당하면 , stack
                parking_line_.x = jj;   parking_line_.y = ii;
                if(direction==SIDE)
                    m_parking_line_side.push(parking_line_);
                else if(direction==CENTER)
                    m_parking_line_cen.push(parking_line_);
            }
        }
}
void matching_template(stack<skeleton_pixel> parking_line , cv::Mat *dist_, vector<POSE> *pose_vec){
    pose_vec->clear();
    double min_dist = 0.0;
    double min_x = 0.0, min_y = 0.0;
    double min_th = 0.0;
    // Stacked Parking Line 에서 Rot Matching
    while(!parking_line.empty()){
        skeleton_pixel pop_parking_line;
        pop_parking_line = parking_line.top();
        parking_line.pop();
        int x = pop_parking_line.x; int y = pop_parking_line.y;
        for (double th_ = 0.0 ; th_ <= 360.0 ; th_+=1.0)
        {
            double distance_sum = 0.0;
            for (int i = 0 ; i < m_shape_points.size() ; i+=1)
            {
                int shape_x = (int)(x + (m_shape_points[i].first)*(pixel_per_meter_ratio)*cosf(th_*DEG2RAD) - (m_shape_points[i].second)*(pixel_per_meter_ratio) *sinf(th_*DEG2RAD));
                int shape_y = (int)(y + (m_shape_points[i].first)*(pixel_per_meter_ratio)*sinf(th_*DEG2RAD) + (m_shape_points[i].second)*(pixel_per_meter_ratio) *cosf(th_*DEG2RAD));
                //안 더해진게 가장 작긴하지... => 밖으로 나가는 경우
                if (0 > shape_x || shape_x >= dist_->cols || 0 > shape_y || shape_y >= dist_->rows) continue;   
                // if (dist_.at<uchar>(shape_x, shape_y) == 255) // 곂치고 안곂치고 문제가아님, distance value sum 을 minimizing 하는 위치를 찾음
                distance_sum += (1.0 - (double)(dist_->at<float>(shape_y , shape_x) ));
            }

            if (min_dist < distance_sum) 
            {
                min_dist = distance_sum;
                min_x = x;  min_y = y;  min_th = th_;
                matching_fail = false;;
            }
        }
    }
    POSE tmp_pose;
    // std::cout<< "distance_sum : " << min_dist - m_shape_points.size() << std::endl;
    if(fabs(min_dist - m_shape_points.size()) > dis_thresh)
    {
        matching_fail = true;
        min_x = 0.0; min_y=0.0; min_th=0.0;
    }
    tmp_pose.x  = min_x;    tmp_pose.y  = min_y;    tmp_pose.th = min_th;
    pose_vec->push_back(tmp_pose);
}
double get_pose(int x1, int y1, int x2, int y2)
{
    double ang;
    double p_m =0, m=0;
    double cross_point_x, cross_point_y;
    double car_x, car_y;
    car_x = double(img_width)*img_ratio / 2.0;
    car_y = double(img_height)*img_ratio / 2.0;
    if( x1==x2 )                        //기울기(m) 무한대
        ang = atan2( 0 , car_x - x1 );
    else
    {
        m = ( double(y1) - double(y2) ) / ( double(x1) - double(x2) );
        if(m == 0)
            ang = atan2( car_y - y1, 0);
        else{
            cross_point_x = ( car_x*(1.0/m + 1.0) - y1 + m*x1 ) / ( m + 1.0/m );      // x = (1/m + 1)*200 - y1 + m*x1     /    (1/m + m)
            cross_point_y = m*cross_point_x + y1 - m*x1;
            ang = atan2 (car_y - cross_point_y , car_x - cross_point_x);
        }
    }
    return ang;
}
void poseArray_pub( vector<POSE> pose_vec){
    for(int i=0;i<pose_vec.size();i++){
        geometry_msgs::Pose pose;
        bool front_point = true;    // 앞뒤 구분

        double org_shape_corner_x = 18.0;
        double org_shape_corner_y = 0.0;

        corner1_x = (int)(pose_vec[i].x - org_shape_corner_x*(pixel_per_meter_ratio)*cosf(pose_vec[i].th*DEG2RAD) - org_shape_corner_y*(pixel_per_meter_ratio)*sinf(pose_vec[i].th*DEG2RAD));
        corner1_y = (int)(pose_vec[i].y - org_shape_corner_x*(pixel_per_meter_ratio)*sinf(pose_vec[i].th*DEG2RAD) + org_shape_corner_y*(pixel_per_meter_ratio)*cosf(pose_vec[i].th*DEG2RAD));
        corner2_x = (int)(pose_vec[i].x + org_shape_corner_x*(pixel_per_meter_ratio)*cosf(pose_vec[i].th*DEG2RAD) - org_shape_corner_y*(pixel_per_meter_ratio)*sinf(pose_vec[i].th*DEG2RAD));
        corner2_y = (int)(pose_vec[i].y + org_shape_corner_x*(pixel_per_meter_ratio)*sinf(pose_vec[i].th*DEG2RAD) + org_shape_corner_y*(pixel_per_meter_ratio)*cosf(pose_vec[i].th*DEG2RAD));
        if (0 > corner1_x || corner1_x >= temp_src.cols || 0 > corner1_y || corner1_y >= temp_src.rows || 
            0 > corner2_x || corner2_x >= temp_src.cols || 0 > corner2_y || corner2_y >= temp_src.rows) continue;

        double pose_ang = get_pose( corner1_x ,corner1_y ,corner2_x ,corner2_y );       // RAD 
        double x,y;
        double mid_x = (corner1_x + corner2_x) / 2.0;
        double mid_y = (corner1_y + corner2_y) / 2.0;

        double yaw = -pose_ang - 90.0*DEG2RAD;      // 이미지 -> world 좌표계
        // corner 가 앞쪽일 때 주차영역 중심위치
        if( front_point == true )
        {
            x =  -(parking_long_line_len)*cosf(yaw)  + (1.0 / double(PIXEL_PER_METER*(pixel_per_meter_ratio)))*((img_width*img_ratio /2.0)-( mid_y ));
            y =  -(parking_long_line_len)*sinf(yaw)  + (1.0 / double(PIXEL_PER_METER*(pixel_per_meter_ratio)))*((img_width*img_ratio /2.0)-( mid_x ));
        }
        // corner가 뒷쪽일 때 주차영역 중심위치
        else
        {
            x =  +(parking_long_line_len)*cosf(yaw)  + (1.0 / double(PIXEL_PER_METER*(pixel_per_meter_ratio)))*((img_width*img_ratio /2.0)-( mid_y ));
            y =  +(parking_long_line_len)*sinf(yaw)  + (1.0 / double(PIXEL_PER_METER*(pixel_per_meter_ratio)))*((img_width*img_ratio /2.0)-( mid_x ));
        }

        pose.orientation = tf::createQuaternionMsgFromYaw(yaw); 
        pose.position.x = x; pose.position.y = y; pose.position.z = 0.1;
        posearray.poses.push_back(pose);
    }
}

void bounding_box_callback(const std_msgs::Float32MultiArray::ConstPtr& msg){
    box_point_vector.clear();
    double arr[8];
    if(is_mod == false){
        occ_ratio = (double)(msg->data.at(9)) / 25.0;     //25 은 template이 제작될 당시 pixel_per_meter 크기
        img_ratio = (double)(msg->data.at(8)) / 400.0;

        is_mod = true;
        pixel_per_meter_ratio = (img_ratio/occ_ratio);
    }
}

void imageCallback_side_seg(const sensor_msgs::ImageConstPtr &msg)
{
    srcImg_side_seg = cv_bridge::toCvCopy(msg, "bgr8" )->image;
}
void imageCallback_cen_seg(const sensor_msgs::ImageConstPtr &msg)
{
    srcImg_cen_seg = cv_bridge::toCvCopy(msg, "bgr8" )->image;
}
void imageCallback_cen_raw(const sensor_msgs::ImageConstPtr &msg)
{
    cv::Mat srcImg_cen = cv::Mat::zeros(img_width*img_ratio, img_height*img_ratio, CV_8UC3);
    srcImg_cen = cv_bridge::toCvCopy(msg, "bgr8" )->image;
    temp_src = srcImg_cen.clone();
    preprocessing(srcImg_cen, srcImg_cen_seg);

    skeletonize_parkingline(srcImg_cen, skel);
    temp_skel = skel;
    if(!skel.empty()) 
    {
        cv::Mat rev_img = cv::Mat::zeros(img_width*img_ratio, img_height*img_ratio, CV_8UC1);
        reverse_image(skel, rev_img);
        distanceTransform(rev_img, dist, DIST_L2, 3);
        normalize(dist, dist, 0, 1.0, NORM_MINMAX); 
    }
}
void imageCallback_side_raw(const sensor_msgs::ImageConstPtr &msg)
{
    cv::Mat srcImg_side = cv::Mat::zeros(img_width*img_ratio, img_height*img_ratio, CV_8UC3);
    srcImg_side = cv_bridge::toCvCopy(msg, "bgr8" )->image;
    temp_src2 = srcImg_side.clone();
    preprocessing(srcImg_side, srcImg_side_seg);
    
    skeletonize_parkingline(srcImg_side, skel2);
    temp_skel2 = skel2;
    if(!skel2.empty()) 
    {
        cv::Mat rev_img2 = cv::Mat::zeros(img_width*img_ratio, img_height*img_ratio, CV_8UC1);
        reverse_image(skel2, rev_img2);
        distanceTransform(rev_img2, dist2, DIST_L2, 3);    //DIST_L2 : euclidean dist
        normalize(dist2, dist2, 0, 1.0, NORM_MINMAX);  
    }
}

void CallBackFunc( int event, int x, int y, int flags, void* userdata )
{
    if ( event == EVENT_LBUTTONDOWN )
    {
        std::cout << "position (" << x << ", " << y << ")" << std::endl;
        clicked_x = x;  clicked_y = y;      // rows, cols
    }
}

void visualize_pattern_pub(vector<pair<int,int>> m_shape_points, vector<POSE> pose_vec)
{
    for(int j =0;j<pose_vec.size() ;j++) {
        visualization_msgs::Marker points;
        points.header.frame_id = "map";
        points.header.stamp = ros::Time();
        points.type = visualization_msgs::Marker::POINTS;

        points.pose.orientation.w = 1.0;
        points.scale.x = 0.1, points.scale.y = 0.1;
        points.color.a = 1.0, points.color.r = 1.0, points.color.g = 1.0, points.color.b = 0.0;

        for (int i=0; i<m_shape_points.size(); i+=shape_discrete_dt_threshold)      
        {
            double pix_p_x = (int)(pose_vec[j].x + (m_shape_points[i].first)*(pixel_per_meter_ratio)*cosf(pose_vec[j].th*DEG2RAD) - (m_shape_points[i].second)*(pixel_per_meter_ratio)*sinf(pose_vec[j].th*DEG2RAD));
            double pix_p_y = (int)(pose_vec[j].y + (m_shape_points[i].first)*(pixel_per_meter_ratio)*sinf(pose_vec[j].th*DEG2RAD) + (m_shape_points[i].second)*(pixel_per_meter_ratio)*cosf(pose_vec[j].th*DEG2RAD));

            double x = (1.0 / double(PIXEL_PER_METER*(pixel_per_meter_ratio)))*((img_width*img_ratio /2.0)-( pix_p_y ));;
            double y = (1.0 / double(PIXEL_PER_METER*(pixel_per_meter_ratio)))*((img_width*img_ratio /2.0)-( pix_p_x ));;

            geometry_msgs::Point p;
            p.x = x, p.y = y, p.z = 0.1;
            
            points.points.push_back(p);
        }
        vis_pattern_pub.publish(points);
    }
}

int main(int argc, char **argv)
{   
    ros::init(argc, argv, "distance_node");
    ros::NodeHandle nodeHandle("~");

    string directory = "/home/beomsoo/catkin_ws/src/distance_transform/include/box.txt";
    ifstream fin;

    fin.open(directory);
    double x, y;
    while(! fin.eof()){
        fin >> x >> y;
        m_shape_points.push_back(make_pair(x, y));
    }
    if(fin.is_open())   fin.close();

    sub_raw_cen         = nodeHandle.subscribe("/AVM_center_image" , 1, imageCallback_cen_raw);
    sub_raw_side        = nodeHandle.subscribe("/AVM_image" , 1, imageCallback_side_raw);
    sub_seg_cen         = nodeHandle.subscribe("/AVM_center_seg_image", 1, imageCallback_cen_seg);
    sub_seg_side        = nodeHandle.subscribe("/AVM_side_seg_image", 1, imageCallback_side_seg);
    sub_bounding_box    = nodeHandle.subscribe("/boundingbox_point", 1, bounding_box_callback);

    pub_pose = nodeHandle.advertise<geometry_msgs::PoseArray>("/parking_pose", 1);
    vis_pattern_pub = nodeHandle.advertise<visualization_msgs::Marker>( "/visualization_marker", 0 );

    ros::Rate loop_rate(20);
    // ros::spin();

    while(ros::ok()) { 
        ros::AsyncSpinner spinner(2+1);
        spinner.start();

        posearray.poses.clear();
        posearray.header.stamp = ros::Time::now();
        posearray.header.frame_id = "map" ;
// double M_result=0;
// clock_t start, end;start = clock();
        if( g_direction == CENTER )
        {
            stack_skeleton(skel , CENTER);
            matching_template(m_parking_line_cen, &dist, &pose_vec_cen);
            dist = cv::Mat::zeros(img_width*img_ratio, img_height*img_ratio, CV_32FC1);
            while( !m_parking_line_cen.empty() ) m_parking_line_cen.pop();  

            if(pose_vec_cen.size()!=0 && matching_fail == false)   
            {
                poseArray_pub(pose_vec_cen);
                visualize_pattern_pub(m_shape_points, pose_vec_cen);
            }
            click_img = temp_skel.clone();
        }
        else if( g_direction == SIDE )
        {
            stack_skeleton(skel2 , SIDE);
            matching_template(m_parking_line_side, &dist2, &pose_vec_side);
            dist2 = cv::Mat::zeros(img_width*img_ratio, img_height*img_ratio, CV_32FC1);
            while( !m_parking_line_side.empty() ) m_parking_line_side.pop();  
            if(pose_vec_side.size()!=0 && matching_fail == false)
            {
                poseArray_pub(pose_vec_side);
                visualize_pattern_pub(m_shape_points, pose_vec_side);
            }
            click_img = temp_skel2.clone();
        }

        if(posearray.poses.size()!=0)  pub_pose.publish(posearray);
// end = clock();
// M_result = (double)(end - start);
// printf("sec : %f\n", M_result/CLOCKS_PER_SEC);
        /// Create a window
        cv::namedWindow( "ImageDisplay", 1 );
        cv::setMouseCallback( "ImageDisplay", CallBackFunc, NULL );
        cv::imshow( "ImageDisplay", click_img );
        /// Wait until user press some key
        waitKey(0);


        cv::imshow("srcImg_side_seg" ,srcImg_side_seg);
        cv::imshow("temp_src2",temp_src2);
        cv::imshow("skel2",skel2);
        cv::waitKey(1);

        ros::spinOnce();
        loop_rate.sleep();
    }
    return 0;
}
