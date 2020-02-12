#include <iostream>
#include <fstream>
#include <list>
#include <vector>
#include <chrono>
#include <ctime>
#include <climits>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/calib3d/calib3d.hpp>

#include <ceres/ceres.h>
#include <ceres/rotation.h>

using namespace std;


struct Measurement
{
    Measurement ( Eigen::Vector3d p, float g ) : pos_world ( p ), grayscale ( g ) {}
    Eigen::Vector3d pos_world;
    float grayscale;
};


inline Eigen::Vector3d project2Dto3D ( int x, int y, int d, float fx, float fy, float cx, float cy, float scale )
{
    float zz = float ( d ) /scale;
    float xx = zz* ( x-cx ) /fx;
    float yy = zz* ( y-cy ) /fy;
    return Eigen::Vector3d ( xx, yy, zz );
}

inline Eigen::Vector2d project3Dto2D ( double x, double y, double z, double fx, double fy, double cx, double cy )
{
    double u = fx*x/z+cx;
    double v = fy*y/z+cy;
    return Eigen::Vector2d ( u,v );
}


struct EdgeSE3ProjectDirect
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    EdgeSE3ProjectDirect(const Eigen::Vector3d &p_world, const float &grayscale, cv::Mat* const image)
     : x_world_(p_world), grayscale_(grayscale), image_(image) {}

     static void addCameraIntrinsics(const Eigen::Matrix3f &K)
    {
        fx_ = K(0, 0);
        fy_ = K(1, 1);
        cx_ = K(0, 2);
        cy_ = K(1, 2);
    }

    template<typename T>
    bool operator() (const T* const r_vec, const T* const t, T* residuals) const;

    static ceres::CostFunction* create(const Eigen::Vector3d &p_world, const float &grayscale, cv::Mat* image)
    {
        return new ceres::NumericDiffCostFunction<EdgeSE3ProjectDirect,ceres::CENTRAL,1,3,3>(new EdgeSE3ProjectDirect(p_world, grayscale, image));
    }

    inline float getPixelValue(float x, float y) const
    {
        uchar* data = &image_->data[(int)(y) * image_->step + (int)(x)];
        float xx = x - floor(x);
        float yy = y - floor(y);
        
        /* 求取分布在集中相邻４个像素点覆盖面积的像素灰度值 */
        return float((1 - xx) * (1 - yy) * data[0] + \
                     xx * (1 - yy) * data[1] + \
                     (1 - xx) * yy * data[image_->step] + \
                     xx * yy * data[image_->step + 1]);
    }
private:
    Eigen::Vector3d x_world_;                   // 3D point in world frame
    float grayscale_;                           // Measurement grayscale
    static float cx_, cy_, fx_, fy_;            // Camera intrinsics
    cv::Mat* image_ = nullptr;                  // reference image
};

float EdgeSE3ProjectDirect::cx_ = 0.f;
float EdgeSE3ProjectDirect::cy_ = 0.f;
float EdgeSE3ProjectDirect::fx_ = 0.f;
float EdgeSE3ProjectDirect::fy_ = 0.f;

template<typename T>
bool EdgeSE3ProjectDirect::operator() (const T* const r_vec, const T* const t, T* residuals) const
{
    T p_world[3] = {(T)x_world_.x(), (T)x_world_.y(), (T)x_world_.z()};

    T p_cam[3];
    ceres::AngleAxisRotatePoint(r_vec, p_world, p_cam);
    p_cam[0] += t[0];
    p_cam[1] += t[1];
    p_cam[2] += t[2];

    Eigen::Vector2d p2d = project3Dto2D(p_cam[0], p_cam[1], p_cam[2], fx_, fy_, cx_, cy_);
    //cout << "p2d:" << p2d.x() << "," << p2d.y() << endl;
    residuals[0] = (T)grayscale_ - (T)getPixelValue(p2d.x(), p2d.y());
    //cout << "residuals:" << residuals[0] << endl;
    return true;
}


bool poseEstimationDirect ( const vector< Measurement >& measurements, cv::Mat* gray, Eigen::Matrix3f& K, Eigen::Isometry3d& Tcw )
{
    EdgeSE3ProjectDirect::addCameraIntrinsics(K);

    double r_vec[3] = {0.f}, t[3] = {0.f};
    ceres::Problem problem;
    for (auto m : measurements) {
        ceres::CostFunction *costFunction = EdgeSE3ProjectDirect::create(m.pos_world, m.grayscale, gray);
        problem.AddResidualBlock(costFunction, nullptr, r_vec, t);
    }

    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_SCHUR;
    options.minimizer_progress_to_stdout = true;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    // 输出结果
    cout<<summary.BriefReport() <<endl;
                                                                                                                                                                                                                                                                                                        

    cv::Mat R_vec = (cv::Mat_<double>(3, 1) << r_vec[0], r_vec[1], r_vec[2]);
    cv::Mat R_matrix;
    cv::Rodrigues(R_vec, R_matrix);

    Eigen::Matrix3d R;
    R << R_matrix.at<double>(0, 0), R_matrix.at<double>(0, 1), R_matrix.at<double>(0, 2),
         R_matrix.at<double>(1, 0), R_matrix.at<double>(1, 1), R_matrix.at<double>(1, 2),
         R_matrix.at<double>(2, 0), R_matrix.at<double>(2, 1), R_matrix.at<double>(2, 2);
    
    Tcw.prerotate(R);
    Tcw.pretranslate(Eigen::Vector3d(t[0], t[1], t[2]));
    return true;
}

int main ( int argc, char** argv )
{
    if ( argc != 2 )
    {
        cout<<"usage: useLK path_to_dataset"<<endl;
        return 1;
    }

    srand ( ( unsigned int ) time ( 0 ) );
    string path_to_dataset = argv[1];
    string associate_file = path_to_dataset + "/associate.txt";

    ifstream fin ( associate_file );

    string rgb_file, depth_file, time_rgb, time_depth;
    cv::Mat color, depth, gray;
    vector<Measurement> measurements;
    // 相机内参
    float cx = 325.5;
    float cy = 253.5;
    float fx = 518.0;
    float fy = 519.0;
    float depth_scale = 1000.0;
    Eigen::Matrix3f K;
    K<<fx,0.f,cx,0.f,fy,cy,0.f,0.f,1.0f;

    Eigen::Isometry3d Tcw = Eigen::Isometry3d::Identity();

    cv::Mat prev_color;
    // 我们以第一个图像为参考，对后续图像和参考图像做直接法
    for ( int index=0; index<10; index++ )
    {
        cout<<"*********** loop "<<index<<" ************"<<endl;
        fin>>time_rgb>>rgb_file>>time_depth>>depth_file;
        color = cv::imread ( path_to_dataset+"/"+rgb_file );
        depth = cv::imread ( path_to_dataset+"/"+depth_file, -1 );
        if ( color.data==nullptr || depth.data==nullptr )
            continue; 
        cv::cvtColor ( color, gray, cv::COLOR_BGR2GRAY );
        if ( index ==0 )
        {
            // 对第一帧提取FAST特征点
            vector<cv::KeyPoint> keypoints;
            cv::Ptr<cv::FastFeatureDetector> detector = cv::FastFeatureDetector::create();
            detector->detect ( color, keypoints );
            for ( auto kp:keypoints )
            {
                // 去掉邻近边缘处的点
                if ( kp.pt.x < 20 || kp.pt.y < 20 || ( kp.pt.x+20 ) >color.cols || ( kp.pt.y+20 ) >color.rows )
                    continue;
                ushort d = depth.ptr<ushort> ( cvRound ( kp.pt.y ) ) [ cvRound ( kp.pt.x ) ];
                if ( d==0 )
                    continue;
                Eigen::Vector3d p3d = project2Dto3D ( kp.pt.x, kp.pt.y, d, fx, fy, cx, cy, depth_scale );
                float grayscale = float ( gray.ptr<uchar> ( cvRound ( kp.pt.y ) ) [ cvRound ( kp.pt.x ) ] );
                measurements.push_back ( Measurement ( p3d, grayscale ) );
            }
            prev_color = color.clone();
            continue;
        }
        // 使用直接法计算相机运动
        chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
        poseEstimationDirect ( measurements, &gray, K, Tcw );
        chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
        chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>> ( t2-t1 );
        cout<<"direct method costs time: "<<time_used.count() <<" seconds."<<endl;
        cout<<"Tcw="<<Tcw.matrix() <<endl;

        // plot the feature points
        cv::Mat img_show ( color.rows*2, color.cols, CV_8UC3 );
        prev_color.copyTo ( img_show ( cv::Rect ( 0,0,color.cols, color.rows ) ) );
        color.copyTo ( img_show ( cv::Rect ( 0,color.rows,color.cols, color.rows ) ) );
        for ( Measurement m:measurements )
        {
            if ( rand() > RAND_MAX/5 )
                continue;
            Eigen::Vector3d p = m.pos_world;
            Eigen::Vector2d pixel_prev = project3Dto2D ( p ( 0,0 ), p ( 1,0 ), p ( 2,0 ), fx, fy, cx, cy );
            Eigen::Vector3d p2 = Tcw*m.pos_world;
            Eigen::Vector2d pixel_now = project3Dto2D ( p2 ( 0,0 ), p2 ( 1,0 ), p2 ( 2,0 ), fx, fy, cx, cy );
            if ( pixel_now(0,0)<0 || pixel_now(0,0)>=color.cols || pixel_now(1,0)<0 || pixel_now(1,0)>=color.rows )
                continue;

            float b = 255*float ( rand() ) /RAND_MAX;
            float g = 255*float ( rand() ) /RAND_MAX;
            float r = 255*float ( rand() ) /RAND_MAX;
            cv::circle ( img_show, cv::Point2d ( pixel_prev ( 0,0 ), pixel_prev ( 1,0 ) ), 8, cv::Scalar ( b,g,r ), 2 );
            cv::circle ( img_show, cv::Point2d ( pixel_now ( 0,0 ), pixel_now ( 1,0 ) +color.rows ), 8, cv::Scalar ( b,g,r ), 2 );
            cv::line ( img_show, cv::Point2d ( pixel_prev ( 0,0 ), pixel_prev ( 1,0 ) ), cv::Point2d ( pixel_now ( 0,0 ), pixel_now ( 1,0 ) +color.rows ), cv::Scalar ( b,g,r ), 1 );
        }
        cv::imshow ( "result", img_show );
        cv::waitKey ( 0 );

    }
    return 0;
}