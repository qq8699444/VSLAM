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


/********************************************
 * 本节演示了RGBD上的半稠密直接法 
 ********************************************/

// 一次测量的值，包括一个世界坐标系下三维点与一个灰度值
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

inline Eigen::Vector2d project3Dto2D ( float x, float y, float z, float fx, float fy, float cx, float cy )
{
    float u = fx*x/z+cx;
    float v = fy*y/z+cy;
    return Eigen::Vector2d ( u,v );
}

class SparseSE3ProjectDirectCost : public ceres::SizedCostFunction<1,6>
{
public:
    SparseSE3ProjectDirectCost(const Eigen::Vector3d &p_world, const float &grayscale, cv::Mat* const image)
     : x_world_(p_world), grayscale_(grayscale), image_(image) {}

     virtual ~SparseSE3ProjectDirectCost() {}

     static void addCameraIntrinsics(const Eigen::Matrix3f &K)
    {
        fx_ = K(0, 0);
        fy_ = K(1, 1);
        cx_ = K(0, 2);
        cy_ = K(1, 2);
    }    

    virtual bool Evaluate(double const* const* parameters,
                        double* residuals,
                        double** jacobians) const ;

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

float SparseSE3ProjectDirectCost::cx_ = 0.f;
float SparseSE3ProjectDirectCost::cy_ = 0.f;
float SparseSE3ProjectDirectCost::fx_ = 0.f;
float SparseSE3ProjectDirectCost::fy_ = 0.f;

bool SparseSE3ProjectDirectCost::Evaluate (double const* const* parameters,
                        double* residuals,
                        double** jacobians)  const
{
    double p_world[3] = {x_world_.x(), x_world_.y(), x_world_.z()};

    double p_cam[3];
    ceres::AngleAxisRotatePoint(parameters[0], p_world, p_cam);
    p_cam[0] += parameters[0][3];
    p_cam[1] += parameters[0][4];
    p_cam[2] += parameters[0][5];

    double x = p_cam[0];
    double y = p_cam[1];
    double z = p_cam[2];
    double invz = 1.0/z;
    double invz_2 = invz*invz;

    double u = x*fx_*invz + cx_;
    double v = y*fy_*invz + cy_;
    //cout << "p2d:" << p2d.x() << "," << p2d.y() << endl;
    if ( u-4<0 || ( u+4 ) >image_->cols || ( v-4 ) <0 || ( v+4 ) >image_->rows )
    {
        residuals[0] = 0;
    }
    else
    {
        residuals[0] = getPixelValue(u, v) - grayscale_;    
    }
    
    //cout << "residuals:" << residuals[0] << endl;

    if (jacobians != NULL && jacobians[0] != NULL) {
        if ( u-4<0 || ( u+4 ) >image_->cols || ( v-4 ) <0 || ( v+4 ) >image_->rows )
        {
            jacobians[0][0] = 0.;
            jacobians[0][1] = 0.;
            jacobians[0][2] = 0.;
            jacobians[0][3] = 0.;
            jacobians[0][4] = 0.;
            jacobians[0][5] = 0.;
        }

        Eigen::Matrix<double, 2, 6> jacobian_uv_ksai;

        jacobian_uv_ksai ( 0,0 ) = - x*y*invz_2 *fx_;
        jacobian_uv_ksai ( 0,1 ) = ( 1+ ( x*x*invz_2 ) ) *fx_;
        jacobian_uv_ksai ( 0,2 ) = - y*invz *fx_;
        jacobian_uv_ksai ( 0,3 ) = invz *fx_;
        jacobian_uv_ksai ( 0,4 ) = 0;
        jacobian_uv_ksai ( 0,5 ) = -x*invz_2 *fx_;

        jacobian_uv_ksai ( 1,0 ) = - ( 1+y*y*invz_2 ) *fy_;
        jacobian_uv_ksai ( 1,1 ) = x*y*invz_2 *fy_;
        jacobian_uv_ksai ( 1,2 ) = x*invz *fy_;
        jacobian_uv_ksai ( 1,3 ) = 0;
        jacobian_uv_ksai ( 1,4 ) = invz *fy_;
        jacobian_uv_ksai ( 1,5 ) = -y*invz_2 *fy_;

        Eigen::Matrix<double, 1, 2> jacobian_pixel_uv;

        jacobian_pixel_uv ( 0,0 ) = ( getPixelValue ( u+1,v )-getPixelValue ( u-1,v ) ) /2;
        jacobian_pixel_uv ( 0,1 ) = ( getPixelValue ( u,v+1 )-getPixelValue ( u,v-1 ) ) /2;

        auto _jacobianRtVec = jacobian_pixel_uv*jacobian_uv_ksai;

        jacobians[0][0] = _jacobianRtVec(0,0);
        jacobians[0][1] = _jacobianRtVec(0,1);
        jacobians[0][2] = _jacobianRtVec(0,2);
        jacobians[0][3] = _jacobianRtVec(0,3);
        jacobians[0][4] = _jacobianRtVec(0,4);
        jacobians[0][5] = _jacobianRtVec(0,5);
    }
    return true;
}


bool poseEstimationDirect ( const vector< Measurement >& measurements, cv::Mat* gray, Eigen::Matrix3f& K, Eigen::Isometry3d& Tcw )
{
    // 初始化g2o
    SparseSE3ProjectDirectCost::addCameraIntrinsics(K);

    double rt_vec[6] = {0.f};
    ceres::Problem problem;
    for (auto m : measurements) {
        ceres::CostFunction *costFunction = new SparseSE3ProjectDirectCost(m.pos_world, m.grayscale, gray);
        problem.AddResidualBlock(costFunction, nullptr, rt_vec);
    }

    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_QR;
    options.minimizer_progress_to_stdout = true;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    // 输出结果
    cout<<summary.BriefReport() <<endl;
                                                                                                                                                                                                                                                                                                        

    cv::Mat R_vec = (cv::Mat_<double>(3, 1) << rt_vec[0], rt_vec[1], rt_vec[2]);
    cv::Mat R_matrix;
    cv::Rodrigues(R_vec, R_matrix);

    Eigen::Matrix3d R;
    R << R_matrix.at<double>(0, 0), R_matrix.at<double>(0, 1), R_matrix.at<double>(0, 2),
         R_matrix.at<double>(1, 0), R_matrix.at<double>(1, 1), R_matrix.at<double>(1, 2),
         R_matrix.at<double>(2, 0), R_matrix.at<double>(2, 1), R_matrix.at<double>(2, 2);
    
    Tcw.prerotate(R);
    Tcw.pretranslate(Eigen::Vector3d(rt_vec[3], rt_vec[4], rt_vec[5]));
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
            // select the pixels with high gradiants 
            for ( int x=10; x<gray.cols-10; x++ )
                for ( int y=10; y<gray.rows-10; y++ )
                {
                    Eigen::Vector2d delta (
                        gray.ptr<uchar>(y)[x+1] - gray.ptr<uchar>(y)[x-1], 
                        gray.ptr<uchar>(y+1)[x] - gray.ptr<uchar>(y-1)[x]
                    );
                    if ( delta.norm() < 50 )
                        continue;
                    ushort d = depth.ptr<ushort> (y)[x];
                    if ( d==0 )
                        continue;
                    Eigen::Vector3d p3d = project2Dto3D ( x, y, d, fx, fy, cx, cy, depth_scale );
                    float grayscale = float ( gray.ptr<uchar> (y) [x] );
                    measurements.push_back ( Measurement ( p3d, grayscale ) );
                }
            prev_color = color.clone();
            cout<<"add total "<<measurements.size()<<" measurements."<<endl;
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

            float b = 0;
            float g = 250;
            float r = 0;
            img_show.ptr<uchar>( pixel_prev(1,0) )[int(pixel_prev(0,0))*3] = b;
            img_show.ptr<uchar>( pixel_prev(1,0) )[int(pixel_prev(0,0))*3+1] = g;
            img_show.ptr<uchar>( pixel_prev(1,0) )[int(pixel_prev(0,0))*3+2] = r;
            
            img_show.ptr<uchar>( pixel_now(1,0)+color.rows )[int(pixel_now(0,0))*3] = b;
            img_show.ptr<uchar>( pixel_now(1,0)+color.rows )[int(pixel_now(0,0))*3+1] = g;
            img_show.ptr<uchar>( pixel_now(1,0)+color.rows )[int(pixel_now(0,0))*3+2] = r;
            cv::circle ( img_show, cv::Point2d ( pixel_prev ( 0,0 ), pixel_prev ( 1,0 ) ), 4, cv::Scalar ( b,g,r ), 2 );
            cv::circle ( img_show, cv::Point2d ( pixel_now ( 0,0 ), pixel_now ( 1,0 ) +color.rows ), 4, cv::Scalar ( b,g,r ), 2 );
        }
        cv::imshow ( "result", img_show );
        cv::waitKey ( 0 );

    }
    return 0;
}