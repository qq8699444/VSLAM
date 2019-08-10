#include <iostream>
#include <ctime>
#include <math.h>

#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Geometry>

#include "sophus/so3.hpp"
#include "sophus/se3.hpp"

using SO3d = Sophus::SO3<double>;
using SE3d = Sophus::SE3<double>;

using namespace std;

int main(int argc,char*argv[])
{
    auto theta  = M_PI/7;
    Eigen::Vector3d v_3d = Eigen::Vector3d(0,0,1);
    Eigen::Matrix3d R = Eigen::AngleAxisd(theta,v_3d).toRotationMatrix();
    Eigen::Quaterniond   q = Eigen::Quaterniond(R);
    cout << "R :" << R << endl;

    SO3d    SO3_R(R);
    //SO3d    so3_v(v_3d[0],0.0,theta);    
    SO3d    SO3_q(q);

    cout << "so3_R :\n" << SO3_R.matrix() << endl;
    cout << "so3_q :\n" << SO3_q.matrix() << endl;


    Eigen::Vector3d so3 = SO3_R.log();
    cout << "so3 hat=\n" << SO3d::hat(so3)<<endl;
    cout << "so3 hat vee=\n" << SO3d::vee(SO3d::hat(so3)).transpose()<<endl;

    Eigen::Vector3d update_so3(1e-4,0,0);
    SO3d    SO3_update = SO3d::exp(update_so3) * SO3_R;
    cout << "SO3_update :" << SO3_update.log().transpose() << endl;



    ////////////////////////////////////////////////////
    //          SE

    Eigen::Vector3d t = Eigen::Vector3d(1,0,0);
    SE3d            SE3_Rt(R,t);
    SE3d            SE3_qt(q,t);
    cout << "SE3_Rt :" << SE3_Rt.log().transpose() << endl;
    cout << "SE3_qt :" << SE3_qt.log().transpose() << endl;

    auto se3 = SE3_Rt.log();
    cout << "so3 hat=\n" << SE3d::hat(se3)<<endl;
    cout << "so3 hat vee=\n" << SE3d::vee(SE3d::hat(se3)).transpose()<<endl;



    Eigen::Matrix<double,6,1>   update_se3;
    update_se3.setZero();
    update_se3(0,0) = 1e-4;

    SE3d            SE3_update = SE3d::exp(update_se3) * SE3_Rt;
    cout << "SE3_update :" << SE3_update.matrix() << endl;
    return 0;
}