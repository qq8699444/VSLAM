#include <iostream>
#include <ctime>
#include <math.h>

#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Geometry>

using namespace std;

int main(int argc,char*argv[])
{
    Eigen::Matrix3d rotation_mat = Eigen::Matrix3d::Identity();
    Eigen::Vector3d v_3d = Eigen::Vector3d(0,0,1);
    Eigen::Matrix3d Skew_matrix;
    Skew_matrix << 0, -v_3d(2), v_3d(1), v_3d(2), 0, -v_3d(0),-v_3d(1), v_3d(0), 0;

    Eigen::AngleAxisd rotation_vector(M_PI/7,v_3d);

    cout << "rotation_mat:\n" << rotation_vector.matrix() << endl;

    auto rotation_mat2 = (1-cos(M_PI/7))*v_3d*v_3d.transpose() + sin(M_PI/7)*Skew_matrix + cos(M_PI/7)*Eigen::Matrix3d::Identity();
    cout << "rotation_mat2:\n" << rotation_mat2 << endl;
    //
    rotation_mat = rotation_vector.toRotationMatrix();

    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d>  eigen_solver (rotation_mat);
    cout << "eigen values:\n" << eigen_solver.eigenvalues() << endl;
    cout << "eigen values:\n" << eigen_solver.eigenvectors() << endl;


    auto theta =  acosf(0.5f*(rotation_mat.trace() -1));
    cout << "theta:\n" << theta << endl;
    cout << "theta diff:\n" << theta - M_PI/7<< endl;


    Eigen::Vector3d v (1,1,1);
    cout << "v after AngleAxisd:\n" << rotation_vector * v<< endl;
    cout << "v after rotationMat:\n" << rotation_mat * v<< endl;


    //euler angles
    Eigen::Vector3d euler_angles = rotation_mat.eulerAngles(2,1,0);
    cout << "yaw pitch roll:\n" << euler_angles .transpose() << endl;

    //euclidean transform
    Eigen::Isometry3d   T = Eigen::Isometry3d::Identity();
    T.rotate(rotation_vector);
    T.pretranslate(Eigen::Vector3d(1,1,1));
    cout << "transform Matrix:\n" << T.matrix() << endl;

    Eigen::Vector3d transform_v  = T*v;
    cout << "transformed v:\n" << transform_v.transpose() << endl;


    Eigen::Quaterniond   q = Eigen::Quaterniond(rotation_vector);
    cout << "Quaterniond :\n" << q.coeffs() << endl;
    Eigen::Vector4d qcoeffs2 (v_3d[0]*sin(M_PI/7/2),v_3d[1]*sin(M_PI/7/2),v_3d[2]*sin(M_PI/7/2),cos(M_PI/7/2));
    cout << "Quaternion2 :\n" << qcoeffs2 << endl;

    q = Eigen::Quaterniond(rotation_mat);
    cout << "Quaterniond from rotate matirx 1:\n" << q.coeffs() << endl;

    auto q0 = sqrtf(rotation_mat.trace()+1)/2;
    auto q1 = (rotation_mat(1,2) - rotation_mat(2,1))/(4*q0);
    auto q2 = (rotation_mat(2,0) - rotation_mat(0,2))/(4*q0);
    auto q3 = (rotation_mat(0,1) - rotation_mat(1,0))/(4*q0);
    qcoeffs2 << q1,q2,q3,q0;
    cout << "Quaterniond from rotate matirx 2:\n" << qcoeffs2 << endl;
    return 0;
}
