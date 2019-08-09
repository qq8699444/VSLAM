#include <iostream>
#include <ctime>


#include <Eigen/Core>
#include <Eigen/Dense>

using namespace std;

#define MATRIX_SIZE 50
int main(int argc,char*argv[])
{
    Eigen::Matrix<float,2,3>    matrix_23;
    Eigen::Vector3d v_3d;

    Eigen::Matrix3d matrix_33 = Eigen::Matrix3d::Zero();
    Eigen::Matrix<float,Eigen::Dynamic,Eigen::Dynamic>    matrix_dynamic;

    Eigen::MatrixXd matrix_x;


    //input
    matrix_23 << 1,2,3,4,5,6;


    //print
    for (size_t i = 0; i < 2; i++)
    {
        for (size_t j = 0; j < 3; j++)
        {
            cout << matrix_23(i,j) << " ";
        }
        cout << endl;
    }
    
    //multi
    v_3d << 1,2,3;
    Eigen::Matrix<double,2,1>    result = matrix_23.cast<double>() * v_3d;
    cout << "result:\n" << result << endl;

    matrix_33 = Eigen::Matrix3d::Random();
    cout << "matrix_33:\n" << matrix_33 << endl;
    cout << "transpose:\n" << matrix_33.transpose() << endl;
    cout << "sum:\n" << matrix_33.sum() << endl;
    cout << "trace:\n" << matrix_33.trace() << endl;
    cout << "10*:\n" << 10* matrix_33  << endl;
    cout << "inverse:\n" << matrix_33.inverse() << endl;
    cout << "det:\n" << matrix_33.determinant() << endl;

    //eigen decompose
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d>  eigen_solver (matrix_33.transpose() * matrix_33);
    cout << "eigen values:\n" << eigen_solver.eigenvalues() << endl;
    cout << "eigen values:\n" << eigen_solver.eigenvectors() << endl;

    //
    Eigen::Matrix<double,MATRIX_SIZE,MATRIX_SIZE>    matrix_NN;
    matrix_NN = Eigen::MatrixXd::Random(MATRIX_SIZE,MATRIX_SIZE);
    Eigen::Matrix<double,MATRIX_SIZE,1>    v_Nd;
    v_Nd = Eigen::MatrixXd::Random(MATRIX_SIZE,1);

    clock_t time_stt  = clock();
    Eigen::Matrix<double,MATRIX_SIZE,1>    x = matrix_NN.inverse()*v_Nd;
    cout << "time use in normal inverse is :" << 1000.f *(clock() - time_stt)/CLOCKS_PER_SEC << "ms" << endl ;


    time_stt  = clock();
    x = matrix_NN.colPivHouseholderQr().solve(v_Nd);
    cout << "time use in QR composition is :" << 1000.f *(clock() - time_stt)/CLOCKS_PER_SEC << "ms" << endl ;

    return 0;
}
