#include <iostream>
#include <opencv2/core/core.hpp>
#include <ceres/ceres.h>
#include <chrono>

using namespace std;


struct CURVE_FITTING_COST
{
    CURVE_FITTING_COST(double x_,double y_):x(x_),y(y_) {}

    template<typename T>
    bool operator() (const T* abc,T* residual )  const
    {
        //exp(ax^2+bx+c)
        residual[0] = T(y) - ceres::exp ( abc[0]*T ( x ) *T ( x ) + abc[1]*T ( x ) + abc[2] );
        return true;
    }

    const double x,y;
};


class CURVE_FITTING_COST2:public ceres::SizedCostFunction<1,3>
{
public:
    CURVE_FITTING_COST2(double x_,double y_):x(x_),y(y_) {}
    virtual ~CURVE_FITTING_COST2() {}

    virtual bool Evaluate(double const* const* parameters,
                        double* residuals,
                        double** jacobians) const {
    const double a = parameters[0][0];
    const double b = parameters[0][1];
    const double c = parameters[0][2];
    auto result = ceres::exp ( a*x*x + b*x+ c);
    residuals[0] = y - result;

    // Compute the Jacobian if asked for.
    if (jacobians != NULL && jacobians[0] != NULL) {
      jacobians[0][0] = -result * x *x;
      jacobians[0][1] = -result * x;
      jacobians[0][2] = -result;
    }
    return true;
  }

private:
  const double x,y;
};


int main ( int argc, char** argv )
{   
    double a=1.0, b=2.0, c=1.0;         // 真实参数值
    int N=100;                          // 数据点
    double w_sigma=1.0;                 // 噪声Sigma值
    cv::RNG rng;                        // OpenCV随机数产生器
    double abc[3] = {0,0,0};            // abc参数的估计值

    vector<double> x_data, y_data;      // 数据

    cout<<"generating data: "<<endl;
    for ( int i=0; i<N; i++ )
    {
        double x = i/100.0;
        x_data.push_back ( x );
        y_data.push_back (
            exp ( a*x*x + b*x + c ) + rng.gaussian ( w_sigma )
        );
        //cout<<x_data[i]<<" "<<y_data[i]<<endl;
    }

    ceres::Problem problem;

    for (size_t i = 0; i < N; i++)
    {
#if 1
    problem.AddResidualBlock(
            new CURVE_FITTING_COST2(x_data[i],y_data[i]) ,
            nullptr,
            abc
        );
#else
        /* code */
        problem.AddResidualBlock(
            new ceres::AutoDiffCostFunction<CURVE_FITTING_COST,1,3>(
                  new CURVE_FITTING_COST(x_data[i],y_data[i])      
            ),
            nullptr,
            abc
        );
    #endif
    }
    
    ceres::Solver::Options options;     // 这里有很多配置项可以填
    options.linear_solver_type = ceres::DENSE_QR;  // 增量方程如何求解
    options.minimizer_progress_to_stdout = true;   // 输出到cout

    ceres::Solver::Summary summary;                // 优化信息
    chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
    ceres::Solve ( options, &problem, &summary );  // 开始优化
    chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
    chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>>( t2-t1 );
    cout<<"solve time cost = "<<time_used.count()<<" seconds. "<<endl;

    // 输出结果
    cout<<summary.BriefReport() <<endl;
    cout<<"estimated a,b,c = ";
    for ( auto a:abc ) cout<<a<<" ";
    cout<<endl;
    return 0;
}