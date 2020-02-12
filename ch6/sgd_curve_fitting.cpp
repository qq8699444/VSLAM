#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>

#include <opencv2/core/core.hpp>

using namespace std;

int main ( int argc, char** argv )
{  
    double a=1.0, b=2.0, c=1.0;         // 真实参数值
    int N=100;                          // 数据点
    double w_sigma=1.0;                 // 噪声Sigma值
    cv::RNG rng;                        // OpenCV随机数产生器
    double abc[3] = {1.0,1.0,1.0};            // abc参数的估计值

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

    double lr = 0.002;
    cout<<"start fitting: "<<endl;
    chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
    for (size_t i = 0; i < 200; i++)
    {
        double delta_a =    0.0;
        double delta_b =    0.0;
        double delta_c =    0.0; 
        double cost = 0;

        for (int n = 0; n < N; n++)
        {
            double x = x_data[n];
            double predict = std::exp(abc[0]*x*x + abc[1]*x + abc[2]);

            delta_a += (predict - y_data[n]) * predict *x*x;
            delta_b += (predict - y_data[n]) * predict *x;
            delta_c += (predict - y_data[n]) * predict;

            cost += (predict - y_data[n]) * (predict - y_data[n]);
        }
        
        delta_a /= N;
        delta_b /= N;
        delta_c /= N;

        abc[0] -= lr*delta_a;
        abc[1] -= lr*delta_b;
        abc[2] -= lr*delta_c;
        cout << "curr abc:" << abc[0] << "," << abc[1] << "," << abc[2] << " --> cost:" << cost << endl;
    }
    chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
    chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>>( t2-t1 );
    cout<<"solve time cost = "<<time_used.count()<<" seconds. "<<endl;

    
    cout<<"estimated a,b,c = ";
    for ( auto a:abc ) cout<<a<<" ";
    cout<<endl;
    return 0;

}