#include <iostream>
#include <ctime>


#include <Eigen/Core>
#include <Eigen/Dense>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
using namespace std;

#define MATRIX_SIZE 50
int main(int argc,char*argv[])
{
    cv::Mat img;
    img = cv::imread(argv[1]);

    if (img.empty())
    {
        cout <<"failed to read picture " << endl;
        return -1;
    }

    cout << "img row:" << img.rows << endl;
    cout << "img col:" << img.cols << endl;
    cv::imshow("img",img);
    cv::waitKey(0);
    
    return 0;
}
