#include <iostream>
#include "tools.h"
#include <stdio.h>      /* printf */
#include <math.h>  

using Eigen::VectorXd;
using Eigen::MatrixXd;
using std::vector;

Tools::Tools() {
}

Tools::~Tools() {
}

VectorXd Tools::CalculateRMSE(const vector<VectorXd> &estimations,
        const vector<VectorXd> &ground_truth) {
    /**
    TODO:
     * Calculate the RMSE here.
     */

    VectorXd rmse(4);
    rmse << 0, 0, 0, 0;

    // check the validity of the following inputs:
    //  * the estimation vector size should not be zero
    //  * the estimation vector size should equal ground truth vector size
    if (estimations.size() != ground_truth.size()
            || estimations.size() == 0) {
        std::cout << "Invalid estimation or ground_truth data" << std::endl;
        return rmse;
    }

    //accumulate squared residuals
    for (unsigned int i = 0; i < estimations.size(); ++i) {

        VectorXd residual = estimations[i] - ground_truth[i];

        //coefficient-wise multiplication
        residual = residual.array() * residual.array();
        rmse += residual;
    }

    //calculate the mean
    rmse = rmse / estimations.size();

    //calculate the squared root
    rmse = rmse.array().sqrt();

    //return the result
    return rmse;


}

MatrixXd Tools::CalculateJacobian(const VectorXd& x_state) {
    /**
    TODO:
     * Calculate a Jacobian here.
     */

    MatrixXd Hj(3, 4);
    //recover state parameters
    float px = x_state(0);
    float py = x_state(1);
    float vx = x_state(2);
    float vy = x_state(3);

    std::cout << "CalculateJacobian ():" << std::endl;
    std::cout << "px: " << px << std::endl;
    std::cout << "py: " << py << std::endl;
    std::cout << "vx: " << vx << std::endl;
    std::cout << "vy: " << vy << std::endl;



    //pre-compute a set of terms to avoid repeated calculation
    float c1 = px * px + py*py;
    float c2 = sqrt(c1);
    float c3 = (c1 * c2);

    //check division by zero
    if (std::fabs(c1) < 0.0001) {
        std::cout << "CalculateJacobian () - Error - Division by Zero" << std::endl;
        return Hj;
    }

    //compute the Jacobian matrix
    Hj << (px / c2), (py / c2), 0, 0,
            -(py / c1), (px / c1), 0, 0,
            py * (vx * py - vy * px) / c3, px * (px * vy - py * vx) / c3, px / c2, py / c2;

    return Hj;
}

VectorXd Tools::CalculateHFunction(const VectorXd& x_state) {

    VectorXd hf(3);
    //recover state parameters
    float px = x_state(0);
    float py = x_state(1);
    float vx = x_state(2);
    float vy = x_state(3);

    std::cout << "CalculateHFunction ():" << std::endl;
    std::cout << "px: " << px << std::endl;
    std::cout << "py: " << py << std::endl;
    std::cout << "vx: " << vx << std::endl;
    std::cout << "vy: " << vy << std::endl;

    float v1 = sqrt(px * px + py * py);
    float v2 = atan(py / px);
    float v3 = (px * vx + py * vy) / v1;

    hf << v1, v2, v3;
    return hf;
}
