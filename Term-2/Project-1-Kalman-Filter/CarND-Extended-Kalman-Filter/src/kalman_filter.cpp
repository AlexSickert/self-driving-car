#include "kalman_filter.h"
#include <stdlib.h>
#include <iostream>
#include "tools.h"
#include <stdio.h>      /* printf */


using Eigen::MatrixXd;
using Eigen::VectorXd;

KalmanFilter::KalmanFilter() {
}

KalmanFilter::~KalmanFilter() {
}

void KalmanFilter::Init(VectorXd &x_in, MatrixXd &P_in, MatrixXd &F_in,
        MatrixXd &H_in, MatrixXd &R_in, MatrixXd &Q_in) {
    x_ = x_in;
    P_ = P_in;
    F_ = F_in;
    H_ = H_in;
    R_ = R_in;
    Q_ = Q_in;
}

void KalmanFilter::Predict() {
    /**
    TODO:
     * predict the state
     */

    //    std::cout << "KalmanFilter::Predict()" << std::endl;
    //    std::cout << "F_" << F_ << std::endl;
    //    std::cout << "P_" << P_ << std::endl;
    //    std::cout << "Q_" << Q_ << std::endl;

    x_ = F_ * x_;
    MatrixXd Ft = F_.transpose();
    P_ = F_ * P_ * Ft + Q_;


}

void KalmanFilter::Update(const VectorXd &z) {
    /**
    TODO:
     * update the state by using Kalman Filter equations
     */

    std::cout << "KalmanFilter::Update()" << std::endl;

    VectorXd z_pred = H_ * x_;
    VectorXd y = z - z_pred;
    MatrixXd Ht = H_.transpose();
    MatrixXd S = H_ * P_ * Ht + R_;
    MatrixXd Si = S.inverse();
    MatrixXd PHt = P_ * Ht;
    MatrixXd K = PHt * Si;

    //new estimate
    x_ = x_ + (K * y);
    long x_size = x_.size();
    MatrixXd I = MatrixXd::Identity(x_size, x_size);
    P_ = (I - K * H_) * P_;

}




VectorXd KalmanFilter::CalculateZ_previous(const VectorXd &z) {
  VectorXd z_pred;
  if (z.size() == 2) {
    z_pred = H_ * x_;
  } else {
    double px = x_(0);
    double py = x_(1);
    double vx = x_(2);
    double vy = x_(3);
  
    double rho = sqrt(px*px + py*py);
    double phi = atan2(py, px);
    double rho_dot = (px*vx + py*vy)/rho;
    z_pred = VectorXd(3);
    z_pred << rho, phi, rho_dot;
  }
  return z_pred;
}

void KalmanFilter::UpdateEKF(const VectorXd &z) {
    
    
    VectorXd z_pred = CalculateZ_previous(z);
    
    
    VectorXd y = z - z_pred;
    MatrixXd Ht = H_.transpose();
    MatrixXd S = H_ * P_ * Ht + R_;
    MatrixXd K = P_ * Ht * S.inverse();

    x_ = x_ + (K * y);
    long x_size = x_.size();
    MatrixXd I = MatrixXd::Identity(x_size, x_size);
    P_ = (I - K * H_) * P_;
}
