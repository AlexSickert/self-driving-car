#include "ukf.h"
#include "tools.h"
#include "Eigen/Dense"
#include <iostream>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

/**
 * Initializes Unscented Kalman filter
 */
UKF::UKF() {

    is_initialized_ = false;

    // if this is false, laser measurements will be ignored (except during init)
    use_laser_ = true;

    // if this is false, radar measurements will be ignored (except during init)
    use_radar_ = true;

    // initial state vector
    x_ = VectorXd(5);

    // initial covariance matrix
    P_ = MatrixXd(5, 5);

    // Process noise standard deviation longitudinal acceleration in m/s^2
    std_a_ = .6; //  .63;

    // Process noise standard deviation yaw acceleration in rad/s^2
    std_yawdd_ = 1.2; // 1.2;

    // Laser measurement noise standard deviation position1 in m
    std_laspx_ = 0.15;
    std_laspx_sq_ = std_laspx_ * std_laspx_;

    // Laser measurement noise standard deviation position2 in m
    std_laspy_ = 0.15;
    std_laspy_sq_ = std_laspy_ * std_laspy_;

    // Radar measurement noise standard deviation radius in m
    std_radr_ = 0.3; //0.9;

    // Radar measurement noise standard deviation angle in rad
    std_radphi_ = 0.03; // 0.005;

    // Radar measurement noise standard deviation radius change in m/s
    std_radrd_ = 0.3; // 0.5;

    /**
    TODO:

    Complete the initialization. See ukf.h for other member properties.

    Hint: one or more values initialized above might be wildly off...
     */

    //    Xsig_pred_ = MatrixXd(11, 5);

    time_us_ = 0;
    n_x_ = 5;
    n_aug_ = n_x_ + 2;
    lambda_ = 3 - n_x_;

    Q_ = MatrixXd(2, 2);
    Q_ << std_a_ * std_a_, 0,
            0, std_yawdd_ * std_yawdd_;

    // initial covariance matrix
    P_ = MatrixXd(5, 5);
    P_ << 1, 0, 0, 0, 0,
            0, 1, 0, 0, 0,
            0, 0, 100, 0, 0,
            0, 0, 0, 10, 0,
            0, 0, 0, 0, 1;

    n_z_radar_ = 3;
    Zsig_ = MatrixXd(n_z_radar_, 2 * n_aug_ + 1);

    Xsig_pred_ = MatrixXd(n_x_, 2 * n_aug_ + 1);

    weights_ = VectorXd(2 * n_aug_ + 1);

    R_radar_ = MatrixXd(n_z_radar_, n_z_radar_);
    R_radar_ << std_radr_ * std_radr_, 0, 0,
            0, std_radphi_ * std_radphi_, 0,
            0, 0, std_radrd_ * std_radrd_;

    H_laser_ = MatrixXd(2, 5);
    H_laser_ << 1, 0, 0, 0, 0,
            0, 1, 0, 0, 0;

    R_laser_ = MatrixXd(2, 2);

    R_laser_ << std_laspx_sq_, 0,
            0, std_laspy_sq_;

    lambda_n_aug_ = sqrt(lambda_ + n_aug_);

}

UKF::~UKF() {
}

/**
 * @param {MeasurementPackage} meas_package The latest measurement data of
 * either radar or laser.
 */
void UKF::ProcessMeasurement(MeasurementPackage meas_package) {
    cout << "UKF::ProcessMeasurement" << endl;

    if (!is_initialized_) {
        time_us_ = meas_package.timestamp_;

        std::cout << "UKF: " << std::endl;

        double px = 0;
        double py = 0;

        if (meas_package.sensor_type_ == MeasurementPackage::RADAR) {
            double rho = meas_package.raw_measurements_[0];
            double phi = meas_package.raw_measurements_[1];

            px = rho * cos(phi);
            py = rho * sin(phi);


            if (fabs(px) < 0.0001) {
                px = 1;
                P_(0, 0) = 1000;
            }
            if (fabs(py) < 0.0001) {
                py = 1;
                P_(1, 1) = 1000;
            }

        } else if (meas_package.sensor_type_ == MeasurementPackage::LASER) {


            px = meas_package.raw_measurements_[0];
            py = meas_package.raw_measurements_[1];
        }


        std::cout << px << std::endl;
        std::cout << py << std::endl;
        x_ << px, py, 0, 0, 0;
        x_aug_ = VectorXd(7);

        is_initialized_ = true;

        std::cout << "initialization done" << std::endl;
        return;

    }

    dt_ = (meas_package.timestamp_ - time_us_) / 1000000.0;
    time_us_ = meas_package.timestamp_;


    Prediction(dt_);

    std::cout << "prediction done" << std::endl;

    if (meas_package.sensor_type_ == MeasurementPackage::LASER) {
        cout << "Laser" << endl;
        UpdateLidar(meas_package);
    } else if (meas_package.sensor_type_ == MeasurementPackage::RADAR) {
        cout << "Radar" << endl;
        UpdateRadar(meas_package);
    }

    std::cout << "update done" << std::endl;

}

/**
 * Predicts sigma points, the state, and the state covariance matrix.
 * @param {double} delta_t the change in time (in seconds) between the last
 * measurement and this one.
 */
void UKF::Prediction(double delta_t) {

    cout << "UKF::Prediction " << endl;

    MatrixXd Xsig_out = MatrixXd(11, 5);

    AugmentedSigmaPoints(&Xsig_out);

    SigmaPointPrediction(&Xsig_out);

    PredictMeanAndCovariance(&x_, &P_, &Xsig_out);

}

/**
 * Updates the state and the state covariance matrix using a laser measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateLidar(MeasurementPackage meas_package) {

    cout << "UKF::UpdateLidar" << endl;

    VectorXd z = meas_package.raw_measurements_;

    VectorXd z_pred = H_laser_ * x_;
    VectorXd z_diff = z - z_pred;
    MatrixXd Ht = H_laser_.transpose();
    MatrixXd S = H_laser_ * P_ * Ht + R_laser_;
    MatrixXd Si = S.inverse();
    MatrixXd PHt = P_ * Ht;
    MatrixXd K = PHt * Si;

    //new estimate
    x_ = x_ + (K * z_diff);
    long x_size = x_.size();
    MatrixXd I = MatrixXd::Identity(x_size, x_size);
    P_ = (I - K * H_laser_) * P_;
    NIS_laser_ = z_diff.transpose() * Si * z_diff;

}

/**
 * Updates the state and the state covariance matrix using a radar measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateRadar(MeasurementPackage meas_package) {

    cout << "UKF::UpdateRadar" << endl;

    for (int i = 0; i < 2 * n_aug_ + 1; i++) {

        double p_x = Xsig_pred_(0, i);
        double p_y = Xsig_pred_(1, i);


        if (p_x < 0.001) {
            p_x = 0.001;
        }

        if (p_y < 0.001) {
            p_y = 0.001;
        }


        double v = Xsig_pred_(2, i);
        double yaw = Xsig_pred_(3, i);

        double v_y = cos(yaw) * v;
        double v_x = sin(yaw) * v;

        double rho = sqrt(p_x * p_x + p_y * p_y);
        double phi = atan2(p_y, p_x);
        double rho_dot = (p_x * v_y + p_y * v_x) / rho;

      

        Zsig_(0, i) = rho;
        Zsig_(1, i) = phi;
        Zsig_(2, i) = rho_dot;
    }

    VectorXd z_pred = VectorXd(n_z_radar_);

    z_pred.fill(0.0);

    for (int i = 0; i < 2 * n_aug_ + 1; i++) {
        z_pred = z_pred + weights_(i) * Zsig_.col(i);
    }

    MatrixXd S = MatrixXd(n_z_radar_, n_z_radar_);
    S.fill(0.0);
    for (int i = 0; i < 2 * n_aug_ + 1; i++) { //2n+1 simga points

        VectorXd z_diff = Zsig_.col(i) - z_pred;


        while (z_diff(1) > M_PI) z_diff(1) -= 2. * M_PI;
        while (z_diff(1) < -M_PI) z_diff(1) += 2. * M_PI;

        S = S + weights_(i) * z_diff * z_diff.transpose();
    }

    S = S + R_radar_;

    MatrixXd Tc = MatrixXd(n_x_, n_z_radar_);
    Tc.fill(0.0);
    for (int i = 0; i < 2 * n_aug_ + 1; i++) { //2n+1 simga points

        VectorXd z_diff = Zsig_.col(i) - z_pred;

        while (z_diff(1) > M_PI) z_diff(1) -= 2. * M_PI;
        while (z_diff(1) < -M_PI) z_diff(1) += 2. * M_PI;


        VectorXd x_diff = Xsig_pred_.col(i) - x_;

        while (x_diff(3) > M_PI) x_diff(3) -= 2. * M_PI;
        while (x_diff(3) < -M_PI) x_diff(3) += 2. * M_PI;

        Tc = Tc + weights_(i) * x_diff * z_diff.transpose();
    }


    MatrixXd K = Tc * S.inverse();


    VectorXd z = meas_package.raw_measurements_;
    VectorXd z_diff = z - z_pred;


    while (z_diff(1) > M_PI) z_diff(1) -= 2. * M_PI;
    while (z_diff(1) < -M_PI) z_diff(1) += 2. * M_PI;


    x_ = x_ + K * z_diff;
    P_ = P_ - K * S * K.transpose();

    NIS_radar_ = z_diff.transpose() * S.inverse() * z_diff;

}



void UKF::AugmentedSigmaPoints(MatrixXd* Xsig_out) {

    std::cout << "UKF::AugmentedSigmaPoints" << std::endl;

    //create augmented mean vector
    VectorXd x_aug = VectorXd(7);

    //create augmented state covariance
    MatrixXd P_aug = MatrixXd(7, 7);

    //create sigma point matrix
    MatrixXd Xsig_aug = MatrixXd(n_aug_, 2 * n_aug_ + 1);



    //create augmented mean state
    x_aug.head(5) = x_;
    x_aug(5) = 0;
    x_aug(6) = 0;

    //create augmented covariance matrix
    P_aug.fill(0.0);
    P_aug.topLeftCorner(5, 5) = P_;
    P_aug(5, 5) = std_a_ * std_a_;
    P_aug(6, 6) = std_yawdd_ * std_yawdd_;

    //create square root matrix
    MatrixXd L = P_aug.llt().matrixL();

    //create augmented sigma points
    Xsig_aug.col(0) = x_aug;
    
    
    
    for (int i = 0; i < n_aug_; i++) {
        Xsig_aug.col(i + 1) = x_aug + lambda_n_aug_ * L.col(i);
        Xsig_aug.col(i + 1 + n_aug_) = x_aug - lambda_n_aug_ * L.col(i);
    }


    //print result
    std::cout << "Xsig_aug = " << std::endl << Xsig_aug << std::endl;

    //write result
    *Xsig_out = Xsig_aug;

}

void UKF::SigmaPointPrediction(MatrixXd* Xsig_augX) {

    MatrixXd Xsig_aug = *Xsig_augX;

    MatrixXd Xsig_pred = MatrixXd(n_x_, 2 * n_aug_ + 1);

    std::cout << "Xsig_aug = " << std::endl << Xsig_aug << std::endl;

    //predict sigma points
    for (int i = 0; i < 2 * n_aug_ + 1; i++) {
        //extract values for better readability

        double p_x = Xsig_aug(0, i);
        double p_y = Xsig_aug(1, i);
        double v = Xsig_aug(2, i);
        double yaw = Xsig_aug(3, i);
        double yawd = Xsig_aug(4, i);

        double nu_a = Xsig_aug(5, i);

        double nu_yawdd = Xsig_aug(6, i);

        //predicted state values
        double px_p, py_p;

        //avoid division by zero
        if (fabs(yawd) > 0.001) {
            px_p = p_x + v / yawd * (sin(yaw + yawd * dt_) - sin(yaw));
            py_p = p_y + v / yawd * (cos(yaw) - cos(yaw + yawd * dt_));
        } else {
            px_p = p_x + v * dt_ * cos(yaw);
            py_p = p_y + v * dt_ * sin(yaw);
        }

        double v_p = v;
        double yaw_p = yaw + yawd*dt_;
        double yawd_p = yawd;

        //add noise
        px_p = px_p + 0.5 * nu_a * dt_ * dt_ * cos(yaw);
        py_p = py_p + 0.5 * nu_a * dt_ * dt_ * sin(yaw);
        v_p = v_p + nu_a*dt_;

        yaw_p = yaw_p + 0.5 * nu_yawdd * dt_*dt_;
        yawd_p = yawd_p + nu_yawdd*dt_;

        //write predicted sigma point into right column
        Xsig_pred(0, i) = px_p;
        Xsig_pred(1, i) = py_p;
        Xsig_pred(2, i) = v_p;
        Xsig_pred(3, i) = yaw_p;
        Xsig_pred(4, i) = yawd_p;
    }

    //print result
    std::cout << "Xsig_pred = " << std::endl << Xsig_pred << std::endl;

    //write result
    *Xsig_augX = Xsig_pred;

}

void UKF::PredictMeanAndCovariance(VectorXd* x_out, MatrixXd* P_out, MatrixXd* Xsig_predX) {

    std::cout << "UKF::PredictMeanAndCovariance" << std::endl;

    MatrixXd Xsig_pred = *Xsig_predX;

    //create vector for weights
    VectorXd weights = VectorXd(2 * n_aug_ + 1);

    //create vector for predicted state
    VectorXd x = VectorXd(n_x_);

    //create covariance matrix for prediction
    MatrixXd P = MatrixXd(n_x_, n_x_);

    // set weights
    double weight_0 = lambda_ / (lambda_ + n_aug_);
    weights(0) = weight_0;
    for (int i = 1; i < 2 * n_aug_ + 1; i++) { //2n+1 weights
        double weight = 0.5 / (n_aug_ + lambda_);
        weights(i) = weight;
    }

    //predicted state mean
    x.fill(0.0);
    for (int i = 0; i < 2 * n_aug_ + 1; i++) { //iterate over sigma points
        x = x + weights(i) * Xsig_pred.col(i);
    }

    //predicted state covariance matrix
    P.fill(0.0);
    for (int i = 0; i < 2 * n_aug_ + 1; i++) { //iterate over sigma points

        // state difference
        VectorXd x_diff = Xsig_pred.col(i) - x;
        //angle normalization
        while (x_diff(3) > M_PI) x_diff(3) -= 2. * M_PI;
        while (x_diff(3)<-M_PI) x_diff(3) += 2. * M_PI;

        P = P + weights(i) * x_diff * x_diff.transpose();
    }

    //print result
    std::cout << "Predicted state" << std::endl;
    std::cout << x << std::endl;
    std::cout << "Predicted covariance matrix" << std::endl;
    std::cout << P << std::endl;

    //write result
    *x_out = x;
    *P_out = P;
}


