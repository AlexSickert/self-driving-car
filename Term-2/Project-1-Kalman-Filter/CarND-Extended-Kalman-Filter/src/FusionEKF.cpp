#include "FusionEKF.h"
#include "tools.h"
#include "Eigen/Dense"
#include <iostream>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

/*
 * Constructor.
 */
FusionEKF::FusionEKF() {
    is_initialized_ = false;

    previous_timestamp_ = 0;

    // initializing matrices
    R_laser_ = MatrixXd(2, 2);
    R_radar_ = MatrixXd(3, 3);
    H_laser_ = MatrixXd(2, 4);
    Hj_ = MatrixXd(3, 4);

    //measurement covariance matrix - laser
    R_laser_ << 0.0225, 0,
            0, 0.0225;

    //measurement covariance matrix - radar
    R_radar_ << 0.09, 0, 0,
            0, 0.0009, 0,
            0, 0, 0.09;

    H_laser_ << 1, 0, 0, 0,
            0, 1, 0, 0;

    /**
    TODO:
     * Finish initializing the FusionEKF.
     * Set the process and measurement noises
     */


    //    KalmanFilter::Init(VectorXd &x_in, MatrixXd &P_in, MatrixXd &F_in,
    //        MatrixXd &H_in, MatrixXd &R_in, MatrixXd &Q_in) {


}

/**
 * Destructor.
 */
FusionEKF::~FusionEKF() {
}

void FusionEKF::ProcessMeasurement(const MeasurementPackage &measurement_pack) {

    const float PI = 3.1415927;


    /*****************************************************************************
     *  Initialization
     ****************************************************************************/
    if (!is_initialized_) {
        cout << "NOT initialized" << endl;
        /**
        TODO:
         * Initialize the state ekf_.x_ with the first measurement.
         * Create the covariance matrix.
         * Remember: you'll need to convert radar from polar to cartesian coordinates.
         */
        // first measurement
        cout << "EKF: " << endl;
        ekf_.x_ = VectorXd(4);
        //        ekf_.x_ << 1, 1, 1, 1;

        ekf_.P_ = MatrixXd(4, 4);
        ekf_.P_ << 1, 0, 0, 0,
                0, 1, 0, 0,
                0, 0, 1000, 0,
                0, 0, 0, 1000;

        //        ekf_.R_ = MatrixXd(2, 2);
        //        ekf_.R_ << 0.0225, 0,
        //                0, 0.0225;

        ekf_.H_ = MatrixXd(2, 4);
        ekf_.H_ << 1, 0, 0, 0,
                0, 1, 0, 0;

        ekf_.F_ = MatrixXd(4, 4);
        ekf_.F_ << 1, 0, 1, 0,
                0, 1, 0, 1,
                0, 0, 1, 0,
                0, 0, 0, 1;


        previous_timestamp_ = measurement_pack.timestamp_;

        if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
            /**
            Convert radar from polar to cartesian coordinates and initialize state.
             */
            cout << "initialize with radar" << endl;
            cout << "first measurement timestamp: " << measurement_pack.timestamp_ << endl;
            cout << "first measurement ro: " << measurement_pack.raw_measurements_(0) << endl;
            cout << "first measurement phi: " << measurement_pack.raw_measurements_(1) << endl;
            cout << "first measurement ro_dot: " << measurement_pack.raw_measurements_(2) << endl;

            double rho = measurement_pack.raw_measurements_(0);
            double phi = measurement_pack.raw_measurements_(1);
            double rho_dot = measurement_pack.raw_measurements_(2);

            double px = rho * cos(phi);
            double py = rho * sin(phi);
            double vx = 0;
            double vy = 0;

            ekf_.x_ << px, py, vx, vy;

        } else if (measurement_pack.sensor_type_ == MeasurementPackage::LASER) {
            /**
            Initialize state.
             */

            //            x, y << measurement_pack.raw_measurements_;
            cout << "initialize with laser" << endl;
            cout << "first measurement timestamp: " << measurement_pack.timestamp_ << endl;
            cout << "first measurement x: " << measurement_pack.raw_measurements_[0] << endl;
            cout << "first measurement y: " << measurement_pack.raw_measurements_[1] << endl;

            float x = measurement_pack.raw_measurements_[0];
            float y = measurement_pack.raw_measurements_[1];

            if (x < 0.0000001) {
                x = 0.0000001;
            }

            if (y < 0.0000001) {
                y = 0.0000001;
            }

            ekf_.x_ << x, y, 0, 0;

        }

        cout << "initialization done" << endl;

        // done initializing, no need to predict or update
        is_initialized_ = true;
        return;
    }

    /*****************************************************************************
     *  Prediction
     ****************************************************************************/

    /**
     TODO:
     * Update the state transition matrix F according to the new elapsed time.
        - Time is measured in seconds.
     * Update the process noise covariance matrix.
     * Use noise_ax = 9 and noise_ay = 9 for your Q matrix.
     */

    float dt = (measurement_pack.timestamp_ - previous_timestamp_) / 1000000.0;
    previous_timestamp_ = measurement_pack.timestamp_;
    float dt_2 = dt * dt;
    float dt_3 = dt_2 * dt;
    float dt_4 = dt_3 * dt;

    ekf_.F_(0, 2) = dt;
    ekf_.F_(1, 3) = dt;

    float noise_ax = 9; // was 5
    float noise_ay = 9; // was 5

    ekf_.Q_ = MatrixXd(4, 4);
    ekf_.Q_ << dt_4 / 4 * noise_ax, 0, dt_3 / 2 * noise_ax, 0,
            0, dt_4 / 4 * noise_ay, 0, dt_3 / 2 * noise_ay,
            dt_3 / 2 * noise_ax, 0, dt_2*noise_ax, 0,
            0, dt_3 / 2 * noise_ay, 0, dt_2*noise_ay;

    ekf_.Predict();

    /*****************************************************************************
     *  Update
     ****************************************************************************/

    /**
     TODO:
     * Use the sensor type to perform the update step.
     * Update the state and covariance matrices.
     */

    if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
        // Radar updates
        ekf_.R_ = R_radar_;

        Hj_ << 0, 0, 0, 0,
                0, 0, 0, 0,
                0, 0, 0, 0;

        Hj_ = tools.CalculateJacobian(ekf_.x_);
        ekf_.H_ = Hj_;
//        ekf_.H_ = tools.CalculateJacobian(ekf_.x_);
        ekf_.UpdateEKF(measurement_pack.raw_measurements_);
    } else {
        // Laser updates
        ekf_.R_ = R_laser_;
        ekf_.H_ = H_laser_;
        ekf_.Update(measurement_pack.raw_measurements_);
    }

    // print the output
    cout << "x_ = " << ekf_.x_ << endl;
    cout << "P_ = " << ekf_.P_ << endl;
}
