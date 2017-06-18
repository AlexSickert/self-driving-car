#include "PID.h"
#include <iostream>

using namespace std;

/*
 * TODO: Complete the PID class.
 */

PID::PID() {
    t_error = 0.0;
    prev_cte = 0.0;
}

PID::~PID() {
}

const string PID::txt_param[3] = {"P", "Integral", "Derivative"};

void PID::Init(double p_ini, double i, double d) {
    Kp = p_ini;
    Ki = i;
    Kd = d;
    step_counter = 0;
    t_error = 0;
    twiddle_count = 0;
    t_mse_sum = 0;
    best_error = 999.9;
    twiddle_state = 0;
    total_count = 0;

    p[0] = p_ini;
    p[1] = i;
    p[2] = d;

    dp[0] = 0.2;
    dp[1] = 0.002;
    dp[2] = 0.2;

    t_s_a = 1;
}

void PID::UpdateError(double cte) {
    total_count += 1; // used for speed control
    step_counter += 1;
    t_error = t_error + cte;
    t_mse_sum = t_mse_sum + cte * cte;
    t_mse = t_mse_sum / step_counter;
    current_cte = cte;
    diff_cte = cte - prev_cte;
    prev_cte = cte;
    long interval_max = 3;

    //    if (total_count > 1000){
    //        interval_max = 10 + total_count / 100;
    //    }

    // was 5 now 10 now 30 
    if (step_counter > interval_max) {
        Twiddle(step_counter);
        step_counter = 0;
        t_mse_sum = 0;
    }
}

double PID::TotalError() {
    return t_error;
}

double PID::GetSteering() {

    //    cout << "----------------------------------------" << endl;
    //    cout << "current_cte: " << current_cte << endl;
    //    cout << "diff_cte: " << diff_cte << endl;
    //    cout << "t_error: " << t_error << endl;
    //    cout << "t_mse: " << t_mse << endl;
    //    cout << "Kp: " << Kp << endl;
    //    cout << "Kd: " << Kd << endl;
    //    cout << "Ki: " << Ki << endl;


    //    double steer = -Kp * current_cte;
    //    double steer = -Kp * current_cte - Kd * diff_cte - Ki * t_error;
    double steer = -0.1 * current_cte - 4.0 * diff_cte - Ki * t_error;
    //    double steer = -Kp * current_cte - Kd * diff_cte;

    if (steer > 1.0) {
        steer = 1.0;
    }

    if (steer < -1.0) {
        steer = -1.0;
    }

    return steer;

}

void PID::Twiddle(long i) {

    int next_twiddle_state;

    cout << "------Twiddle()------------- " << endl;
    cout << "update frequency: " << i << endl;
    cout << "current_cte: " << current_cte << endl;
    cout << "diff_cte: " << diff_cte << endl;
    cout << "TOTAL ERROR t_error: " << t_error << endl;
    cout << "t_mse: " << t_mse << endl;
    cout << "Kp: " << Kp << endl;
    cout << "Kd: " << Kd << endl;
    cout << "Ki: " << Ki << endl;

    if (twiddle_count > 2) {
        twiddle_count = 0;
    }

    cout << "Twiddle: twiddle_count: " << twiddle_count << endl;
    cout << "Twiddle: optimizing parameter: " << txt_param[twiddle_count] << endl;
    cout << "Twiddle: t_s_a: " << t_s_a << endl;
    cout << "Twiddle: t_mse: " << t_mse << endl;
    cout << "Twiddle: best_error: " << best_error << endl;
    cout << "Twiddle: twiddle_state: " << twiddle_state << endl;



    /*
     *  this is the initial state and we only increase the value
     */
    if (twiddle_state == 0) {
        p[twiddle_count] += dp[twiddle_count];
        next_twiddle_state = 1;
    }

    /*
     * now we check if the error improved
     */
    if (twiddle_state == 1) {
        if (t_mse < best_error) {
            cout << "Twiddle: new best error found for twiddle_count: " << twiddle_count << endl;
            dp[twiddle_count] *= 1.1;
            /*
             * we optimized the parameter and can go to the next parameter
             */
            best_error = t_mse;
            twiddle_count += 1;
            next_twiddle_state = 0;
        } else {
            /*
             * the modification did not improve the MSE, so we reduce the parameter
             * and do another round of testing
             */
            cout << "Twiddle:new error is worse - inverting delta " << endl;
            p[twiddle_count] -= 2 * dp[twiddle_count];
            next_twiddle_state = 2;
        }

    }

    if (twiddle_state == 2) {
        /*
         * we reduced the parameter and check what the result is
         */
        if (t_mse < best_error) {
            cout << "Twiddle: new best error in state 2 for twiddle_count: " << twiddle_count << endl;

            dp[twiddle_count] *= 1.1;
            best_error = t_mse;
            /*
             * we are done with this parameter and can move to the next one
             */
            twiddle_count += 1;
            next_twiddle_state = 0;
        } else {
            cout << "Twiddle:new error is worse in state 2 - reducing delta " << endl;
            p[twiddle_count] += dp[twiddle_count];
            dp[twiddle_count] *= 0.9;
            /*
             * we are done with this parameter and can move to the next one
             */
            twiddle_count += 1;
            next_twiddle_state = 0;
        }

    }



    // set the next twiddle parameter to optimize
    Kp = p[0];
    Ki = p[1];
    Kd = p[2];

    twiddle_state = next_twiddle_state;
    //    cout << "----------------------------------------" << endl;
}

double PID::GetThrottle(double steer, double curr_speed) {

    //    double acceleration = 1;
    double acceleration = 1; // was 0.3 and 1 and 10  is too much
    double steer_square;
    double max_speed = 50.0;
    double min_speed = 8.0;

    steer_square = steer * steer;
    //    cout << "steer_square " << steer_square << " speed " << curr_speed << endl;

    // speed limit
    if (curr_speed > max_speed) {
        return 0.0;
    }


    /*
     * reduce speed if we have a steering angle     
     */

    if (steer_square > 0.2) { // was 0.1
        if (curr_speed > min_speed) {
            //            return -3;
            return -3 * steer_square; // was 5 now 3
        } else {
            return 0.1; // no acceleration in curves but prevent car stands still
        }
    }

    if (steer_square > 0.08) {
        return 0.0;
    }

    /*
     * at the beginning reduce acceleration to leave time for calibration
     */
    if (total_count < 500) { // was 10.000
        return acceleration = 0.1;
    }


    return acceleration;

}

