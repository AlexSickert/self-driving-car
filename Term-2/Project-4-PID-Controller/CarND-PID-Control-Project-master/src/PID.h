#ifndef PID_H
#define PID_H

#include <string>

class PID {
public:
    /*
     * Errors
     */
    double p_error;
    double i_error;
    double d_error;
    double t_error;
    double prev_cte;
    double diff_cte;
    double current_cte;
    long step_counter;
    double t_mse;
    int twiddle_count; // counts form 0 to 2 for the 3 parameters
    double t_mse_sum; // needed to calculate mse
    double best_error;
    double p[3];
    double dp[3];
    long total_count;

    int t_s_a; // twiddle second loop
    int twiddle_state;

    /*
     * Coefficients
     */
    double Kp;
    double Ki;
    double Kd;

    /*
     * Constructor
     */
    PID();

    /*
     * Destructor.
     */
    virtual ~PID();

    /*
     * Initialize PID.
     */
    void Init(double Kp, double Ki, double Kd);

    /*
     * Update the PID error variables given cross track error.
     */
    void UpdateError(double cte);

    /*
     * Calculate the total PID error.
     */
    double TotalError();

    /*
     * Calculate the new steering angle
     */
    double GetSteering();

    /*
     * Calculate the new speed
     */
    double GetThrottle(double steer, double speed);

    void Twiddle(long i);

private:
    static const std::string txt_param[];

};

#endif /* PID_H */
