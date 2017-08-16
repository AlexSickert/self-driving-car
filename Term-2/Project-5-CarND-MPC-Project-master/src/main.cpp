#include <math.h>
#include <uWS/uWS.h>
#include <chrono>
#include <iostream>
#include <thread>
#include <vector>
#include "Eigen-3.3/Eigen/Core"
#include "Eigen-3.3/Eigen/QR"
#include "MPC.h"
#include "json.hpp"
#include <cppad/cppad.hpp>

// for convenience
using json = nlohmann::json;

// For converting back and forth between radians and degrees.

constexpr double pi() {
    return M_PI;
}

double deg2rad(double x) {
    return x * pi() / 180;
}

double rad2deg(double x) {
    return x * 180 / pi();
}

// Checks if the SocketIO event has JSON data.
// If there is data the JSON object in string format will be returned,
// else the empty string "" will be returned.

string hasData(string s) {
    auto found_null = s.find("null");
    auto b1 = s.find_first_of("[");
    auto b2 = s.rfind("}]");
    if (found_null != string::npos) {
        return "";
    } else if (b1 != string::npos && b2 != string::npos) {
        return s.substr(b1, b2 - b1 + 2);
    }
    return "";
}

// Evaluate a polynomial.
double polyeval(Eigen::VectorXd coeffs, double x) {
    double result = 0.0;
    for (int i = 0; i < coeffs.size(); i++) {
        result += coeffs[i] * pow(x, i);
    }
    return result;
}

// Fit a polynomial.
// Adapted from
// https://github.com/JuliaMath/Polynomials.jl/blob/master/src/Polynomials.jl#L676-L716

Eigen::VectorXd polyfit(Eigen::VectorXd xvals, Eigen::VectorXd yvals,
        int order) {
    assert(xvals.size() == yvals.size());
    assert(order >= 1 && order <= xvals.size() - 1);
    Eigen::MatrixXd A(xvals.size(), order + 1);

    for (int i = 0; i < xvals.size(); i++) {
        A(i, 0) = 1.0;
    }

    for (int j = 0; j < xvals.size(); j++) {
        for (int i = 0; i < order; i++) {
            A(j, i + 1) = A(j, i) * xvals(j);
        }
    }

    auto Q = A.householderQr();
    auto result = Q.solve(yvals);
    return result;
}


/*
 * For more detailed explanations please read the README file here:
 * https://github.com/AlexSickert/Udacity-SDC-T2-P5/blob/master/README.md 
 */

int main() {
    uWS::Hub h;

    // MPC is initialized here!
    MPC mpc;

    h.onMessage([&mpc](uWS::WebSocket<uWS::SERVER> ws, char *data, size_t length, uWS::OpCode opCode) {
        // "42" at the start of the message means there's a websocket message event.
        // The 4 signifies a websocket message
        // The 2 signifies a websocket event
        string sdata = string(data).substr(0, length);
//        cout << sdata << endl;
        if (sdata.size() > 2 && sdata[0] == '4' && sdata[1] == '2') {
            string s = hasData(sdata);
            if (s != "") {
                auto j = json::parse(s);
                        string event = j[0].get<string>();
                if (event == "telemetry") {
                    vector<double> ptsx = j[1]["ptsx"];
                    vector<double> ptsy = j[1]["ptsy"];
                    double px = j[1]["x"];
                    double py = j[1]["y"];
                    double psi = j[1]["psi"];
                    double v = j[1]["speed"];

//                    std::cout << "------------------------------" << std::endl;
                    
                    
                    /*
                     * For more detailed explanations please read the README file here:
                     * https://github.com/AlexSickert/Udacity-SDC-T2-P5/blob/master/README.md 
                     */
                    
                    /*
                     * we want to handle latency and for doing that we create a
                     * prediction where the car will be in 100ms as our latency 
                     * is 100 ms
                     */

                    double dt = 0.1; // tried 0.15 but no improvement
                    double Lf = 2.67;
                    
                    px = px + v * CppAD::cos(psi) * dt;
                    py = py + v * CppAD::sin(psi) * dt;
                    psi = psi + v / Lf * 0 * dt; // i left this constant
                    v = v + 0 * dt; // I left this constant
                    
                    
//                    
//                    std::cout << "ptsx:";
//
//                    for (int i = 0; i < ptsx.size(); ++i) {
//                        std::cout << ptsx[i] << ' ';
//                    }
//                    std::cout << std::endl;
//
//                    std::cout << "ptsy: ";
//                    for (int i = 0; i < ptsy.size(); ++i) {
//                        std::cout << ptsy[i] << ' ';
//                    }
//                    std::cout << std::endl;
//
//                    std::cout << "px: " << px << std::endl;
//                    std::cout << "py: " << py << std::endl;
//                    std::cout << "psi: " << psi << std::endl;
//                    std::cout << "v: " << v << std::endl;


                    /*
                     * TODO: Calculate steering angle and throttle using MPC.
                     *
                     * Both are in between [-1, 1].
                     *
                     */
                    double steer_value;
                    double throttle_value;
                    
                    
                    double new_x;
                    double new_y;
                    
                    Eigen::VectorXd local_ptsx;
                    Eigen::VectorXd local_ptsy;
                    
                    local_ptsx.resize(ptsx.size());
                    local_ptsy.resize(ptsx.size());
                    
                    
                    /*
                     * For more detailed explanations please read the README file here:
                     * https://github.com/AlexSickert/Udacity-SDC-T2-P5/blob/master/README.md 
                     */
                    
                    /*
                     * We convert the postitions from the coordinate system of
                     * the map to the coordinate system of the car which changes
                     * whenever the car is moving the car is always at coordinate
                     * x, y = 0 with a psi angle of 0
                     */
                    
                    for (size_t i = 0; i < ptsx.size(); i++) {
                        new_x = (ptsx[i] - px)*cos(-psi) - (ptsy[i] - py)*sin(-psi);
                        new_y = (ptsx[i] - px)*sin(-psi) + (ptsy[i] - py)*cos(-psi);  
                        local_ptsx[i] = new_x;
                        local_ptsy[i] = new_y;
                    }                 
                    
                    
                    
                    json msgJson;


//                    steer_value = 0;
                    throttle_value = 0.1;                    
                    
//                    auto coeffs = polyfit(local_ptsx,local_ptsy , 3);
//                    auto coeffs = polyfit(local_ptsx,local_ptsy , 3);
                    auto coeffs = polyfit(local_ptsx,local_ptsy , 4);
                    
                    //calculate cross track error at current position of the car
                    double cte = polyeval(coeffs, px) - py;
                    
                    double epsi = psi - atan(coeffs[1]);
                    
                    Eigen::VectorXd state(6);
//                    state << px, py, psi, v, cte, epsi;
                    state << 0, 0, 0, v, cte, epsi;


                    
                    auto vars = mpc.Solve(state, coeffs);
//                    
//                    std::cout << "vars: ";
//                    for (size_t i = 0; i < vars.size(); i++) {
//                        std::cout << i << std::endl;                      
//                        std::cout << vars[i] << std::endl;                       
//                    }
//                    
                    
                    double new_steering;
                    
                    new_steering = -vars[6] / deg2rad(25) ;
                    
//                    std::cout << "new_steering ";
//                    std::cout << new_steering<< std::endl;
                    msgJson["steering_angle"] = new_steering;
                    
                    
                    throttle_value = vars[7];
//                    std::cout << "throttle_value ";
//                    std::cout << throttle_value << std::endl;
                    msgJson["throttle"] = throttle_value;
//                    msgJson["throttle"] = 0.1;

                    //Display the MPC predicted trajectory 
                    vector<double> mpc_x_vals;
                    vector<double> mpc_y_vals;


                    
                    /*
                     * For more detailed explanations please read the README file here:
                     * https://github.com/AlexSickert/Udacity-SDC-T2-P5/blob/master/README.md 
                     */

                    /*
                     * drawing the green line which shows the polynomial
                     */
                    double xTmp;
                    
                    for (int i = 0; i < 20; ++i) {
                        xTmp = pow(i, 2);
                        mpc_x_vals.push_back(xTmp);
                        mpc_y_vals.push_back(polyeval(coeffs, xTmp));
                    }               
                    
                    // add the values for the green line to the JSON object                 
                    msgJson["mpc_x"] = mpc_x_vals;
                    msgJson["mpc_y"] = mpc_y_vals;

                    //Display the waypoints/reference line
                    vector<double> next_x_vals;
                    vector<double> next_y_vals;

                    //.. add (x,y) points to list here, points are in reference to the vehicle's coordinate system
                    // the points in the simulator are connected by a Yellow line


                    /*
                     * add the ideal path which we obtained from the simulator
                     * and then converted to the coordinate system of the car
                     * as a yellow line back to the simulator
                     */
                    for (size_t i = 0; i < ptsx.size(); i++) {
                        new_x = (ptsx[i] - px)*cos(-psi) - (ptsy[i] - py)*sin(-psi);
                        new_y = (ptsx[i] - px)*sin(-psi) + (ptsy[i] - py)*cos(-psi);            
                        next_x_vals.push_back( new_x );
                        next_y_vals.push_back(new_y);
                    }
                    
                    
                    msgJson["next_x"] = next_x_vals;
                    msgJson["next_y"] = next_y_vals;


                    auto msg = "42[\"steer\"," + msgJson.dump() + "]";
//                    std::cout << msg << std::endl;
                    // Latency
                    // The purpose is to mimic real driving conditions where
                    // the car does actuate the commands instantly.
                    //
                    // Feel free to play around with this value but should be to drive
                    // around the track with 100ms latency.
                    //
                    // NOTE: REMEMBER TO SET THIS TO 100 MILLISECONDS BEFORE
                    // SUBMITTING.
                    this_thread::sleep_for(chrono::milliseconds(100));
                    ws.send(msg.data(), msg.length(), uWS::OpCode::TEXT);
                }
            } else {
                // Manual driving
                std::string msg = "42[\"manual\",{}]";
                        ws.send(msg.data(), msg.length(), uWS::OpCode::TEXT);
            }
        }
    });

    // We don't need this since we're not using HTTP but if it's removed the
    // program
    // doesn't compile :-(
    h.onHttpRequest([](uWS::HttpResponse *res, uWS::HttpRequest req, char *data,
            size_t, size_t) {
        const std::string s = "<h1>Hello world!</h1>";
        if (req.getUrl().valueLength == 1) {
            res->end(s.data(), s.length());
        } else {
            // i guess this should be done more gracefully?
            res->end(nullptr, 0);
        }
    });

    h.onConnection([&h](uWS::WebSocket<uWS::SERVER> ws, uWS::HttpRequest req) {
        std::cout << "Connected!!!" << std::endl;
    });

    h.onDisconnection([&h](uWS::WebSocket<uWS::SERVER> ws, int code,
            char *message, size_t length) {
        ws.close();
        std::cout << "Disconnected" << std::endl;
    });

    int port = 4567;
    if (h.listen(port)) {
        std::cout << "Listening to port " << port << std::endl;
    } else {
        std::cerr << "Failed to listen to port" << std::endl;
        return -1;
    }
    h.run();
}
