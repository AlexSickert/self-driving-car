#include <fstream>
#include <math.h>
#include <uWS/uWS.h>
#include <chrono>
#include <iostream>
#include <thread>
#include <vector>
#include "Eigen-3.3/Eigen/Core"
#include "Eigen-3.3/Eigen/QR"
#include "json.hpp"
#include "spline.h"

using namespace std;

// for convenience
using json = nlohmann::json;

// For converting back and forth between radians and degrees.
constexpr double pi() { return M_PI; }
double deg2rad(double x) { return x * pi() / 180; }
double rad2deg(double x) { return x * 180 / pi(); }

// Checks if the SocketIO event has JSON data.
// If there is data the JSON object in string format will be returned,
// else the empty string "" will be returned.
string hasData(string s) {
  auto found_null = s.find("null");
  auto b1 = s.find_first_of("[");
  auto b2 = s.find_first_of("}");
  if (found_null != string::npos) {
    return "";
  } else if (b1 != string::npos && b2 != string::npos) {
    return s.substr(b1, b2 - b1 + 2);
  }
  return "";
}

double distance(double x1, double y1, double x2, double y2)
{
	return sqrt((x2-x1)*(x2-x1)+(y2-y1)*(y2-y1));
}
int ClosestWaypoint(double x, double y, vector<double> maps_x, vector<double> maps_y)
{

	double closestLen = 100000; //large number
	int closestWaypoint = 0;

	for(int i = 0; i < maps_x.size(); i++)
	{
		double map_x = maps_x[i];
		double map_y = maps_y[i];
		double dist = distance(x,y,map_x,map_y);
		if(dist < closestLen)
		{
			closestLen = dist;
			closestWaypoint = i;
		}

	}

	return closestWaypoint;

}

int NextWaypoint(double x, double y, double theta, vector<double> maps_x, vector<double> maps_y)
{

	int closestWaypoint = ClosestWaypoint(x,y,maps_x,maps_y);

	double map_x = maps_x[closestWaypoint];
	double map_y = maps_y[closestWaypoint];

	double heading = atan2( (map_y-y),(map_x-x) );

	double angle = abs(theta-heading);

	if(angle > pi()/4)
	{
		closestWaypoint++;
	}

	return closestWaypoint;

}

// Transform from Cartesian x,y coordinates to Frenet s,d coordinates
vector<double> getFrenet(double x, double y, double theta, vector<double> maps_x, vector<double> maps_y)
{
	int next_wp = NextWaypoint(x,y, theta, maps_x,maps_y);

	int prev_wp;
	prev_wp = next_wp-1;
	if(next_wp == 0)
	{
		prev_wp  = maps_x.size()-1;
	}

	double n_x = maps_x[next_wp]-maps_x[prev_wp];
	double n_y = maps_y[next_wp]-maps_y[prev_wp];
	double x_x = x - maps_x[prev_wp];
	double x_y = y - maps_y[prev_wp];

	// find the projection of x onto n
	double proj_norm = (x_x*n_x+x_y*n_y)/(n_x*n_x+n_y*n_y);
	double proj_x = proj_norm*n_x;
	double proj_y = proj_norm*n_y;

	double frenet_d = distance(x_x,x_y,proj_x,proj_y);

	//see if d value is positive or negative by comparing it to a center point

	double center_x = 1000-maps_x[prev_wp];
	double center_y = 2000-maps_y[prev_wp];
	double centerToPos = distance(center_x,center_y,x_x,x_y);
	double centerToRef = distance(center_x,center_y,proj_x,proj_y);

	if(centerToPos <= centerToRef)
	{
		frenet_d *= -1;
	}

	// calculate s value
	double frenet_s = 0;
	for(int i = 0; i < prev_wp; i++)
	{
		frenet_s += distance(maps_x[i],maps_y[i],maps_x[i+1],maps_y[i+1]);
	}

	frenet_s += distance(0,0,proj_x,proj_y);

	return {frenet_s,frenet_d};

}

// Transform from Frenet s,d coordinates to Cartesian x,y
vector<double> getXY(double s, double d, vector<double> maps_s, vector<double> maps_x, vector<double> maps_y)
{
	int prev_wp = -1;

	while(s > maps_s[prev_wp+1] && (prev_wp < (int)(maps_s.size()-1) ))
	{
		prev_wp++;
	}

	int wp2 = (prev_wp+1)%maps_x.size();

	double heading = atan2((maps_y[wp2]-maps_y[prev_wp]),(maps_x[wp2]-maps_x[prev_wp]));
	// the x,y,s along the segment
	double seg_s = (s-maps_s[prev_wp]);

	double seg_x = maps_x[prev_wp]+seg_s*cos(heading);
	double seg_y = maps_y[prev_wp]+seg_s*sin(heading);

	double perp_heading = heading-pi()/2;

	double x = seg_x + d*cos(perp_heading);
	double y = seg_y + d*sin(perp_heading);

	return {x,y};

}

/**
 * -----------------------------------------------------------------------------
 * @param car_s
 * @param next_x_vals
 * @param next_y_vals
 * @param map_waypoints_s
 * @param map_waypoints_x
 * @param map_waypoints_y
 * @param speed
 * @param lane
 */

void make_wp_simple(double car_s, vector<double> &next_x_vals, vector<double> &next_y_vals, vector<double> map_waypoints_s, vector<double> map_waypoints_x, vector<double> map_waypoints_y, double speed, double car_d, double next_d){
   

    double dist_inc;
    
    dist_inc = 0.0089 * speed; // converting miles per hour to way point chunks
    
    int steps = 50;
    double d_inc = car_d;
    
    double d_delta = (next_d - car_d)/ steps;
    
    for(int i = 0; i < steps; i++)
    {

        double next_s = car_s + (i + 1) * dist_inc;
        d_inc += d_delta;
        //double next_d = 6;
        vector<double> xy = getXY(next_s, d_inc, map_waypoints_s, map_waypoints_x, map_waypoints_y);

        next_x_vals.push_back(xy[0]);
        next_y_vals.push_back(xy[1]);

    }
    
    
    // just a test 
    
//    for(int i = 0; i < 1000; i++)
//    {
//        double next_s = car_s + (i + 1) * dist_inc;
//        vector<double> xy = getXY(next_s, d_inc, map_waypoints_s, map_waypoints_x, map_waypoints_y);
//        
//        std::cout <<  "TEST, " << xy[0] << " ,  " << xy[1]  << std::endl;
//        
//    }
    

}


void make_wp_very_smooth(vector<double> previous_path_x, vector<double> previous_path_y, double car_x, double car_y, double car_yaw, double car_s, vector<double> &next_x_vals, vector<double> &next_y_vals, vector<double> map_waypoints_s, vector<double> map_waypoints_x, vector<double> map_waypoints_y, double speed, int lane){
    
//    int lane = 1;
    double ref_val = speed;
    
    int prev_size = previous_path_x.size();

    vector<double> ptsx;
    vector<double> ptsy;
        
    double ref_x = car_x;
    double ref_y = car_y;
    double ref_yaw = deg2rad(car_yaw);

    if(prev_size < 2){

        double prev_car_x = car_x - cos(car_yaw);
        double prev_car_y = car_y - sin(car_yaw);

        ptsx.push_back(prev_car_x);
        ptsx.push_back(car_x);

        ptsy.push_back(prev_car_y);
        ptsy.push_back(car_y);   

    }else{
        ref_x = previous_path_x[prev_size - 1];
        ref_y = previous_path_y[prev_size - 1];

        double ref_x_prev = previous_path_x[prev_size - 2];
        double ref_y_prev = previous_path_y[prev_size - 2];

        ref_yaw = atan2(ref_y - ref_y_prev, ref_x - ref_x_prev);

        ptsx.push_back(ref_x_prev);
        ptsx.push_back(ref_x);

        ptsy.push_back(ref_y_prev);
        ptsy.push_back(ref_y); 

    }

//    vector<double> next_wp0 = getXY(car_s + 30, (2 + 4 * lane), map_waypoints_s, map_waypoints_x,map_waypoints_y);
    vector<double> next_wp1 = getXY(car_s + 60, (2 + 4 * lane), map_waypoints_s, map_waypoints_x,map_waypoints_y);
    vector<double> next_wp2 = getXY(car_s + 90, (2 + 4 * lane), map_waypoints_s, map_waypoints_x,map_waypoints_y);

//    ptsx.push_back(next_wp0[0]);
    ptsx.push_back(next_wp1[0]);
    ptsx.push_back(next_wp2[0]);

//    ptsy.push_back(next_wp0[1]);
    ptsy.push_back(next_wp1[1]);
    ptsy.push_back(next_wp2[1]);

    for(int i = 0; i < ptsx.size(); i++){
        
        double shift_x = ptsx[i] - ref_x;
        double shift_y = ptsy[i] - ref_y;        

        ptsx[i] = (shift_x * cos(0 - ref_yaw) - shift_y * sin(0 - ref_yaw)); 
        ptsy[i] = (shift_x * sin(0 - ref_yaw) + shift_y * cos(0 - ref_yaw));
    }

    tk::spline s;

//    for (int i = 0; i < ptsy.size(); i++) {
//        std::cout << "x " << i << " = " << ptsx[i] << std::endl;                    
////        std::cout << "y " << i << " = " << ptsy[i] << std::endl;
//    }

    s.set_points(ptsx, ptsy);  
    
    next_x_vals.clear();
    next_y_vals.clear();

    for (int i = 0; i < previous_path_x.size(); i++) {
        next_x_vals.push_back(previous_path_x[i]);
        next_y_vals.push_back(previous_path_y[i]);
    }


    double target_x = 30.0;
    double target_y = s(target_x);
    double target_dist = sqrt((target_x) * (target_x) + (target_y) * (target_y));

    double x_add_on = 0;

    for (int i = 1; i <= 50 - previous_path_x.size(); i++) {

        double N = (target_dist / (.02 * ref_val / 2.24));
        double x_point = x_add_on + (target_x) / N;
        double y_point = s(x_point);

        x_add_on = x_point;
                
        double x_ref = x_point;
        double y_ref = y_point;

        x_point = (x_ref * cos(ref_yaw) - y_ref * sin(ref_yaw));
        y_point = (x_ref * sin(ref_yaw) + y_ref * cos(ref_yaw));

        x_point += ref_x;
        y_point += ref_y;

        next_x_vals.push_back(x_point);
        next_y_vals.push_back(y_point);
    }
}

void make_wp_simple2(double car_s, vector<double> &next_x_vals, vector<double> &next_y_vals, vector<double> map_waypoints_s, vector<double> map_waypoints_x, vector<double> map_waypoints_y, double speed, double next_d){
   

    double dist_inc;
    
    dist_inc = 0.0089 * speed; // converting miles per hour to way point chunks
    
    for(int i = 0; i < 5; i++)
    {

        double next_s = car_s + (i + 1) * dist_inc;
        //double next_d = 6;
        vector<double> xy = getXY(next_s, next_d, map_waypoints_s, map_waypoints_x, map_waypoints_y);

        next_x_vals.push_back(xy[0]);
        next_y_vals.push_back(xy[1]);

    }
    
    
//    getFrenet(double x, double y, double theta, vector<double> maps_x, vector<double> maps_y)

}


void make_wp_smooth(vector<double> previous_path_x, vector<double> previous_path_y,double car_s, vector<double> &next_x_vals, vector<double> &next_y_vals, vector<double> map_waypoints_s, vector<double> map_waypoints_x, vector<double> map_waypoints_y, double speed, double next_d, double car_x, double car_y, double car_yaw){
   
    
    double dist_inc;
    
    speed = 5;
    
    dist_inc = 0.0089 * speed;
    
    std::cout <<  " car_x = " << car_x << " car_y " << car_y << " yaw = " << car_yaw << std::endl;
            
            
    vector<double> ptsx;
    vector<double> ptsy;    
    
    double ref_yaw = deg2rad(car_yaw);  // ?????????
    
    double prev_car_x = car_x - cos(car_yaw);
    double prev_car_y = car_y - sin(car_yaw);
    
    //double prev_car_x = -1;
    //double prev_car_y = 0;
    
    ptsx.push_back(prev_car_x);
    ptsx.push_back(car_x);

    ptsy.push_back(prev_car_y);
    ptsy.push_back(car_y);   
    
    vector<double> next_wp1 = getXY(car_s + 50, next_d, map_waypoints_s, map_waypoints_x,map_waypoints_y);
    
    ptsx.push_back(next_wp1[0]);
    ptsy.push_back(next_wp1[1]);
    
    for(int i = 0; i < ptsx.size(); i++){
        
        double shift_x = ptsx[i] - car_x;
        double shift_y = ptsy[i] - car_y;        

        ptsx[i] = (shift_x * cos(0 - ref_yaw) - shift_y * sin(0 - ref_yaw)); 
        ptsy[i] = (shift_x * sin(0 - ref_yaw) + shift_y * cos(0 - ref_yaw));
        
        std::cout << i << " ptsx[i] = " << ptsx[i] << " ptsy[i] " << ptsy[i] << std::endl;
    }
    
    std::cout << "-----" << std::endl;

    tk::spline s;
    
    s.set_points(ptsx, ptsy);  
    
    for (int i = 0; i <= 50; i++) {
        
        double x_point = (i + 1) * dist_inc;
        double y_point = s(x_point);
        
        
        //x_point = (car_x * cos(ref_yaw) - car_y * sin(ref_yaw));
        //y_point = (car_x * sin(ref_yaw) - car_y * cos(ref_yaw));

        x_point += car_x;
        y_point += car_y;

        next_x_vals.push_back(x_point);
        next_y_vals.push_back(y_point);       
        
        std::cout << i << " x_point = " << x_point << " y_point " << y_point << std::endl;
        
    }
    
}

/*
 * -----------------------------------------------------------------------------
 */

int identify_lane(double d){    
     if(d < 4){ return 0;}
     if( d > 4 && d < 8){ return 1;}
     if( d > 8 ){ return 2;}    
}

/*
 * -----------------------------------------------------------------------------
 */

int lane_to_meter(double i){   
    
    if( i == 0 ){ return 2;} 
    if( i == 1 ){ return 6;} 
    if( i == 2 ){ return 10;} 
    
}



/**
 * -----------------------------------------------------------------------------
// */
//double check_lane_switch(double car_d, bool lane_free[]){
//    
//    int current_lane = identify_lane(car_d);
//    
//    double new_d;
//    
//    if(current_lane == 1){  
//        if(lane_free[2]){ new_d = 10.0;}; 
//        if(lane_free[0]){ new_d = 2.0;};         
//        return new_d;
//    } 
//
//    if(current_lane == 0){  
//        if(lane_free[1]){ new_d = 6.0;};        
//        return new_d;
//    }     
//
//    if(current_lane == 2){  
//        if(lane_free[1]){ new_d = 6.0;};        
//        return new_d;
//    }       
//    
//}

double check_lane_switch_dynamic(double car_d, double max_collision_risk_ahead[], double max_collision_risk_behind[]){
    
    int current_lane = identify_lane(car_d);
    
    double new_d;
    
    if(current_lane == 1){  
        if(max_collision_risk_ahead[2] < 0.01 && max_collision_risk_behind[2] < 0.01){ new_d = 10.0;}; 
        if(max_collision_risk_ahead[0] < 0.01 && max_collision_risk_behind[0] < 0.01){ new_d = 2.0;};         
        return new_d;
    } 

    if(current_lane == 0){  
        if(max_collision_risk_ahead[1] < 0.01 && max_collision_risk_behind[1] < 0.01){ new_d = 6.0;};        
        return new_d;
    }     

    if(current_lane == 2){  
        if(max_collision_risk_ahead[1] < 0.01 && max_collision_risk_behind[1] < 0.01){ new_d = 6.0;};        
        return new_d;
    }       
    
}

/**
 * calcualte a collsion risk in seconds to collision if we consider a break defined
 * by the acceleration factor 
 */

double collision_risk_xxx(double car_ahead_position, double car_ahead_speed, double car_behind_position, double car_behind_speed, double break_acceleration){
    
    double ret;
    double diff;
    
    // ensure there is security space
    if(car_ahead_position > car_behind_position){
        diff = car_ahead_position - car_behind_position;
        if(diff < 30){
            return 1.0;
        }
    } 
    
    if(car_ahead_speed >= car_behind_speed){        
        //no risk
        return  -1.0;
    }
    
    
//    
//    double time_delta = 0.2;
//    
//    if(car_ahead_speed >= car_behind_speed){
//        // no risk
//        return  -1.0;
//    }else{
//        double delta_speed = car_behind_speed - car_ahead_speed;
//        double distance = (car_ahead_position + (time_delta * car_ahead_speed))- (car_behind_position + (time_delta * car_behind_speed));
//        
//        if(distance < 30){
//            return 1.0;
//        }
//        
//        
//        double p_1 = sqrt(2 * distance / break_acceleration);
//        double p_2 = delta_speed / break_acceleration;
//        
//        double t = p_1 - p_2;
//        
//        // this means if it takes 10 sec to collision then 
//        // the return value would be 0.1
//        return  1 / t;
//        
//    }
    
   
   
    
}
//
//double collision_risk_xxx(double car_ahead_position, double car_ahead_speed, double car_behind_position, double car_behind_speed, double break_acceleration){
//    
//     double ret;
//    
//    if(car_ahead_speed >= car_behind_speed){
//        // no risk
//        ret =  -1.0;
//    }else{
//        double delta_speed = car_behind_speed - car_ahead_speed;
//        double distance = car_ahead_position - car_behind_position;
//        
//        double p_1 = sqrt(2 * distance / break_acceleration);
//        double p_2 = delta_speed / break_acceleration;
//        
//        double t = p_1 - p_2;
//        
//        // this means if it takes 10 sec to collision then 
//        // the return value would be 0.1
//        ret =  1 / t;
//        
//    }
//    
//    
//    return 222.333;
//    
//}
//    

//
//bool check_collision_risk(bool collision_risk, double check_car_s, double car_s, double check_speed, double car_speed, double meters_ahead){
//    
//    double speed_diff = check_speed - car_speed;
//    double position_diff;
//    
//    speed_diff = speed_diff * 0.447; // convert to meters per second
//    
//    if(speed_diff > 0){
//        
////        std::cout << "check_car_s =  " << check_car_s << " and car_s = " << car_s << std::endl;
//
//        // we check the distance between the cars
//        // do this only if the car is not already behind us
//        
//        
//        
//        if((check_car_s > car_s) && ((check_car_s - car_s) < meters_ahead)){
//            
//            position_diff = check_car_s - car_s; 
//            
//            // car too close
//            std::cout << "collision ahead !!!  "  << std::endl;
//            return true;
//        }
//    
//        
//    }else{
//        // in this case there is no collision risk and we return the given value
//        // the given value can be "true" due to checks for other cars
//        return collision_risk;
//    }
//    
//    
//   
//    
//}


/**
 * -----------------------------------------------------------------------------
 * @return 
 */

int main() {
  uWS::Hub h;

  // Load up map values for waypoint's x,y,s and d normalized normal vectors
  vector<double> map_waypoints_x;
  vector<double> map_waypoints_y;
  vector<double> map_waypoints_s;
  vector<double> map_waypoints_dx;
  vector<double> map_waypoints_dy;

  // Waypoint map to read from
  string map_file_ = "../data/highway_map.csv";
  // The max s value before wrapping around the track back to 0
  double max_s = 6945.554;

  ifstream in_map_(map_file_.c_str(), ifstream::in);

  string line;
  while (getline(in_map_, line)) {
  	istringstream iss(line);
  	double x;
  	double y;
  	float s;
  	float d_x;
  	float d_y;
  	iss >> x;
  	iss >> y;
  	iss >> s;
  	iss >> d_x;
  	iss >> d_y;
  	map_waypoints_x.push_back(x);
  	map_waypoints_y.push_back(y);
  	map_waypoints_s.push_back(s);
  	map_waypoints_dx.push_back(d_x);
  	map_waypoints_dy.push_back(d_y);
  }

  h.onMessage([&map_waypoints_x,&map_waypoints_y,&map_waypoints_s,&map_waypoints_dx,&map_waypoints_dy](uWS::WebSocket<uWS::SERVER> ws, char *data, size_t length,
                     uWS::OpCode opCode) {
    // "42" at the start of the message means there's a websocket message event.
    // The 4 signifies a websocket message
    // The 2 signifies a websocket event
    //auto sdata = string(data).substr(0, length);
    //cout << sdata << endl;
    if (length && length > 2 && data[0] == '4' && data[1] == '2') {

      auto s = hasData(data);

      if (s != "") {
        auto j = json::parse(s);
        
        string event = j[0].get<string>();
        
        if (event == "telemetry") {
          // j[1] is the data JSON object
            
                std::cout << "---------------------------" <<  std::endl;
          
        	// Main car's localization Data
          	double car_x = j[1]["x"];
          	double car_y = j[1]["y"];
          	double car_s = j[1]["s"];
          	double car_d = j[1]["d"];
          	double car_yaw = j[1]["yaw"];
          	double car_speed = j[1]["speed"];
                

                // calculate on which lane we are                
                int current_lane = identify_lane(car_d); // not good, needs to be changes

          	// Previous path data given to the Planner
          	auto previous_path_x = j[1]["previous_path_x"];
          	auto previous_path_y = j[1]["previous_path_y"];
          	// Previous path's end s and d values 
          	double end_path_s = j[1]["end_path_s"];
          	double end_path_d = j[1]["end_path_d"];

          	// Sensor Fusion Data, a list of all other cars on the same side of the road.
          	auto sensor_fusion = j[1]["sensor_fusion"];

          	json msgJson;

          	vector<double> next_x_vals;
          	vector<double> next_y_vals;
                
                bool collision_risk = false;
                bool move_to_middle_ok = true;
                

                double max_collision_risk_ahead[3];
                double max_collision_risk_behind[3];
                double tmp_collision_risk;
                double closest_car_behind[3];
                double closest_car_ahead[3];
               
                
                for(int i = 0; i < 3; i++){
                    max_collision_risk_ahead[i] = - 1;
                    max_collision_risk_behind[i] = - 1;
                    closest_car_behind[i] = -1000;
                    closest_car_ahead[i] = 1000;
                }


                for(int i = 0; i < sensor_fusion.size(); i++){
        
                    float d = (float) sensor_fusion[i][6];
                    

                    double vx = sensor_fusion[i][3];
                    double vy = sensor_fusion[i][4];
                    double check_car_s = sensor_fusion[i][5];
                    double check_speed = sqrt(vx * vx + vx * vx);
                    double meters_ahead = 50.0;
                    double acc = -3.0;
                    double diff;
                    int lane_check;
                    
                    diff = check_car_s - car_s;
                    lane_check = identify_lane(d);
                    
                    // identify the closest car behind and ahead
                    if(diff > 0){
                        if(diff < closest_car_ahead[lane_check]){
                            closest_car_ahead[lane_check] = diff;
                        }
                    }else{
                        if(diff > closest_car_behind[lane_check]){
                            closest_car_behind[lane_check] = diff;
                        }
                    }
                    
                    // checking the cars ahead
                     if(check_car_s > car_s ){
                            tmp_collision_risk = collision_risk_xxx(check_car_s, check_speed, car_s, car_speed, acc);

                            if(tmp_collision_risk > max_collision_risk_ahead[lane_check]){
                                max_collision_risk_ahead[lane_check] = tmp_collision_risk;
                            }    
                    }else{
                         // checking cars behind
                         tmp_collision_risk = collision_risk_xxx(car_s, car_speed, check_car_s, check_speed,  -1.0);   
                         
                        if(tmp_collision_risk > max_collision_risk_behind[lane_check]){
                                max_collision_risk_behind[lane_check] = tmp_collision_risk;
                        } 
                    }
                    
                    
                    
//                    if(identify_lane(d) == current_lane){            
//            
//                        if(check_car_s > car_s ){
//                            tmp_collision_risk = collision_risk_xxx(check_car_s, check_speed, car_s, car_speed, acc);
//
//                            if(tmp_collision_risk > max_collision_risk[current_lane]){
//                                max_collision_risk[current_lane] = tmp_collision_risk;
//                            }    
//                        }
//                      
//                    }else{
//                                   
//                        // check other lanes for lane switches
//                        int other_car_lane = identify_lane(d);  // not good, needs to be changes
//                        
//                        if(check_car_s > car_s ){
//                            
//                            // check the car ahead
//                            // we don't want to massively break, so we set acceleration lower
//                            tmp_collision_risk = collision_risk_xxx(check_car_s, check_speed, car_s, car_speed, -1.0);                           
//
//                            if(tmp_collision_risk > max_collision_risk[current_lane]){
//                                max_collision_risk[other_car_lane] = tmp_collision_risk;
//                            }                           
//                            
//                            
//                        }else{
//                            // car behind - we don't want to make the other car breaking 
//                            // also we do not want to accelerate                            
//                            tmp_collision_risk = collision_risk_xxx(car_s, car_speed, check_car_s, check_speed,  -1.0);
//                            
//                            if(tmp_collision_risk > max_collision_risk[current_lane]){
//                                max_collision_risk[other_car_lane] = tmp_collision_risk;
//                            } 
//                        }  
//                    } 
                }
                
                
                std::cout << "collision risk ahead " << max_collision_risk_ahead[current_lane] <<  std::endl; 
                std::cout << "closest car ahead  L0 " << closest_car_ahead[0] << " L1 " << closest_car_ahead[1] << " L2 " << closest_car_ahead[2] << std::endl;
                std::cout << "closest car behind L0 " << closest_car_behind[0] << " L1 " << closest_car_behind[1] << " L2 " << closest_car_behind[2] << std::endl;
                
                
                // check for middle lane policy
                
                 // check for milddle lane policy
                // identify large area if it is free
                // 0.07 means 14 seconds to collision
                if(max_collision_risk_ahead[1] < 0.05 && max_collision_risk_behind [1] < 0.05){                    
                    move_to_middle_ok = false;     
                }
                
                
                if(max_collision_risk_ahead[current_lane] != -1.0){
                   std::cout << "max_collision_risk on current lane is " << max_collision_risk_ahead[current_lane] << std::endl;                 
                }
                
                
                /*
                 * speed control section based on collision risk
                 * if the car is too slow, then accelerate
                 * if there is a risk of collision, then reduce speed
                 */
                
                double max_speed = 48;                
                // for state machine                
                double new_car_speed;
                
                if(max_collision_risk_ahead[current_lane] > 0){
                    if(car_speed > 0){
                        if(max_collision_risk_ahead[current_lane] > 0.1){
                            std::cout << "reducing speed because collision risk" << std::endl;
                            new_car_speed = car_speed - 3;
                        }
                    }                    
                }else{
                    if(car_speed < max_speed){                                   
                        
                        if(car_speed < max_speed - 7){    
//                            std::cout << "accelerating by 5 because slow" << std::endl;             
                            new_car_speed = car_speed + 4.5;
                        }else{
//                            std::cout << "accelerating by 1 because slow" << std::endl;  
                            new_car_speed = car_speed + 1;
                        }
                        
                    }else{
                        new_car_speed = max_speed;
                    }
                }
                

             
                double new_d;
                
                
                /*
                 * if there is collision risk then try to change lane if possible
                 * and if the middle lane is free, then try to go there
                 */
                if(max_collision_risk_ahead[current_lane] > 0.1){
                    new_d = check_lane_switch_dynamic(car_d, max_collision_risk_ahead, max_collision_risk_behind);
                }else{
                    // if we have no collision risk, then we could switch
                    // tot he middle lane if there is space
                    if(move_to_middle_ok){
                        if(identify_lane(car_d) != 1){
                            std::cout << "moving back to middle lane - middle lane policy" << std::endl;
                        }                        
                        new_d = 6.0;
                    }else{
                        new_d = lane_to_meter(identify_lane(car_d));
                    }
                    
                }
                
                
                
                if(identify_lane(car_d) != identify_lane(new_d)){
                   std::cout << "changing lane to new_d = " << new_d <<  " which is lane " << identify_lane(new_d) << std::endl; 
                }
        
//                std::cout << "car_d = " << car_d <<  " new_d =  " << new_d << std::endl; 
                
                // make the new way points  lane_to_meter(identify_lane(new_d))
                
                
//                make_wp_simple(car_s, next_x_vals, next_y_vals, map_waypoints_s,map_waypoints_x, map_waypoints_y, new_car_speed, car_d, new_d);
                
                make_wp_very_smooth(previous_path_x,previous_path_y,  car_x,  car_y,  car_yaw, car_s, next_x_vals, next_y_vals, map_waypoints_s, map_waypoints_x, map_waypoints_y, new_car_speed, identify_lane(new_d));
    
                
                //make_wp_smooth(previous_path_x, previous_path_y, car_s, next_x_vals, next_y_vals, map_waypoints_s,map_waypoints_x, map_waypoints_y, new_car_speed, new_d, car_x, car_y, car_yaw);
                 
                
          	msgJson["next_x"] = next_x_vals;
          	msgJson["next_y"] = next_y_vals;

          	auto msg = "42[\"control\","+ msgJson.dump()+"]";

          	//this_thread::sleep_for(chrono::milliseconds(1000));
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
















































































