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

/* variable needed to prevent that after a speed increase we have immediately
 * a break action
 */
double last_speed_action = 1;

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
        
        // this part of the if statement is needed when the car starts
        // created two way points so that we can use that as a beginning

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
    
    /*
     * we have now the beginning of the path but we also need the end of the 
     * path. this is defined by a point further ahead which is a distance and 
     * a lane. And as we want to go in parallel to the street we create 
     * two points at the end of out path so that the curve ends parallel to
     * the street
     */

//    vector<double> next_wp0 = getXY(car_s + 30, (2 + 4 * lane), map_waypoints_s, map_waypoints_x,map_waypoints_y);
    vector<double> next_wp1 = getXY(car_s + 60, (2 + 4 * lane), map_waypoints_s, map_waypoints_x,map_waypoints_y);
    vector<double> next_wp2 = getXY(car_s + 90, (2 + 4 * lane), map_waypoints_s, map_waypoints_x,map_waypoints_y);

//    ptsx.push_back(next_wp0[0]);
    ptsx.push_back(next_wp1[0]);
    ptsx.push_back(next_wp2[0]);

//    ptsy.push_back(next_wp0[1]);
    ptsy.push_back(next_wp1[1]);
    ptsy.push_back(next_wp2[1]);
    
    // by now the array has 4 way points - two at the beginning and two at the end
    // we need to transform them into a coordinate system that starts at 0,0 
    // with an yaw of zero. to do that we make a transforamtion

    for(int i = 0; i < ptsx.size(); i++){
        
        double shift_x = ptsx[i] - ref_x;
        double shift_y = ptsy[i] - ref_y;        

        ptsx[i] = (shift_x * cos(0 - ref_yaw) - shift_y * sin(0 - ref_yaw)); 
        ptsy[i] = (shift_x * sin(0 - ref_yaw) + shift_y * cos(0 - ref_yaw));
    }

    tk::spline s;


    // her we call the slpline function that draws basically a smooth curve
    // through all the points we hand over to this function
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

    /*
     * we need more than 4 points on the curve. in fact we need a point every 20ms
     * and by that we define the speed of the car. the following lines perform 
     * this calculation. The resulting points  need to be brought back to the 
     * coordinate system of the map. this is also done here. 
     */
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
//
//void make_wp_simple2(double car_s, vector<double> &next_x_vals, vector<double> &next_y_vals, vector<double> map_waypoints_s, vector<double> map_waypoints_x, vector<double> map_waypoints_y, double speed, double next_d){
//   
//
//    double dist_inc;
//    
//    dist_inc = 0.0089 * speed; // converting miles per hour to way point chunks
//    
//    for(int i = 0; i < 5; i++)
//    {
//
//        double next_s = car_s + (i + 1) * dist_inc;
//        //double next_d = 6;
//        vector<double> xy = getXY(next_s, next_d, map_waypoints_s, map_waypoints_x, map_waypoints_y);
//
//        next_x_vals.push_back(xy[0]);
//        next_y_vals.push_back(xy[1]);
//
//    }
//    
//    
////    getFrenet(double x, double y, double theta, vector<double> maps_x, vector<double> maps_y)
//
//}
//
//
//void make_wp_smooth(vector<double> previous_path_x, vector<double> previous_path_y,double car_s, vector<double> &next_x_vals, vector<double> &next_y_vals, vector<double> map_waypoints_s, vector<double> map_waypoints_x, vector<double> map_waypoints_y, double speed, double next_d, double car_x, double car_y, double car_yaw){
//   
//    
//    double dist_inc;
//    
//    speed = 5;
//    
//    dist_inc = 0.0089 * speed;
//    
//    std::cout <<  " car_x = " << car_x << " car_y " << car_y << " yaw = " << car_yaw << std::endl;
//            
//            
//    vector<double> ptsx;
//    vector<double> ptsy;    
//    
//    double ref_yaw = deg2rad(car_yaw);  // ?????????
//    
//    double prev_car_x = car_x - cos(car_yaw);
//    double prev_car_y = car_y - sin(car_yaw);
//    
//    //double prev_car_x = -1;
//    //double prev_car_y = 0;
//    
//    ptsx.push_back(prev_car_x);
//    ptsx.push_back(car_x);
//
//    ptsy.push_back(prev_car_y);
//    ptsy.push_back(car_y);   
//    
//    vector<double> next_wp1 = getXY(car_s + 50, next_d, map_waypoints_s, map_waypoints_x,map_waypoints_y);
//    
//    ptsx.push_back(next_wp1[0]);
//    ptsy.push_back(next_wp1[1]);
//    
//    for(int i = 0; i < ptsx.size(); i++){
//        
//        double shift_x = ptsx[i] - car_x;
//        double shift_y = ptsy[i] - car_y;        
//
//        ptsx[i] = (shift_x * cos(0 - ref_yaw) - shift_y * sin(0 - ref_yaw)); 
//        ptsy[i] = (shift_x * sin(0 - ref_yaw) + shift_y * cos(0 - ref_yaw));
//        
//        std::cout << i << " ptsx[i] = " << ptsx[i] << " ptsy[i] " << ptsy[i] << std::endl;
//    }
//    
//    std::cout << "-----" << std::endl;
//
//    tk::spline s;
//    
//    s.set_points(ptsx, ptsy);  
//    
//    for (int i = 0; i <= 50; i++) {
//        
//        double x_point = (i + 1) * dist_inc;
//        double y_point = s(x_point);
//        
//        
//        //x_point = (car_x * cos(ref_yaw) - car_y * sin(ref_yaw));
//        //y_point = (car_x * sin(ref_yaw) - car_y * cos(ref_yaw));
//
//        x_point += car_x;
//        y_point += car_y;
//
//        next_x_vals.push_back(x_point);
//        next_y_vals.push_back(y_point);       
//        
//        std::cout << i << " x_point = " << x_point << " y_point " << y_point << std::endl;
//        
//    }
//    
//}

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

/*
 * convert a lane number to the meters from center of the way points
 */
int lane_to_meter(double i){   
    
    if( i == 0 ){ return 2;} 
    if( i == 1 ){ return 6;} 
    if( i == 2 ){ return 10;} 
    
}

/*
 * calculate the advantage of speed a potential lane change will give. Because
 * it makes no sense to change into a lane where a car is ahead that is even
 * slower than the car we are currently behind
 */

std::vector<double> speed_advantage(double dictances_ahead[], double speed_ahead[], int current_lane){
    
    double ref_speed = speed_ahead[current_lane];
    std::vector<double> ret(3);
    
    for(int i = 0; i < 3; i++){
        
        if(dictances_ahead[i] > 300){
            // assume good speed advantage
            ret[i] = 100;
        }else{
            ret[i] = speed_ahead[i] - ref_speed;
        }
    }
    return ret;      
}

/* Check if a lane switch is possible. For that we check if there is enough
 * space ahead of the potential lane and if there is also enough space behind
 * the car because it could be that an approaching car is crushing into our car
 *  
 */
double check_lane_switch_dynamic(int  car_lane, double max_collision_risk_ahead[], double max_collision_risk_behind[], std::vector<double> speed_advantage[]){
    
    /*     
     * we want to solve two questions: 
     * 1. is there enough space to change the lane
     * 2. is is beneficial to change the lane or is the other lane not faster?  
     */
    
    
    int current_lane = car_lane;
    
    int new_d = car_lane;
    
   
    
    std::cout << "collision risk lane 0 ahead " <<  max_collision_risk_ahead[0] << " behind " << max_collision_risk_behind[0]  << std::endl;
    std::cout << "collision risk lane 1 ahead " <<  max_collision_risk_ahead[1] << " behind " << max_collision_risk_behind[1]  << std::endl;
    std::cout << "collision risk lane 2 ahead " <<  max_collision_risk_ahead[2] << " behind " << max_collision_risk_behind[2]  << std::endl;
    
    
    
    if(current_lane == 1){  
        if((max_collision_risk_ahead[2] < 0.01) && (max_collision_risk_behind[2] < 0.01)){ new_d = 10.0;}; 
        if((max_collision_risk_ahead[0] < 0.01) && (max_collision_risk_behind[0] < 0.01)){ new_d = 2.0;};         
        return identify_lane(new_d);
    } 

    if(current_lane == 0){  
        if((max_collision_risk_ahead[1]) < 0.01 && (max_collision_risk_behind[1] < 0.01)){ new_d = 6.0;};        
        return identify_lane(new_d);
    }     

    if(current_lane == 2){  
        if((max_collision_risk_ahead[1]) < 0.01 && (max_collision_risk_behind[1] < 0.01)){ new_d = 6.0;};        
        return identify_lane(new_d);
    }       
    
}

/**
 * Basically a helper function to check a lane switch. if there is not a lot
 * of space in the lane we want to move into but the car ahead is faster, then
 * it still makes sense. 
 */

double collision_risk_check(double car_ahead_position, double car_ahead_speed, double car_behind_position, double car_behind_speed, double break_acceleration){
    
  
    // ensure there is security space
    
    std::cout << "closest distance to car ahead " <<  car_ahead_position << std::endl;
    
    if(car_ahead_speed >= car_behind_speed){        
        //no risk
        std::cout << "collision_risk_check NO risk due to speed diff 1. " <<  car_ahead_speed << " 2. " << car_behind_speed << std::endl;
        return  -1.0;
    }    

    // if car ahead is too close then we have a problem
    if(car_ahead_position < 30){        
        std::cout << "collision_risk_check distance = " <<  car_ahead_position << std::endl;
        return 1.0;        
    } 

    
    return  -1.0;
        
   /// speed_diff = speed_diff * 0.447; // convert to meters per second
    
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

/**
 * -----------------------------------------------------------------------------
 * @return 
 */

std::vector<double>  state_machine(double closest_car_ahead[], double closest_car_ahead_speed[], double closest_car_behind[], double closest_car_behind_speed[], double  car_lane, double car_s, double car_speed){
    
    std::vector<double> ret(2);
    double max_speed = 48;  
    int current_lane = car_lane;
    double new_d;
    double new_car_speed;
    double max_collision_risk_ahead[3];
    double max_collision_risk_behind[3];
    
    std::vector<double> speed_adv(3);
    
    /* calculate who much speed gain we have if we switch into a certain lane
     */
    speed_adv = speed_advantage(closest_car_ahead, closest_car_ahead_speed, current_lane);
    
    /* evaluate the collision risk on all lanes. 
     */
    for(int i = 0; i < 3; i++){
        std::cout << "------- " <<  i << "------- " << std::endl;
        max_collision_risk_ahead[i] = collision_risk_check(closest_car_ahead[i], closest_car_ahead_speed[i], car_s, car_speed, -3);
        max_collision_risk_behind[i] = collision_risk_check(closest_car_behind[i], car_speed, car_s, closest_car_behind_speed[i],  -0.5);
    } 
    
    // DO WE HAVE A PROBLEM ?   
    if(max_collision_risk_ahead[current_lane] > 0){
        
        std::cout << "WE HAVE A PROBLEM and are on lane " <<  current_lane << std::endl;        
        std::cout << "collision risk ahead in line " <<  current_lane << std::endl;
        
        // can we change lane? this means if before and after the car ther eis 30 meter space
        //  speed_adv
        // check_lane_switch_dynamic(int  car_lane, double max_collision_risk_ahead[], double max_collision_risk_behind[], double speed_advantage[]){
        new_d = check_lane_switch_dynamic(car_lane, max_collision_risk_ahead, max_collision_risk_behind, &speed_adv);
        
        if(new_d != car_lane){
            
            std::cout << "ok, we change lane to lane " <<  identify_lane(new_d) << std::endl;
            std::cout << "speed advantage:  " <<  speed_adv[new_d] << std::endl;
            
            // ok we can change lane            
            // speed stays the same because otherwise acceleration too high if
            // we do two things at on time
            new_car_speed = car_speed;   
            
        }else{
            std::cout << "we cannot change lane although we would like to" <<   std::endl;
            // if we cannot change lane, then we need to break
            
            if(last_speed_action > 0){
                last_speed_action = 0;
                new_car_speed = car_speed - 0; 
            }else{
                last_speed_action = -3;
                new_car_speed = car_speed - 3; 
            }
            
                       
        }        
        
    }else{
        
        std::cout << "NO PROBLEM" <<   std::endl;        
        new_d = car_lane;
        
        // if no problem, then we can drive faster
        if(car_speed < max_speed){                                   

            if(car_speed < max_speed - 7){    


                if(last_speed_action < 0){
                    last_speed_action = 0;
                    new_car_speed = car_speed + 0; 
                }else{
                    last_speed_action = 4.5;
                    new_car_speed = car_speed + 4.5;
                } 
                
            }else{
                
                if(last_speed_action < 0){
                    last_speed_action = 0;
                    new_car_speed = car_speed + 0; 
                }else{
                    last_speed_action = 1;
                    new_car_speed = car_speed + 1;
                } 
            }

        }else{
            new_car_speed = max_speed;
        }        
    }   
    
    
    // speed, d 
    ret[0] = new_car_speed;
    ret[1] = new_d;
    return ret;
}
                


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
            
                std::cout << "=======================================================" <<  std::endl;
          
        	// Main car's localization Data
          	double car_x = j[1]["x"];
          	double car_y = j[1]["y"];
          	double car_s = j[1]["s"];
          	double car_d = j[1]["d"];
          	double car_yaw = j[1]["yaw"];
          	double car_speed = j[1]["speed"];
                

                // calculate on which lane we are                
                int current_lane = identify_lane(car_d); // not good, needs to be changes
                
                std::cout << "we are on lane " <<  current_lane << std::endl;
                std::cout << "our speed is " <<  car_speed << std::endl;

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
                

               
                double tmp_collision_risk;
                double closest_car_behind[3];
                double closest_car_ahead[3];
                double closest_car_behind_speed[3];
                double closest_car_ahead_speed[3];
               
                // initialize vairables
                for(int i = 0; i < 3; i++){
                    //max_collision_risk_ahead[i] = - 1;
                    //max_collision_risk_behind[i] = - 1;
                    closest_car_behind[i] = 1000;
                    closest_car_ahead[i] = 1000;
                }

                // from all cars filter out the ones that are closest
                // and save the distance and speed
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
                        // car is ahead
                        if(diff < closest_car_ahead[lane_check]){
                            closest_car_ahead[lane_check] = diff;
                            closest_car_ahead_speed[lane_check] = check_speed;
                        }
                    }else{
                        diff = diff * -1;  // to convert from negative sign
                        // car is behind
                        if(diff < closest_car_behind[lane_check]){
                            closest_car_behind[lane_check] = diff;
                            closest_car_behind_speed[lane_check] = check_speed;
                        }
                    }                    
                   
                }
                
                
                //std::cout << "collision risk ahead " << max_collision_risk_ahead[current_lane] <<  std::endl; 
                std::cout << "closest car ahead  L-0 " << closest_car_ahead[0] << " L-1 " << closest_car_ahead[1] << " L-2 " << closest_car_ahead[2] << std::endl;
                std::cout << "closest car behind L-0 " << closest_car_behind[0] << " L-1 " << closest_car_behind[1] << " L-2 " << closest_car_behind[2] << std::endl;
                
                
                // plug the parameters in a state machine and get action back
                double new_lane;
                double new_car_speed;
                vector<double> result;
                
                /*
                 * Here we put the data that we so far collected and filtered from sensor
                 * fusion and put them in a state machine. Ther result is a vector
                 * with two parameters - for speed and lane
                 */
                result = state_machine(closest_car_ahead, closest_car_ahead_speed, closest_car_behind, closest_car_behind_speed, current_lane, car_s, car_speed);
                
                new_car_speed = result[0];
                new_lane = result[1];
                
                std::cout << "new_lane " << new_lane << std::endl;
                std::cout << "new_car_speed " << new_car_speed << std::endl;
                           
                /*
                 * here we plug the desired speed and path into a function which
                 * generates the way points
                 */
                make_wp_very_smooth(previous_path_x,previous_path_y,  car_x,  car_y,  car_yaw, car_s, next_x_vals, next_y_vals, map_waypoints_s, map_waypoints_x, map_waypoints_y, new_car_speed, new_lane);
                    
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
















































































