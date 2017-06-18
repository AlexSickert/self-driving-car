# CarND-Controls-MPC
Self-Driving Car Engineer Nanodegree Program

---


## The Model
 
Mz approach to the project was to use the code example of the lectures and then fine tune the parameters. 
 
The files I worked on were main.cpp and MPC.cpp
 
# main.cpp
 
In line 88 we get the message form the simulator and subsequently extract from it the parameters. 
 
The simulator includes a latency of 100 milliseconds. To deal with this issue, we do not use the position parameters directly, but first make an estimate where the car would be in 100 ms and us this as an input for all subsequent calculations. This happens in line 117 - 122
 
The value we get from the simulator are relative to the map coordinates, but the input of the simulator and the creation of the green and yellow lines expect a coordinate system relative to the car. Therefore we need to transform the map coordinates to the car  coordinates, which happens in lines 150 - 177
 
In line 187 we use the waypoints that we received from the simulator and which e transformed to the car coordinate system and then fit a polynomial line into these way points. The polynomial i use has a dimension of 4. A value of 1 would be a straight line. I experimented with 2 and 3. 4 gave best results. At 5 and above the car was going off track quickly. 
 
In line 190 we calculate the cross track error. 
 
In line 196 we put the current state in a vector. As we transformed the values to the car coordinate system the values for x, y and psi are zero. 
 
The mentioned vector is then an input variable for the MPC solver. We then get from the solver a vector that contains steering angle and acceleration. 
 
In lines 211 to 270 we set the new throttle and convert the new steering angle to a value suitable for the simulator. In addition we create vectors of the polygon and the way points that are then used to draw the yellow and green lines. Finally a JSON object is constructed which  contains the new steering angle, throttle, yellow and green lines. The JSON object is then sent to the socket. 
 
# MPC.cpp
 
I used the solver as described in the lectures. To improve model performance I tuned the following parameters: 
 
Speed reference: Using my setup the maximum value is 50 mph. 
I realized that an emphasis on minimizing the value gap between sequential actuations makes the car driving smoother. So I experimented with various values. If the value is too small the car swings like a pendulum. If the value is too big, then the car gets off track in a curve. Ultimately i used a value of 2000. 
 
## Timestep Length and Elapsed Duration (N & dt)
 
My theory was that the more time steps we calculate into the future the better and the smaller the time elapsed the better. But during my tests it turned out this is not the case. I started with N = 25 and increased to 50 with no improvement of overall performance. I then reduced to 8 and experienced best performance. 
 
The initial value of dt was 0.05 and a reduction to 0.02 did not improve performance. I then increased it to 0.1. The overall system worked well with this. 
 
## Latency
The problem with latency is handled in main.cpp in line 109 to 122. We know that the latency is 100ms and therefore we calculate where the care probably would be in 100ms. 




