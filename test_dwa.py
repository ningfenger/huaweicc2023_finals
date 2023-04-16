# -*-coding:utf-8-*-
# @Time       : 2023/4/14 16:38
# @Author     : Feng Rui
# @Site       : 
# @File       : test_dwa.py
# @Software   : PyCharm
# @Description:

import math
import numpy as np
import matplotlib.pyplot as plt

# Define robot parameters
MAX_SPEED = 5.0  # Maximum robot speed in m/s
MAX_TURN = 1.0  # Maximum robot turn rate in rad/s
MAX_ACCEL = 1.0  # Maximum robot acceleration in m/s^2
MAX_TURN_ACCEL = 2.0  # Maximum robot turning acceleration in rad/s^2
ROBOT_RADIUS = 0.5  # Robot radius in meters

# Define simulation parameters
TIMESTEP = 0.01  # Simulation timestep in seconds
SIM_TIME = 10.0  # Simulation time in seconds

# Define target path
target_path = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 0.0], [3.0, 1.0], [4.0, 0.0]])

# Define 360 degree laser range data
laser_data = np.zeros(360)

# Define robot state variables
robot_pos = np.array([0.0, 0.0])
robot_vel = np.array([0.0, 0.0])
robot_heading = 0.0
robot_angular_vel = 0.0
robot_path_error = 0.0

# Define simulation loop
for sim_time in np.arange(0, SIM_TIME, TIMESTEP):
    # Calculate robot's current heading
    robot_heading = math.atan2(robot_vel[1], robot_vel[0])

    # Calculate target point along the path
    target_point = target_path[np.argmin(np.sum((target_path - robot_pos)**2, axis=1))]

    # Calculate the heading error and wrap it to -pi to pi range
    target_heading = math.atan2(target_point[1] - robot_pos[1], target_point[0] - robot_pos[0])
    heading_error = target_heading - robot_heading
    if heading_error > np.pi:
        heading_error -= 2*np.pi
    elif heading_error < -np.pi:
        heading_error += 2*np.pi

    # Calculate path error and update path following state
    path_error = np.sqrt((target_point[1] - robot_pos[1])**2 + (target_point[0] - robot_pos[0])**2)
    if path_error < robot_path_error:
        robot_path_error = path_error
    else:
        robot_path_error += TIMESTEP*robot_vel[0]

    # Calculate acceleration and turning acceleration limits
    accel_limit = MAX_ACCEL
    turn_accel_limit = MAX_TURN_ACCEL

    # Calculate desired speed and turn rate using DWA
    for v in np.arange(0, MAX_SPEED, 0.1):
        for w in np.arange(-MAX_TURN, MAX_TURN, 0.1):
            # Simulate robot motion for 1 second
            sim_pos = robot_pos
            sim_vel = np.array([v*math.cos(robot_heading), v*math.sin(robot_heading)])
            sim_heading = robot_heading
            sim_angular_vel = w
            for i in range(100):
                # Calculate forward and angular acceleration
                forward_accel = min(accel_limit, v - np.sqrt(sim_vel[0]**2 + sim_vel[1]**2))
                angular_accel = min(turn_accel_limit, w - sim_angular_vel)

                # Calculate robot motion
                sim_vel += TIMESTEP*forward_accel*np.array([math.cos(sim_heading), math.sin(sim_heading)])
                sim_angular_vel += TIMESTEP*angular_accel
                sim_heading += TIMESTEP*sim_angular_vel
                sim_pos = sim_pos + TIMESTEP*sim_vel

                # Check if robot collided with an obstacle
                if np.min(laser_data[0:90]) < ROBOT_RADIUS or np.min(laser_data[270:360]) < ROBOT_RADIUS:
                    break

            # Calculate path error for simulated motion
            sim_path_error = np.sqrt((target_point[1] - sim_pos[1])**2 + (target_point[0] - sim_pos[0])**2)

            # Check if this motion is better than the previous one
            if sim_path_error < path_error:
                path_error = sim_path_error
                robot_vel = sim_vel
                robot_angular_vel = sim_angular_vel

    # Update robot pose
    robot_pos += TIMESTEP*robot_vel
    robot_heading += TIMESTEP*robot_angular_vel

    # Plot robot position and target path
    plt.plot(robot_pos[0], robot_pos[1], 'ro')
    plt.plot(target_path[:, 0], target_path[:, 1], 'b--')
    plt.xlim([-1, 5])
    plt.ylim([-1, 5])
    plt.show()