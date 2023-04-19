# -*-coding:utf-8-*-
# @Time       : 2023/4/16 13:25
# @Author     : Feng Rui
# @Site       : 
# @File       : debug.py
# @Software   : PyCharm
# @Description:
import pickle
import math
import numpy as np

import robot
import workmap
from controller import Controller
save_file_robot = "F:/huaweicc/src/temp_robot.pkl"
save_file_controller = "/src/temp_controller_bck.pkl"

def save_robot(data):
    with open(save_file_robot, 'wb') as file:
        pickle.dump(data, file)

def save_controller(data):
    with open(save_file_controller, 'wb') as file:
        pickle.dump(data, file)


def load_robot():
    with open(save_file_robot, 'rb') as file:
        data = pickle.load(file)
    return data

def load_controller():
    with open(save_file_controller, 'rb') as file:
        data = pickle.load(file)
    return data

def show(robot:robot.Robot, map:workmap.Workmap, controller:Controller):
    fig = plt.figure(figsize=(20, 16))
    plt.imshow(map.map_gray[::-1], origin='lower', extent=[0, 50, 0, 50])
    plt.plot(robot.path[:, 0], robot.path[:, 1])
    plt.plot(robot.loc[0], robot.loc[1], 'o')
    theta_radar = np.arange(0, 2 * math.pi, math.pi / 180) + robot.toward
    radar_x = robot.radar_info * np.cos(theta_radar) + robot.loc[0]
    radar_y = robot.radar_info * np.sin(theta_radar) + robot.loc[1]

    plt.plot(radar_x, radar_y)

    plt.plot(robot.temp_target[0], robot.temp_target[1], 'b*')

    controller.re_path(robot)
    target_select = controller.select_target(0)[0]

    plt.plot(target_select[0], target_select[1], 'g*')
    plt.show()
    pass



if __name__ == '__main__':
    import matplotlib.pyplot as plt
    map = workmap.Workmap(debug=True)
    map.read_map_directly('F:/huaweicc/maps/2.txt')
    map.draw_map()
    controller = load_controller()
    robot = controller.robots[0]
    show(robot, map, controller)
    pass