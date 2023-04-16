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
import tools
import robot
import workmap
from controller import Controller
save_file_robot = "F:/huaweicc/src/temp_robot.pkl"
save_file_controller = "F:/huaweicc/src/temp_controller.pkl"

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


def detect_rival(robot):
    radar_x = robot.radar_info_x
    radar_y = robot.radar_info_y
    mask = tools.is_multiple_of_half(radar_y) & tools.is_multiple_of_half(radar_x)
    return mask

def show(idx_robot, map:workmap.Workmap, controller:Controller):
    robot = controller.robots[idx_robot]
    fig = plt.figure(figsize=(20, 16))
    plt.imshow(map.map_gray[::-1], origin='lower', extent=[0, 50, 0, 50])
    plt.plot(robot.path[:, 0], robot.path[:, 1])
    plt.plot(robot.loc[0], robot.loc[1], 'o')
    theta_radar = np.arange(0, 2 * math.pi, math.pi / 180) + robot.toward
    radar_x = robot.radar_info_dis * np.cos(theta_radar) + robot.loc[0]
    radar_y = robot.radar_info_dis * np.sin(theta_radar) + robot.loc[1]

    mask = detect_rival(robot)
    loc_rival, r_rival = controller.detect_rival(robot, robot.path[-1], 10)
    plt.plot(radar_x, radar_y)
    plt.plot(radar_x[mask], radar_y[mask], 'r.')
    plt.plot(robot.temp_target[0], robot.temp_target[1], 'b*')
    plt.plot(loc_rival[0], loc_rival[1], 'r*', markersize=15)

    controller.re_path(robot)
    target_select = controller.select_target(idx_robot)[0]

    plt.plot(target_select[0], target_select[1], 'g*')
    plt.show()
    pass


def show2(map:workmap.Workmap, controller:Controller):
    fig = plt.figure(figsize=(20, 16))
    plt.imshow(map.map_gray[::-1], origin='lower', extent=[0, 50, 0, 50])
    for robot in controller.robots:
        # robot = controller.robots[idx_robot]
        plt.plot(robot.path[:, 0], robot.path[:, 1])
        plt.plot(robot.loc[0], robot.loc[1], 'o')
        theta_radar = robot.radar_info_theta
        radar_x = robot.radar_info_x
        radar_y = robot.radar_info_y

        mask = detect_rival(robot)

        # 绘制雷达包围圈
        plt.plot(radar_x, radar_y)

        # 绘敌方机器人制候选点
        plt.plot(radar_x[mask], radar_y[mask], 'r.')

        # 绘制当前机器人临时目标点
        plt.plot(robot.temp_target[0], robot.temp_target[1], 'b*')

    for item_rival in controller.rival_list:
        plt.plot(item_rival[0][0], item_rival[0][1], 'r*', markersize=15)

    plt.show()
    pass

if __name__ == '__main__':
    idx_robot = 0
    import matplotlib.pyplot as plt
    map = workmap.Workmap(debug=True)
    map.read_map_directly('F:/huaweicc/maps/3.txt')
    map.draw_map()
    controller = load_controller()
    show2(map, controller)
    pass