# coding=utf-8
import sys
import time

from workmap import Workmap
from robot import Robot
from workbench import Workbench
from controller import Controller
from typing import Optional, List
import numpy as np
import os
from tools import *
try:
    os.chdir('./src')
except:
    pass


def finish():
    print('OK')
    sys.stdout.flush()


if __name__ == '__main__':
    # time.sleep(10)
    workmap = Workmap()
    robots: List[Robot] = []  # 机器人列表
    workbenchs: List[Workbench] = []  # 工作台列表
    blue_flag = True
    if input() == 'RED':
        blue_flag = False
    # 读入初始化地图
    workmap.read_map()
    robots = [None]*4
    for (i,j), idx in workmap.get_robots(blue_flag).items():
        loc = workmap.loc_int2float_normal(i,j)
        robots[idx] = Robot(idx, loc)
    # 我方工作台对象列表
    workbenchs = [None]*len(workmap.get_workbenchs(blue_flag))
    for (i,j), (wb_type, idx) in workmap.get_workbenchs(blue_flag).items():
        loc = workmap.loc_int2float_normal(i,j)
        workbenchs[idx] = Workbench(idx, wb_type, loc)
    # 敌方工作台对象列表
    rival_workbenchs = [None]*len(workmap.get_workbenchs(not blue_flag))
    for (i,j), (wb_type, idx) in workmap.get_workbenchs(not blue_flag).items():
        loc = workmap.loc_int2float_normal(i,j)
        rival_workbenchs[idx] = Workbench(idx, wb_type, loc)
    workmap.init_roads()
    workmap.draw_map()
    r2ws, r2ws_another = workmap.robot2workbench(blue_flag)
    # 可达的我方工作台
    for idx, r2w in enumerate(r2ws):
        robots[idx].target_workbench_list = r2w
    # 可达的敌方工作台
    for idx, r2w_another in enumerate(r2ws_another):
        robots[idx].anoter_workbench_list = r2w_another
    for idx, w2w in enumerate(workmap.workbench2workbench(blue_flag)):
        workbenchs[idx].target_workbench_list = w2w
    # 计算一下路径
    workmap.gen_paths()
    controller = Controller(robots, workbenchs, rival_workbenchs, workmap, blue_flag)
    # controller.init_workbench_other()
    # controller.attack_all()
    try:
        controller.init_workbench_other()
        controller.attack_all()
    except BaseException as e:
        sys.stderr.write(f'{e}\n')
    finish()

    while True:
        frame_id, money = map(int, input().split())
        input()  # 工作台数量
        for workbench in workbenchs:  # 更新工作台
            workbench.update(input())
        for robot in robots:  # 更新机器人
            robot.update(input())
        for robot in robots:
            robot.update_radar(input())  # 读入激光雷达输入
        OK_str = input()  # 读一个ok
        controller.control(frame_id, money)
        finish()
