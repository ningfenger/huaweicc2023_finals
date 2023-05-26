# coding=utf-8
import sys
import time
import subprocess
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
    # argv = sys.argv
    # if len(argv) <= 1:
    #     subprocess.run(["/usr/local/bin/pypy", "main.py", "restart"], cwd='./', shell=False, timeout=300)
    # elif len(argv) == 2 and argv[1] == 'restart':
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
    # 敌方机器人对象列表
    rival_robots = [None]*4
    for (i,j), idx in workmap.get_robots(not blue_flag).items():
        loc = workmap.loc_int2float_normal(i,j)
        rival_robots[idx] = Robot(idx, loc)
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
    
    rival_r2ws, _ = workmap.robot2workbench(not blue_flag)
    # 敌方机器人可达的工作台
    for idx, rival_r2w in enumerate(rival_r2ws):
        rival_robots[idx].target_workbench_list = rival_r2w
    # 敌方工作台可达的工作台
    for idx, rival_w2w in enumerate(workmap.workbench2workbench(not blue_flag)):
        rival_workbenchs[idx].target_workbench_list = rival_w2w
    # 计算一下路径
    workmap.gen_paths()
    controller = Controller(robots, rival_robots, workbenchs, rival_workbenchs, workmap, blue_flag)
    # controller.init_workbench_other()
    # controller.attack_all()
    try:
        controller.init_workbench_other()
        controller.attack_all()
        # sys.stderr.write(f"地图类型：{controller.map_type}\n")
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
