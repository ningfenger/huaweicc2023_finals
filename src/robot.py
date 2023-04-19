# coding=utf-8
import copy
from typing import Optional, List, Tuple

import numpy as np

from tools import *

'''
机器人类
'''


class Robot:
    # 状态常量 0 空闲, 1 购买途中, 2 等待购买, 3 出售途中, 4 等待出售, 5 避撞, 6阻止对方
    FREE_STATUS = 0
    MOVE_TO_BUY_STATUS = 1
    WAIT_TO_BUY_STATUS = 2
    MOVE_TO_SELL_STATUS = 3
    WAIT_TO_SELL_STATUS = 4
    AVOID_CLASH = 5
    BLOCK_OTRHER = 6

    MOV_TO_ATTACK = 0
    WAIT_TO_ATTACK = 1
    ATTACK = 2
    BCK_TO_ATTACK = 3
    def __init__(self, ID: int, loc: Tuple[int]):
        self.ID = ID
        self.loc = copy.deepcopy(loc)
        self.workbench_ID = -1  # 所处工作台ID -1代表没有
        self.item_type = 0  # 携带物品类型
        self.time_value = 0.0  # 时间价值系数
        self.clash_value = 0.0  # 碰撞价值系数
        self.palstance = 0.0  # 角速度
        self.speed = (0.0, 0.0)  # 线速度
        self.toward = 0.0  # 朝向
        self.status: int = 0  # 0 空闲, 1 购买途中, 2 等待购买, 3 出售途中, 4 等待出售
        self.move_status: int = 0  # 移动时的状态
        self.attack_status: int = 0  # 攻击时的状态
        self.target = -1  # 当前机器人的目标控制台 -1代表无目标
        self.__plan = (-1, -1)  # 设定买和卖的目标工作台
        self.target_workbench_list = []  # 可到达的工作台列表
        self.anoter_workbench_list = []  # 可到达的敌方工作台列表
        self.path = []
        self.block_model = False # 崽种模式
        self.free_frames = 0 # 空闲帧数
        # 关于检测机器人对眼死锁的成员变量
        self.pre_position = np.array(list(self.loc))
        self.pre_frame = -1  # 记录上次一帧内移动距离大于min_dis
        self.pre_toward = 0  # 记录上次一帧内移动距离大于min_dis的角度
        self.loc_np = np.array(list(self.loc))
        self.is_stuck = False  # True if the robot is stuck with wall
        self.last_status = self.FREE_STATUS  # 用于冲撞避免的恢复 如果是等待购买和等待出售直接设置为购买/出售途中，并重新导航
        self.deadlock_with = -1
        # 避让等待
        self.frame_wait = 0
        self.frame_backword = 0  # 后退帧数
        # 预估剩余时间
        self.frame_reman_buy = 0  # 预计多久能买任务
        self.frame_reman_sell = 0  # 预计多久能卖任务
        # 路径追踪的临时点
        self.temp_target = None
        self.temp_target_idx = None  # 记录临时点的下标，用于计算是否转弯点
        self.last_target = -1  # 记录上一个目标，解除死锁用
        self.anoter_robot = -1  # 记录和它冲突的机器人
        self.temp_idx = None
        self.buy_use = False
        self.sell_use = False

        # 雷达信息
        self.radar_info_dis = None
        self.radar_info_theta = None
        self.radar_info_x = None
        self.radar_info_y = None
        self.radar_info_obt = None
        self.radar_info_rival = None

        self.repath_wait = 0
        self.re_path_int = (-1, -1)
    def get_frame_reman(self):
        '''
        预计还需要多久可以完成当前任务, 与状态有关
        '''
        if self.status in [self.MOVE_TO_BUY_STATUS, self.WAIT_TO_BUY_STATUS]:
            return self.frame_reman_buy
        elif self.status in [self.MOVE_TO_SELL_STATUS, self.WAIT_TO_SELL_STATUS]:
            return self.frame_reman_sell
        else:
            return 3000

    def update_frame_reman(self):
        '''
        更新预估值, 与状态有关
        '''
        if self.status in [self.MOVE_TO_BUY_STATUS, self.WAIT_TO_BUY_STATUS]:
            self.frame_reman_buy -= 1
        elif self.status in [self.MOVE_TO_SELL_STATUS, self.WAIT_TO_SELL_STATUS]:
            self.frame_reman_sell -= 1

    def trans_toward(self, toward):
        if toward < 0:
            return 2 * np.pi + toward
        return toward

    def update_frame_pisition(self, frame):
        """
        更新pre_frame,pre_position,pre_toward
        """
        self.pre_frame = frame
        self.pre_position = np.array(list(self.loc))
        self.pre_toward = self.trans_toward(self.toward)

    def set_plan(self, buy_ID: int, sell_ID: int):
        '''
        设置机器人计划, 传入购买和出售工作台的ID
        '''
        self.__plan = buy_ID, sell_ID

    def set_path(self, path: List[Tuple[float, float]]):
        '''
        设置机器人移动路径
        :param path: 路径, float型的坐标列表
        :return:
        '''
        self.path = np.array(path)
        self.temp_target = None

    def get_buy(self) -> int:
        '''
        获取买的目标
        '''
        return self.__plan[0]

    def get_sell(self) -> int:
        '''
        获取卖的目标
        '''
        return self.__plan[1]

    def find_temp_tar(self):
        robot_pos = np.array(list(self.loc))
        dists = np.sqrt(np.sum((self.path - robot_pos) ** 2, axis=1))
        nearest_row = np.argmin(dists)
        row1 = min(nearest_row + 1, len(self.path) - 1)
        row2 = min(nearest_row + 2, len(self.path) - 1)
        if nearest_row == 0:

            rowb1 = min(nearest_row + 1, len(self.path))
            rowb2 = nearest_row
        else:
            rowb1 = nearest_row
            rowb2 = nearest_row - 1

        target1 = self.path[row1, :]
        target2 = self.path[row2, :]
        # targetb1 = self.path[rowb1, :]
        # targetb2 = self.path[rowb2, :]
        return target1, target2
        # return line_ray_intersection2(target1, target2, targetb1, targetb2)
        # return line_ray_intersection(target1, target2, self.loc, self.toward)

    def find_temp_tar_idx(self):
        robot_pos = np.array(list(self.loc))
        dists = np.sqrt(np.sum((self.path - robot_pos) ** 2, axis=1))
        nearest_row = np.argmin(dists)
        row1 = min(nearest_row + 1, len(self.path) - 1)
        return row1

    def find_temp_tar_idx_path_input(self, path):
        robot_pos = np.array(list(self.loc))
        dists = np.sqrt(np.sum((path - robot_pos) ** 2, axis=1))
        nearest_row = np.argmin(dists)

        row1 = min(nearest_row + 1, len(path) - 1)

        return row1

    def dist2path(self):
        path = np.array(self.path)
        robot_pos = np.array(self.loc)
        dists = np.sqrt(np.sum((path - robot_pos) ** 2, axis=1))
        nearest_row = np.argmin(dists)


        return nearest_row
    
    # 四个动作
    def forward(self, speed: float):
        '''
        设置前进速度，单位为米/秒。
        正数表示前进。 
        负数表示后退。
        '''
        print("forward", self.ID, speed)

    def rotate(self, palstance: float):
        '''
        设置旋转速度，单位为弧度/秒。
        负数表示顺时针旋转。
        正数表示逆时针旋转。
        '''
        print('rotate', self.ID, palstance)

    def buy(self):
        '''
        购买当前工作台的物品，以输入数据的身处工作台 ID 为准。
        所处工作台与目标工作台一致才出售
        '''
        if self.workbench_ID == self.target:
            print("buy", self.ID)
            return True
        return False

    def sell(self) -> bool:
        '''
        出售物品给当前工作台，以输入数据的身处工作台 ID 为准。
        所处工作台与目标工作台一致才出售
        '''
        if self.workbench_ID == self.target:
            print("sell", self.ID)
            return True
        return False

    def destroy(self):
        '''
        销毁物品。
        '''
        print("destroy", self.ID)

    def update(self, s: str):
        '''
        根据判题器的输入更新机器人状态, 记得更新status
        '''
        s = s.split()
        self.workbench_ID, self.item_type = map(int, s[:2])
        self.time_value, self.clash_value, self.palstance, speed_x, speed_y, self.toward, x, y = map(
            float, s[2:])
        self.speed = (speed_x, speed_y)
        self.loc = (x, y)
        self.loc_np = np.array(list(self.loc))

    def update_radar(self, s: str):
        '''
        根据判题器的输入更新机器人的雷达数据, 记得更新status
        '''

        s = s.split()
        float_list = [float(str_item) for str_item in s]
        self.radar_info_theta = np.arange(0, 2*math.pi, math.pi / 180) + self.toward
        self.radar_info_dis = np.array(float_list)
        self.radar_info_x = self.radar_info_dis * np.cos(self.radar_info_theta) + self.loc[0]
        self.radar_info_y = self.radar_info_dis * np.sin(self.radar_info_theta) + self.loc[1]
        mask = is_multiple_of_half(self.radar_info_x) & is_multiple_of_half(self.radar_info_y)

        # 取反
        self.radar_info_obt = np.logical_not(mask)


    def avoid_obt(self, t, target_loc, flag_avoid_rival):
        # 评估t秒时是否会碰撞



        thr_theta = math.pi * 0.4
        # 机器人自身指向目标点的向量
        vec_robot2target = np.array(target_loc) - np.array(self.loc)


        # 机器人自身指向目标点的角度
        theta_robot2target = np.arctan2(vec_robot2target[1], vec_robot2target[0])

        # 机器人自身的速度向量角度
        theta_velo = np.arctan2(self.speed[1], self.speed[0])

        # 机器人自身的速度向量角度 和 机器人自身指向目标点的向量的夹角 如果大于阈值，说明机器人不会碰撞
        if abs(theta_robot2target - theta_velo) > thr_theta:
            return False, 0

        # 评估t秒时是否会碰撞
        if self.item_type == 0:
            thr_dis = 0.451
        else:
            thr_dis = 0.531


        # t秒后相对自身的
        radar_x_offset = self.radar_info_x - t * self.speed[0] - self.loc[0]
        radar_y_offset = self.radar_info_y - t * self.speed[1] - self.loc[1]

        dis_offset = np.sqrt(radar_x_offset ** 2 + radar_y_offset ** 2)

        num_points = len(dis_offset)
        flag = False
        potential_theta_id = -1
        min_dis = 200
        for idx_point in range(num_points):
            dis = dis_offset[idx_point]
            if flag_avoid_rival:
                if dis < thr_dis and dis < min_dis and (self.radar_info_obt[idx_point] or self.radar_info_rival[idx_point]):
                    flag = True
                    potential_theta_id = idx_point
                    min_dis = dis

            else:
                if dis < thr_dis and self.radar_info_obt[idx_point] and dis < min_dis:
                    flag = True
                    potential_theta_id = idx_point
                    min_dis = dis

        if flag:
            # 有障碍物且距离小于阈值

            obt_theta = self.radar_info_theta[potential_theta_id]
            delta_theta = theta_velo - obt_theta
            if abs(delta_theta) > math.pi / 2:
                # 不是前方碰撞
                return False, 0

            # 左右两个候选方向
            avoid_left = obt_theta + math.pi / 2
            avoid_right = obt_theta - math.pi / 2

            # 角度的循环取模 计算夹角
            delta_theta_left = (avoid_left - theta_robot2target + math.pi) % (2 * math.pi) - math.pi
            delta_theta_right = (avoid_right - theta_robot2target + math.pi) % (2 * math.pi) - math.pi

            # 选择离目标点更近的方向
            if abs(delta_theta_right) < abs(delta_theta_left):
                return True, delta_theta_right
            else:
                return True, delta_theta_left

        else:
            # 无障碍物或距离大于阈值
            return False, 0

