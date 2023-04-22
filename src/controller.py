# coding=utf-8
import sys
import numpy as np
import random
from robot import Robot
from workbench import Workbench
from workmap import Workmap
from typing import Optional, List
import tools
import math
# import debug
import copy
from functools import cmp_to_key
from collections import deque

'''
控制类 决策，运动
'''


class Controller:
    # 总帧数
    TOTAL_FRAME = 50 * 60 * 4
    # 控制参数
    MOVE_SPEED_MUL = 4.1
    MOVE_SPEED = 50 * 0.5 / MOVE_SPEED_MUL  # 因为每个格子0.5, 所以直接用格子数乘以它就好了
    MAX_WAIT_MUL = 1.5
    MAX_WAIT = MAX_WAIT_MUL * 50  # 最大等待时间
    SELL_WEIGHT = 1.85  # 优先卖给格子被部分占用的
    SELL_DEBUFF = 0.55  # 非 7 卖给89的惩罚
    CONSERVATIVE = 1 + 2 / MOVE_SPEED  # 保守程度 最后时刻要不要操作
    STARVE_WEIGHT = SELL_WEIGHT

    FRAME_DIFF_TO_DETECT_DEADLOCK = 20  # 单位为帧,一个机器人 frame_now - pre_frame >这个值时开始检测死锁
    FRAME_DIFF = 10  # 单位为帧
    MIN_DIS_TO_DETECT_DEADLOCK = 0.15  # 如果机器人在一个时间段内移动的距离小于这个值,
    MIN_TOWARD_DIF_TO_DETECT_STUCK = np.pi / 30  # 并且角度转动小于这个值,需要进行检测

    # 判断两个机器人是否死锁的距离阈值,单位为米
    MIN_DIS_TO_DETECT_DEADLOCK_BETWEEN_N_N = 0.92  # 两个机器人都未装载物品
    MIN_DIS_TO_DETECT_DEADLOCK_BETWEEN_N_Y = 1.01  # 一个机器人装载物品,一个未装
    MIN_DIS_TO_DETECT_DEADLOCK_BETWEEN_Y_Y = 1.1  # 两个机器人都装载物品
    WILL_CLASH_DIS = 1.5  # 很可能要撞的距离
    WILL_HUQ_DIS = 2.5  # 可能冲突的距离

    # 避让等待的帧数
    AVOID_FRAME_WAIT = 20

    # 最长空闲帧数, 一直空闲切换为捣乱模式
    MAX_FREE = 10*50
    # 最长超时, 超时放弃此目标
    MAX_TIME_OUT = -10*50
    # 最长额外买卖， 如果太长直接崽人
    MAX_BUY_ZAI = 20*50/MOVE_SPEED  # 第一个参数是最长能接受的秒数

    # 最多不可达工作台帧数, 超时重置为可达
    MAX_CAN_NOT_REACH = 2*50

    # 地图类型
    MAP_TYPE_UNKNOW = 0  # 未知
    MAP_TYPE_BROAD = 1  # 相对开阔的地形，各自的资源点都比较丰富，双方运输路径存在交错。
    MAP_TYPE_3V1 = 2  # 相对开阔的地形，各自的资源点都比较丰富，地图被分割为两个不连通区域，其中一个区域为红色基地，只有红色工作台，机器人初始为3红1蓝，另一个区域为蓝色基地，只有蓝色工作台，机器人初始为3蓝1红。
    MAP_TYPE_NARROW = 3  # 相对狭窄的地形，双方有各自独立基地，基地之间存在多条路径连通，并且基地外部存在一些分散的红蓝工作台可使用。
    MAP_TYPE_4V4 = 4  # 极为开阔的地形，整张地图完全没有蓝色工作台，只有红色工作台，且红色工作台点比较丰富。

    # 攻击策略，优先攻击的工作台类型
    ATTACK_TYPE_ORDER = {
        1: 1, 2: 1, 3: 1,
        4: 4, 5: 4, 6: 4,
        7: 3, 8: 2, 9: 2,
    }

    def __init__(self, robots: List[Robot], rival_robots: List[Robot], workbenchs: List[Workbench], rival_workbenchs: List[Workbench], m_map: Workmap, blue_flag: bool):
        self.robots = robots
        # 敌方机器人
        self.rival_robots = rival_robots
        self.workbenchs = workbenchs
        # 敌方工作台
        self.rival_workbenchs = rival_workbenchs
        self.m_map = m_map
        self.m_map_arr = np.array(m_map.map_gray)
        self.blue_flag = blue_flag
        self.starve = {4: 0, 5: 0, 6: 0}  # 当7急需4 5 6 中的一个时, +1 鼓励生产
        # 预防跳帧使用, 接口先留着
        self.buy_list = []  # 执行过出售操作的机器人列表
        self.sell_list = []  # 执行过购买操作的机器人列表
        self.tmp_avoid = {}  # 暂存避让计算结果

        # 攻守
        self.other_workbenchs_order = deque()  # 按进攻优先级排序工作台
        self.other_workbenchs_list = []  # 已经进攻过的工作台
        self.can_not_reach_workbenchs = {}  # 记录无法到达的工作台即持续帧数
        self.rival_list = []
        # 开始派多少机器人去捣乱
        self.max_block_robots = 1 if self.blue_flag else 1
        # 记录工作台被拉黑了多少次
        self.black_workbenchs = {}
        # 地图类型
        self.map_type = self.MAP_TYPE_UNKNOW

    def set_control_parameters(self, move_speed: float, max_wait: int, sell_weight: float, sell_debuff: float):
        '''
        设置参数， 建议取值范围:
        move_speed: 3-5 估算移动时间
        max_wait: 1-5 最大等待时间
        sell_weight: 1-2 优先卖给格子被部分占用的
        sell_debuff: 0.5-1 将456卖给9的惩罚因子
        '''
        self.MOVE_SPEED = 50 * 0.5 / move_speed  # 估算移动时间
        self.MAX_WAIT = max_wait * 50  # 最大等待时间
        self.SELL_WEIGHT = sell_weight  # 优先卖给格子被部分占用的
        self.SELL_DEBUFF = sell_debuff  # 将456卖给9的惩罚因子

    def init_workbench_other(self):
        '''
        初始化敌方工作台信息
        '''
        # 敌方可达的工作台列表
        other_workbenchs_reach_buy, _ = self.m_map.robot2workbench(
            not self.blue_flag)
        other_workbenchs_reach_sell = self.m_map.workbench2workbench(
            not self.blue_flag)
        other_workbenchs_reach_set = set()
        for workbenchs in other_workbenchs_reach_buy:
            other_workbenchs_reach_set.update(workbenchs)
        for w_ID in other_workbenchs_reach_set.copy():
            other_workbenchs_reach_set.update(
                other_workbenchs_reach_sell[w_ID])
        other_workbenchs_score = [0]*9  # 按类型记录敌方得分 idx 类型 v 分数(攻击难度)
        other_workbenchs = [[]for _ in range(9)]  # 按类型记录敌方工作台 idx 类型 v 编号
        for (i, j), (w_type, w_ID) in self.m_map.get_workbenchs(not self.blue_flag).items():
            if w_ID not in other_workbenchs_reach_set:  # 敌方到不了的工作台, 不用管
                continue

            score = self.m_map_arr[i][j]
            # 9 可以让4-8+1
            if w_type == 9:
                for idx in range(4, 9):
                    other_workbenchs[idx].append(w_ID)
                    other_workbenchs_score[idx] += score
            else:
                other_workbenchs[w_type].append(w_ID)
                other_workbenchs_score[w_type] += score
        # 优先选数目最少的工作台 同数目选障碍物多的 同障碍物选类型编号小的

        def cmp(idx1, idx2):
            if len(other_workbenchs[idx1]) < len(other_workbenchs[idx2]):
                return -1
            elif len(other_workbenchs[idx1]) == len(other_workbenchs[idx2]):
                if other_workbenchs_score[idx1] < other_workbenchs_score[idx2]:
                    return -1
                elif other_workbenchs_score[idx1] == other_workbenchs_score[idx2] and idx1 < idx2:
                    return -1
            return 1
        # 按优先级对类型排序
        other_workbenchs_type_order = list(range(1, 9))
        other_workbenchs_type_order.sort(key=cmp_to_key(cmp))
        # 工作台的类型得分
        type_score = [0]*10
        for score, w_type in enumerate(other_workbenchs_type_order):
            # 排序越靠前得分越高
            type_score[w_type] = 10-score
            # 与可出售对象里面最高的高半级
            if w_type >= 4 and type_score[9] == 0:
                type_score[9] = 10-score+0.5
        arrive_score = [0]*len(self.rival_workbenchs)  # 按照敌方到达的顺序打分

        # 假装自己是敌方, 对每个工作台打分
        # 暂存一下己方数据, 防止影响后序使用
        tmp_robots = self.robots
        tmp_workbenchs = self.workbenchs
        self.blue_flag = not self.blue_flag
        # 设置敌方工作台123的状态为生产完成
        for rwb in self.rival_workbenchs:
            if rwb.typeID < 4:
                rwb.product_status = 1
        # 给敌方披上友军的衣服
        self.robots = self.rival_robots
        self.workbenchs = self.rival_workbenchs
        # 结束标志， 当敌方机器人都无事可做 模拟完成
        over_flag = False
        score = 1000
        while not over_flag:
            over_flag = True
            # 模拟决策
            for robot in self.robots:
                # 有一个机器人还有事可做
                if self.choise(1, robot):
                    over_flag = False
                    idx_workbench_to_buy = robot.get_buy()
                    idx_workbench_to_sell = robot.get_sell()
                    robot.target = idx_workbench_to_buy
                    self.re_path(robot)
                    # 预定工作台
                    self.workbenchs[idx_workbench_to_buy].pro_buy()
                    self.workbenchs[idx_workbench_to_sell].pro_sell(
                        self.workbenchs[idx_workbench_to_buy].typeID)
                else:
                    robot.set_plan(-1, -1)
            # 模拟已经完成上一步
            for robot in self.robots:
                idx_workbench_to_buy = robot.get_buy()
                idx_workbench_to_sell = robot.get_sell()
                if idx_workbench_to_buy == -1 or idx_workbench_to_sell == -1:
                    continue
                # 取消预售预购
                workbench_to_buy = self.workbenchs[idx_workbench_to_buy]
                workbench_to_sell = self.workbenchs[idx_workbench_to_sell]
                workbench_to_buy.pro_buy(False)
                workbench_to_sell.pro_sell(
                    self.workbenchs[idx_workbench_to_buy].typeID, False)
                # 模拟购入售出
                if arrive_score[idx_workbench_to_buy] == 0:
                    arrive_score[idx_workbench_to_buy] = score
                    score -= 1
                # 1 2 3 不用处理售出
                if workbench_to_buy.typeID > 3:
                    # 直接拉黑这个工作台，防止无法结束
                    self.can_not_reach_workbenchs[idx_workbench_to_buy] = 1
                # 8 9 号的处理
                if workbench_to_sell.typeID in [8, 9]:
                    if arrive_score[idx_workbench_to_sell] == 0:
                        arrive_score[idx_workbench_to_sell] = score
                        score -= 1
                    self.can_not_reach_workbenchs[idx_workbench_to_sell] = 1
                # 装格, 满了应该也不会有人将其作为目标了
                workbench_to_sell.material += (1 << workbench_to_buy.typeID)
                if workbench_to_sell.check_materials_full:
                    workbench_to_sell.product_status = 1
                    if arrive_score[idx_workbench_to_sell] == 0:
                        arrive_score[idx_workbench_to_sell] = score
                        score -= 1
                # 处理机器人坐标
                robot.loc = workbench_to_sell.loc
        # 恢复现场
        self.robots = tmp_robots
        self.workbenchs = tmp_workbenchs
        self.can_not_reach_workbenchs.clear()
        self.blue_flag = not self.blue_flag
        # 最终排序, 先按类型排序再按到达排序再按类型试试

        def cmp_final(idx1, idx2):
            type_ID1 = self.rival_workbenchs[idx1].typeID
            type_ID2 = self.rival_workbenchs[idx2].typeID
            # 优先看类型大类
            if self.ATTACK_TYPE_ORDER[type_ID1] > self.ATTACK_TYPE_ORDER[type_ID2]:
                return -1
            elif self.ATTACK_TYPE_ORDER[type_ID1] < self.ATTACK_TYPE_ORDER[type_ID2]:
                return 1
            # 之后看具体类型的评价
            if type_score[type_ID1] > type_score[type_ID2]:
                return -1
            elif type_score[type_ID1] < type_score[type_ID2]:
                return 1
            # 之后看访问到的顺序
            if arrive_score[idx1] > arrive_score[idx2]:
                return -1
            return 1
        other_workbenchs_list = set()
        for ows in other_workbenchs:
            for ow in ows:
                other_workbenchs_list.add(ow)
        self.other_workbenchs_order = deque(
            sorted(list(other_workbenchs_list), key=cmp_to_key(cmp_final)))
        # 将工作台队列拷贝到一个列表, 供巡逻使用
        self.other_workbenchs_list = list(self.other_workbenchs_order)

    def get_attack_path(self, robot: Robot, workbench_block, mast_run=False):
        '''
        规划一个阻碍工作台的路径
        '''
        # 手里已经有东西
        if robot.item_type > 0:
            block_path = self.m_map.get_path(
                robot.loc, workbench_block, not self.blue_flag, True)
            if block_path:
                robot.set_plan(workbench_block, -1)
                return True
            return False
        # 直接去崽人
        derect_block_path = self.m_map.get_path(
            robot.loc, workbench_block, not self.blue_flag, False)
        if not derect_block_path:
            return False
        # 先尝试买一个东西再去崽人
        min_path_length = None  # 记录最短路线
        if not mast_run:
            for workbench_buy in robot.target_workbench_list:
                if self.workbenchs[workbench_buy].typeID not in [1, 2, 3]:
                    continue
                block_path = self.m_map.get_path(
                    self.workbenchs[workbench_buy].loc, workbench_block, not self.blue_flag, True)
                if not block_path:
                    continue
                buy_path = self.m_map.get_path(
                    robot.loc, workbench_buy, self.blue_flag, False)
                if not min_path_length or len(buy_path) + len(block_path) < min_path_length:
                    min_path_length = len(buy_path) + len(block_path)
                    robot.set_plan(workbench_buy, workbench_block)
        if min_path_length and min_path_length - len(derect_block_path) < self.MAX_BUY_ZAI:
            return True
        else:
            robot.set_plan(workbench_block, -1)
            return True

    def attack_all(self):
        '''
        按照优先级选择进攻目标
        为了防止极端情况下直接报错退出, 建议调用此函数时直接try一下, 如果报错放弃崽人
        '''
        # 说明是地图4并且我方是红方
        if not self.other_workbenchs_order:
            self.map_type = self.MAP_TYPE_4V4
            return
        zz_robots: List[Robot] = []  # 无事可做机器人
        for robot in self.robots:
            for w_idx_buy in robot.target_workbench_list:
                # 可以完成一次完整的买卖, 能赚钱的机器人
                if self.workbenchs[w_idx_buy].target_workbench_list:
                    break
            else:
                zz_robots.append(robot)
        # 说明是地图2或者地图4, 全部进攻即可
        if zz_robots:
            if len(zz_robots) == 4:
                # 说明是地图4并且我方蓝方
                self.map_type = self.MAP_TYPE_4V4
            else:
                # 说明是地图2
                self.map_type = self.MAP_TYPE_3V1
            for _ in range(len(self.other_workbenchs_order)):
                workbench_block = self.other_workbenchs_order.popleft()
                for robot in zz_robots:
                    if robot.block_model:
                        continue
                    if self.get_attack_path(robot, workbench_block):
                        robot.block_model = True
                        break
                self.other_workbenchs_order.append(workbench_block)
                if all([robot.block_model for robot in zz_robots]):
                    break
        else:
            # 相对狭窄说明是地图3
            if self.m_map.block_num > Workmap.BLOCK_NUM_THRESHOLD:
                self.map_type = self.MAP_TYPE_NARROW
            else:
                self.map_type = self.MAP_TYPE_BROAD

            attack_num = 0
            for _ in range(len(self.other_workbenchs_order)):
                if attack_num >= self.max_block_robots:
                    break
                workbench_block = self.other_workbenchs_order.popleft()
                for robot in self.robots:
                    if robot.block_model:
                        continue
                    if self.get_attack_path(robot, workbench_block):
                        robot.block_model = True
                        attack_num += 1
                        break
                self.other_workbenchs_order.append(workbench_block)
        # 最后集中处理一下机器人状态:
        for robot in self.robots:
            if robot.block_model:
                if robot.get_sell() != -1:  # 先买个东西再去崽
                    robot.block_workbench_index = self.other_workbenchs_list.index(
                        robot.get_sell())
                    robot.target = robot.get_buy()
                    robot.set_path(self.m_map.get_float_path(
                        robot.loc, robot.target, self.blue_flag, False))
                    robot.status = Robot.MOVE_TO_BUY_STATUS
                else:  # 直接去找事
                    robot.block_workbench_index = self.other_workbenchs_list.index(
                        robot.get_buy())
                    robot.target = robot.get_buy()
                    robot.set_path(self.m_map.get_float_path(
                        robot.loc, robot.target, not self.blue_flag, False))
                    robot.status = Robot.BLOCK_OTRHER
                    self.rival_workbenchs[robot.target].attack_value = Workbench.MAX_ATTCK_VALUE

    def attack_one(self, robot: Robot, mast_run=False):
        '''
        一个闲着没事的机器人想要没事找事
        mast_run: 不要去买东西了
        '''
        for _ in range(len(self.other_workbenchs_order)):
            workbench_block = self.other_workbenchs_order.popleft()
            self.other_workbenchs_order.append(workbench_block)
            if self.get_attack_path(robot, workbench_block, mast_run):
                robot.block_model = True
                if robot.get_sell() != -1:  # 先买个东西再去崽
                    robot.target = robot.get_buy()
                    robot.set_path(self.m_map.get_float_path(
                        robot.loc, robot.target, self.blue_flag, False))
                    robot.status = Robot.MOVE_TO_BUY_STATUS
                else:  # 直接去找事
                    robot.target = robot.get_buy()
                    robot.set_path(self.m_map.get_float_path(
                        robot.loc, robot.target, not self.blue_flag, robot.item_type > 0))
                    robot.status = Robot.BLOCK_OTRHER
                    self.rival_workbenchs[robot.target].attack_value = Workbench.MAX_ATTCK_VALUE
                break

    def dis2target(self, idx_robot):
        idx_workbench = self.robots[idx_robot].target
        w_loc = self.workbenchs[idx_workbench].loc if self.robots[
            idx_robot].status != Robot.BLOCK_OTRHER else self.rival_workbenchs[idx_workbench].loc
        r_loc = self.robots[idx_robot].loc
        return np.sqrt((r_loc[0] - w_loc[0]) ** 2 + (r_loc[1] - w_loc[1]) ** 2)

    def re_mession(self, robot: Robot, frame_wait=0, destroy=False):
        '''
        当机器人长期无法完成任务时，尝试切换机器人状态
        frame_wait: 调用参方法后的等待帧数，如果还是无法完成任务，就倒车几帧
        destroy: 是否强制销毁, 如果为True, 4-6也会被销毁
        返回True为切换成功, False为切换失败, 只能撞开
        '''
        if frame_wait > 0:
            # 超时死锁处理，倒车几帧
            robot.frame_wait = frame_wait
            robot.backword_speed = -2
        robot.frame_reman_buy = 0
        robot.frame_reman_sell = 0
        sell_idx = robot.get_sell()
        buy_idx = robot.get_buy()
        # 取消预售预购
        self.workbenchs[buy_idx].pro_buy(False)
        self.workbenchs[sell_idx].pro_sell(
            self.workbenchs[robot.get_buy()].typeID, False)
        # 手中持有物品
        if robot.item_type > 0:
            item_type = robot.item_type
            # sys.stderr.write(f'item_type:{item_type}\n')
            # *(1<<self.black_workbenchs.get(sell_idx,0))
            self.can_not_reach_workbenchs[sell_idx] = self.MAX_CAN_NOT_REACH
            self.black_workbenchs[sell_idx] = self.black_workbenchs.get(
                sell_idx, 0)+1
            # 尝试找个地方卖了
            min_sell_frame = None
            for idx_workbench_to_sell in self.workbenchs[robot.get_buy()].target_workbench_list:
                if idx_workbench_to_sell in self.can_not_reach_workbenchs:
                    continue
                workbench_sell = self.workbenchs[idx_workbench_to_sell]
                if workbench_sell.check_material_pro(item_type):
                    continue
                if workbench_sell.check_material(item_type):
                    continue
                # frame_move_to_buy, frame_move_to_sell= self.get_time_rww(idx_robot, idx_workstand, idx_worksand_to_sell)
                frame_move_to_sell = len(self.m_map.get_path(robot.loc, idx_workbench_to_sell, self.blue_flag,
                                                             True)) * self.MOVE_SPEED
                if not min_sell_frame or min_sell_frame > frame_move_to_sell:
                    min_sell_frame = frame_move_to_sell
                    # sys.stderr.write(
                    #     f'last_sell:{sell_idx} new_sell:{idx_workbench_to_sell}\n')
                    robot.set_plan(robot.get_buy(), idx_workbench_to_sell)
                    robot.frame_reman_sell = frame_move_to_sell
            if min_sell_frame:
                robot.status = Robot.MOVE_TO_SELL_STATUS
                # 重设超时时间
                robot.frame_reman_sell = min_sell_frame
                # 预售
                self.workbenchs[robot.get_sell()].pro_sell(item_type, True)
                robot.target = robot.get_sell()
                self.re_path(robot)
                return True
            elif robot.item_type < 4 or (destroy and robot.item_type < 7):
                robot.destroy()
            else:
                # 重新预售, 直接返回
                self.workbenchs[sell_idx].pro_sell(
                    self.workbenchs[robot.get_buy()].typeID)
                return False
        else:
            # 设置工作台不可达状态
            # *(1<<self.black_workbenchs.get(buy_idx,0))
            self.can_not_reach_workbenchs[buy_idx] = self.MAX_CAN_NOT_REACH
            self.black_workbenchs[buy_idx] = self.black_workbenchs.get(
                buy_idx, 0)+1
        # 重置为空闲状态
        robot.status = Robot.FREE_STATUS
        return True

    def detect_deadlock(self, frame):
        # if frame % 10 != 0:
        #     return
        """
        每帧调用一次。当检测到两个机器人卡在一起时,会把机器人的成员变量 is_deadlock 设为True。
        在采取措施解决两个机器人死锁时, call set_robot_state_undeadlock(self,robot_idx, frame)
        把该机器人设为不死锁状态
        @param: frame: 当前的帧数
        """
        for robot in self.robots:
            frame_diff_to_detect = self.FRAME_DIFF_TO_DETECT_DEADLOCK + \
                random.randint(5, 30)
            distance = np.sqrt(
                np.sum(np.square(robot.loc_np - robot.pre_position)))
            toward_diff = abs(robot.toward - robot.pre_toward)
            toward_diff = min(toward_diff, 2 * np.pi - toward_diff)
            # 一帧内移动距离大于MIN_DIS_TO_DETECT_DEADLOCK，说明没有死锁，update

            if distance > self.MIN_DIS_TO_DETECT_DEADLOCK or \
                    toward_diff > self.MIN_TOWARD_DIF_TO_DETECT_STUCK:
                robot.update_frame_pisition(frame)
                robot.is_stuck = False
                continue

            if robot.is_stuck:
                robot.update_frame_pisition(frame)
                continue

            if frame - robot.pre_frame < frame_diff_to_detect:
                continue

            # 50帧内移动距离小于MIN_DIS_TO_DETECT_DEADLOCK
            if robot.status == robot.WAIT_TO_BUY_STATUS or robot.status == robot.WAIT_TO_SELL_STATUS:
                robot.update_frame_pisition(frame)
                continue
            robot.is_stuck = True
            # sys.stderr.write("检测到卡墙" + ",robot_id:" + str(robot.ID) + "\n")
        sys.stderr.flush()

    def set_robot_state_undeadlock(self, robot_idx, frame):
        """
        当开始做出解除死锁的动作时,调用此函数,把机器人设置为不死锁
        @param: robot_idx 机器人的idx
        @param: frame 当前的帧数
        """
        robot = self.robots[robot_idx]
        robot.update_frame_pisition(frame)
        robot.is_stuck = False

    def radar(self, idx_robot, d_theta):
        # 当前位置与朝向
        point = self.robots[idx_robot].loc
        theta = self.robots[idx_robot].toward + d_theta
        theta = (theta + math.pi) % (2 * math.pi) - math.pi
        # 当前位置所处的格子
        # raw, col = int(round(point[1] * 2 - 0.5)), int(round(point[0] * 2 - 0.5))
        raw, col = int(point[1] // 0.5), int(point[0] // 0.5)

        # 取出所有边界点
        if theta == 0:
            # 正右
            x_set_all = np.arange(col + 1, min(col + 4, 99), 1) * 0.5
            y_set_all = np.ones_like(x_set_all) * point[1]
        elif theta == math.pi / 2:
            # 正上
            y_set_all = np.arange(raw + 1, min(raw + 4, 99), 1) * 0.5
            x_set_all = np.ones_like(y_set_all) * point[0]
        elif theta == math.pi:
            # 正左
            x_set_all = np.arange(col, max(col - 3, 0), -1) * 0.5
            y_set_all = np.ones_like(x_set_all) * point[1]
        elif theta == -math.pi / 2:
            # 正下
            y_set_all = np.arange(raw, max(raw - 3, 0), -1) * 0.5
            x_set_all = np.ones_like(y_set_all) * point[0]
        else:
            # 其他方向

            # x方向栅格点集
            if -math.pi / 2 < theta < math.pi / 2:
                # 1 4 象限
                x_set_xgrid = np.arange(col + 1, min(col + 4, 99), 1) * 0.5
            else:
                # 2 3 象限
                x_set_xgrid = np.arange(col, max(col - 3, 0), -1) * 0.5
            y_set_xgrid = np.tan(theta) * (x_set_xgrid - point[0]) + point[1]

            # y方向栅格点集
            if 0 < theta < math.pi:
                # 1 2 象限
                y_set_ygrid = np.arange(raw + 1, min(raw + 4, 99), 1) * 0.5

            else:
                # 3 4 象限
                y_set_ygrid = np.arange(raw, max(raw - 3, 0), -1) * 0.5
            x_set_ygrid = 1 / np.tan(theta) * \
                (y_set_ygrid - point[1]) + point[0]
            x_set_all = np.concatenate((x_set_xgrid, x_set_ygrid))
            y_set_all = np.concatenate((y_set_xgrid, y_set_ygrid))

            # 得到排序后的索引
            idx = np.argsort(y_set_all)
            # 将坐标按照排序后的索引进行排序
            if theta < 0:
                x_set_all = x_set_all[idx]
                y_set_all = y_set_all[idx]
            else:
                x_set_all = x_set_all[idx[::-1]]
                y_set_all = y_set_all[idx[::-1]]

        # 取出所有边界点↑
        x_set_near = x_set_all[:-1]
        x_set_far = x_set_all[1:]

        y_set_near = y_set_all[:-1]
        y_set_far = y_set_all[1:]

        x_set_mid = (x_set_near + x_set_far) / 2
        y_set_mid = (y_set_near + y_set_far) / 2

        mask = np.zeros_like(x_set_mid, dtype=bool)
        mask[(x_set_mid >= 0) & (x_set_mid <= 50) & (
            y_set_mid >= 0) & (y_set_mid <= 50)] = True
        x_set_mid = x_set_mid[mask]
        y_set_mid = y_set_mid[mask]
        idx_ob = -1
        for i_point in range(len(x_set_mid)):
            x = x_set_mid[i_point]
            y = y_set_mid[i_point]
            raw, col = tools.cor2rc(x, y)

            if self.m_map_arr[raw, col] == 0:
                idx_ob = i_point
                break

        if idx_ob == -1:
            return 100
        else:
            # 障碍物距离
            return np.sqrt((x_set_near[idx_ob] - point[0]) ** 2 + (y_set_near[idx_ob] - point[1]) ** 2)

    def could_run(self, loc0, loc1, carry_flag):
        # 位置0到位置1区间是否有符合要求

        # loc0→loc1的距离
        dis = np.sqrt(np.sum((loc1 - loc0) ** 2))

        # loc0→loc1的方向
        theta = np.arctan2(loc1[1] - loc0[1], loc1[0] - loc0[0])

        # loc0所处的格子

        raw, col = int(loc0[1] // 0.5), int(loc0[0] // 0.5)

        max_num = np.ceil(dis / 0.5)
        # 取出所有边界点
        if theta == 0:
            # 正右
            x_set_all = np.arange(
                col + 1, min(col + max_num + 1, 101), 1) * 0.5
            y_set_all = np.ones_like(x_set_all) * loc0[1]
        elif theta == math.pi / 2:
            # 正上
            y_set_all = np.arange(
                raw + 1, min(raw + max_num + 1, 101), 1) * 0.5
            x_set_all = np.ones_like(y_set_all) * loc0[0]
        elif theta == math.pi:
            # 正左
            x_set_all = np.arange(col, max(col - max_num - 1, -1), -1) * 0.5
            y_set_all = np.ones_like(x_set_all) * loc0[1]
        elif theta == -math.pi / 2:
            # 正下
            y_set_all = np.arange(raw, max(raw - max_num - 1, -1), -1) * 0.5
            x_set_all = np.ones_like(y_set_all) * loc0[0]
        else:
            # 其他方向

            # x方向栅格点集
            if -math.pi / 2 < theta < math.pi / 2:
                # 1 4 象限
                x_set_xgrid = np.arange(
                    col + 1, min(col + max_num + 1, 101), 1) * 0.5
            else:
                # 2 3 象限
                x_set_xgrid = np.arange(
                    col, max(col - max_num - 1, -1), -1) * 0.5
            y_set_xgrid = np.tan(theta) * (x_set_xgrid - loc0[0]) + loc0[1]

            # y方向栅格点集
            if 0 < theta < math.pi:
                # 1 2 象限
                y_set_ygrid = np.arange(
                    raw + 1, min(raw + max_num + 1, 101), 1) * 0.5

            else:
                # 3 4 象限
                y_set_ygrid = np.arange(
                    raw, max(raw - max_num - 1, -1), -1) * 0.5
            x_set_ygrid = 1 / np.tan(theta) * \
                (y_set_ygrid - loc0[1]) + loc0[0]
            x_set_all = np.concatenate((x_set_xgrid, x_set_ygrid))
            y_set_all = np.concatenate((y_set_xgrid, y_set_ygrid))

            # 得到排序后的索引
            idx = np.argsort(y_set_all)
            # 将坐标按照排序后的索引进行排序
            if theta < 0:
                x_set_all = x_set_all[idx]
                y_set_all = y_set_all[idx]
            else:
                x_set_all = x_set_all[idx[::-1]]
                y_set_all = y_set_all[idx[::-1]]

        # 取出所有边界点↑
        x_set_near = x_set_all[:-1]
        x_set_far = x_set_all[1:]

        y_set_near = y_set_all[:-1]
        y_set_far = y_set_all[1:]

        x_set_mid = (x_set_near + x_set_far) / 2
        y_set_mid = (y_set_near + y_set_far) / 2

        dis_set = np.sqrt(
            (x_set_near - loc0[0]) ** 2 + (y_set_near - loc0[1]) ** 2)
        mask = np.zeros_like(x_set_mid, dtype=bool)
        mask[dis_set <= dis] = True
        x_set_mid = x_set_mid[mask]
        y_set_mid = y_set_mid[mask]
        idx_ob = -1
        count = 0
        thr_count = 3
        if carry_flag:
            # 携带物品
            for i_point in range(len(x_set_mid)):
                x = x_set_mid[i_point]
                y = y_set_mid[i_point]
                raw, col = tools.cor2rc(x, y)
                road_level = self.m_map_arr[raw, col]
                if raw <= -1 or raw >= 100 or col <= -1 or col >= 100 or road_level < Workmap.SUPER_BROAD_ROAD:
                    if road_level >= Workmap.BROAD_ROAD:
                        # 稍微不好 累计次数
                        count += 1
                    else:
                        # 太不好了，直接False
                        return False
                    if count >= thr_count:
                        # 计数次数过多
                        return False

        else:
            for i_point in range(len(x_set_mid)):
                x = x_set_mid[i_point]
                y = y_set_mid[i_point]
                raw, col = tools.cor2rc(x, y)
                road_level = self.m_map_arr[raw, col]
                if raw <= -1 or raw >= 100 or col <= -1 or col >= 100 or road_level < Workmap.BROAD_ROAD or (raw, col) in self.m_map.broad_shifting:
                    if road_level >= Workmap.BROAD_ROAD:
                        # 宽度符合要求
                        pass
                    elif road_level >= Workmap.ROAD:
                        # 不太好 进行计数
                        count += 1
                    else:
                        # 太不好
                        return False
                    if count >= thr_count:
                        # 计次超时
                        return False

        # sys.stderr.write('huq\n')
        return True
        #     # 全程无障碍
        #     return True
        # else:
        #     # 出现障碍
        #     return False

    def obt_detect(self, loc0, loc1):
        # 和could_run 区别在于只检测是否为障碍物
        # 位置0到位置1区间是否有符合要求

        # loc0→loc1的距离
        dis = np.sqrt(np.sum((loc1 - loc0) ** 2))

        # loc0→loc1的方向
        theta = np.arctan2(loc1[1] - loc0[1], loc1[0] - loc0[0])

        # loc0所处的格子

        raw, col = int(loc0[1] // 0.5), int(loc0[0] // 0.5)

        max_num = np.ceil(dis / 0.5)
        # 取出所有边界点
        if theta == 0:
            # 正右
            x_set_all = np.arange(
                col + 1, min(col + max_num + 1, 101), 1) * 0.5
            y_set_all = np.ones_like(x_set_all) * loc0[1]
        elif theta == math.pi / 2:
            # 正上
            y_set_all = np.arange(
                raw + 1, min(raw + max_num + 1, 101), 1) * 0.5
            x_set_all = np.ones_like(y_set_all) * loc0[0]
        elif theta == math.pi:
            # 正左
            x_set_all = np.arange(col, max(col - max_num - 1, -1), -1) * 0.5
            y_set_all = np.ones_like(x_set_all) * loc0[1]
        elif theta == -math.pi / 2:
            # 正下
            y_set_all = np.arange(raw, max(raw - max_num - 1, -1), -1) * 0.5
            x_set_all = np.ones_like(y_set_all) * loc0[0]
        else:
            # 其他方向

            # x方向栅格点集
            if -math.pi / 2 < theta < math.pi / 2:
                # 1 4 象限
                x_set_xgrid = np.arange(
                    col + 1, min(col + max_num + 1, 101), 1) * 0.5
            else:
                # 2 3 象限
                x_set_xgrid = np.arange(
                    col, max(col - max_num - 1, -1), -1) * 0.5
            y_set_xgrid = np.tan(theta) * (x_set_xgrid - loc0[0]) + loc0[1]

            # y方向栅格点集
            if 0 < theta < math.pi:
                # 1 2 象限
                y_set_ygrid = np.arange(
                    raw + 1, min(raw + max_num + 1, 101), 1) * 0.5

            else:
                # 3 4 象限
                y_set_ygrid = np.arange(
                    raw, max(raw - max_num - 1, -1), -1) * 0.5
            x_set_ygrid = 1 / np.tan(theta) * \
                (y_set_ygrid - loc0[1]) + loc0[0]
            x_set_all = np.concatenate((x_set_xgrid, x_set_ygrid))
            y_set_all = np.concatenate((y_set_xgrid, y_set_ygrid))

            # 得到排序后的索引
            idx = np.argsort(y_set_all)
            # 将坐标按照排序后的索引进行排序
            if theta < 0:
                x_set_all = x_set_all[idx]
                y_set_all = y_set_all[idx]
            else:
                x_set_all = x_set_all[idx[::-1]]
                y_set_all = y_set_all[idx[::-1]]

        # 取出所有边界点↑
        x_set_near = x_set_all[:-1]
        x_set_far = x_set_all[1:]

        y_set_near = y_set_all[:-1]
        y_set_far = y_set_all[1:]

        x_set_mid = (x_set_near + x_set_far) / 2
        y_set_mid = (y_set_near + y_set_far) / 2

        dis_set = np.sqrt(
            (x_set_near - loc0[0]) ** 2 + (y_set_near - loc0[1]) ** 2)
        mask = np.zeros_like(x_set_mid, dtype=bool)
        mask[dis_set <= dis] = True
        x_set_mid = x_set_mid[mask]
        y_set_mid = y_set_mid[mask]
        idx_ob = -1
        for i_point in range(len(x_set_mid)):
            x = x_set_mid[i_point]
            y = y_set_mid[i_point]
            raw, col = tools.cor2rc(x, y)
            if raw <= -1 or raw >= 100 or col <= -1 or col >= 100 or self.m_map_arr[raw, col] == 0:
                return False
        return True

    def select_target(self, idx_robot):
        # 雷达前的版本
        robot = self.robots[idx_robot]
        robot_loc_m = np.array(robot.loc).copy()
        path_loc_m = robot.path.copy()
        vec_r2p = robot_loc_m - path_loc_m
        dis_r2p = np.sqrt(np.sum(vec_r2p ** 2, axis=1))
        mask_greater = dis_r2p < 40
        mask_smaller = 0.3 < dis_r2p
        mask = mask_greater & mask_smaller
        path_loc_m = path_loc_m[mask]
        if self.robots[idx_robot].item_type == 0:
            carry_flag = False
        else:
            carry_flag = True

        # robot_loc_m 指向各个点的方向
        theta_set = np.arctan2(
            path_loc_m[:, 1] - robot_loc_m[1], path_loc_m[:, 0] - robot_loc_m[0]).reshape(-1, 1)

        len_path = path_loc_m.shape[0]
        detect_m = np.full(len_path, False)

        for idx_point in range(len_path):
            loc0 = robot_loc_m
            loc1 = path_loc_m[idx_point, :]
            detect_m[idx_point] = self.could_run(loc0, loc1, carry_flag)

        if detect_m.any():
            m_index = np.where(detect_m)[0]
            idx_target = m_index[len(m_index) - 1]
            target_point = path_loc_m[idx_target, :]
            idx_point = np.where(mask)[0][idx_target]
        # elif detect_2.any():
        #     m_index = np.where(detect_2)[0]
        #     idx_target = m_index[len(m_index) - 1]
        #     target_point = path_loc_m[idx_target, :]
        #     idx_point = np.where(mask)[0][idx_target]
        else:
            idx_target = robot.find_temp_tar_idx()
            target_point = robot.path[idx_target, :]
            idx_point = idx_target
        # ,

        # 修正select导致在墙角导致回头的现象：
        # 如果机器人已经选中的点下一个点并接近选中点的下一个点
        # 强条件修正
        if idx_point < len(robot.path) - 1:
            # 取出机器人坐标
            point_robot = np.array(robot.loc)

            # 取出当前点
            target_point_this = robot.path[idx_target, :]

            # 取出下一个点
            target_point_next = robot.path[idx_target + 1, :]

            # 下一个点和当前点的距离
            dis_this2next = np.sqrt(np.sum((target_point_next - target_point_this) ** 2))

            # 下一个点和机器人的距离
            dis_robot2next = np.sqrt(np.sum((target_point_next - point_robot) ** 2))

            if dis_robot2next < dis_this2next:
                # 如果下一个点和机器人的距离小于当前点和下一个点的距离
                # 则将下一个点作为目标点
                target_point = target_point_next
                idx_point = idx_target + 1

        # 修正结束

        return target_point, idx_point

    def obt_detect(self, loc0, loc1):
        # 位置0到位置1区间是否有符合要求

        # loc0→loc1的距离
        dis = np.sqrt(np.sum((loc1 - loc0) ** 2))

        # loc0→loc1的方向
        theta = np.arctan2(loc1[1] - loc0[1], loc1[0] - loc0[0])

        # loc0所处的格子

        raw, col = int(loc0[1] // 0.5), int(loc0[0] // 0.5)

        max_num = np.ceil(dis / 0.5)
        # 取出所有边界点
        if theta == 0:
            # 正右
            x_set_all = np.arange(
                col + 1, min(col + max_num + 1, 101), 1) * 0.5
            y_set_all = np.ones_like(x_set_all) * loc0[1]
        elif theta == math.pi / 2:
            # 正上
            y_set_all = np.arange(
                raw + 1, min(raw + max_num + 1, 101), 1) * 0.5
            x_set_all = np.ones_like(y_set_all) * loc0[0]
        elif theta == math.pi:
            # 正左
            x_set_all = np.arange(col, max(col - max_num - 1, -1), -1) * 0.5
            y_set_all = np.ones_like(x_set_all) * loc0[1]
        elif theta == -math.pi / 2:
            # 正下
            y_set_all = np.arange(raw, max(raw - max_num - 1, -1), -1) * 0.5
            x_set_all = np.ones_like(y_set_all) * loc0[0]
        else:
            # 其他方向

            # x方向栅格点集
            if -math.pi / 2 < theta < math.pi / 2:
                # 1 4 象限
                x_set_xgrid = np.arange(
                    col + 1, min(col + max_num + 1, 101), 1) * 0.5
            else:
                # 2 3 象限
                x_set_xgrid = np.arange(
                    col, max(col - max_num - 1, -1), -1) * 0.5
            y_set_xgrid = np.tan(theta) * (x_set_xgrid - loc0[0]) + loc0[1]

            # y方向栅格点集
            if 0 < theta < math.pi:
                # 1 2 象限
                y_set_ygrid = np.arange(
                    raw + 1, min(raw + max_num + 1, 101), 1) * 0.5

            else:
                # 3 4 象限
                y_set_ygrid = np.arange(
                    raw, max(raw - max_num - 1, -1), -1) * 0.5
            x_set_ygrid = 1 / np.tan(theta) * \
                          (y_set_ygrid - loc0[1]) + loc0[0]
            x_set_all = np.concatenate((x_set_xgrid, x_set_ygrid))
            y_set_all = np.concatenate((y_set_xgrid, y_set_ygrid))

            # 得到排序后的索引
            idx = np.argsort(y_set_all)
            # 将坐标按照排序后的索引进行排序
            if theta < 0:
                x_set_all = x_set_all[idx]
                y_set_all = y_set_all[idx]
            else:
                x_set_all = x_set_all[idx[::-1]]
                y_set_all = y_set_all[idx[::-1]]

        # 取出所有边界点↑
        x_set_near = x_set_all[:-1]
        x_set_far = x_set_all[1:]

        y_set_near = y_set_all[:-1]
        y_set_far = y_set_all[1:]

        x_set_mid = (x_set_near + x_set_far) / 2
        y_set_mid = (y_set_near + y_set_far) / 2

        dis_set = np.sqrt(
            (x_set_near - loc0[0]) ** 2 + (y_set_near - loc0[1]) ** 2)
        mask = np.zeros_like(x_set_mid, dtype=bool)
        mask[dis_set <= dis] = True
        x_set_mid = x_set_mid[mask]
        y_set_mid = y_set_mid[mask]
        idx_ob = -1
        for i_point in range(len(x_set_mid)):
            x = x_set_mid[i_point]
            y = y_set_mid[i_point]
            raw, col = tools.cor2rc(x, y)
            if raw <= -1 or raw >= 100 or col <= -1 or col >= 100 or self.m_map_arr[raw, col] == 0:
                return False
        return True


    def select_target_4(self, idx_robot):
        # 4条线

        # 取出机器人
        robot = self.robots[idx_robot]

        # 取出机器人坐标 质心
        robot_loc_m = np.array(robot.loc).copy()

        # 取出目标点坐标
        path_loc_m = robot.path.copy()

        # 计算机器人到目标点的向量
        vec_r2p = robot_loc_m - path_loc_m

        # 计算机器人到目标点的距离
        dis_r2p = np.sqrt(np.sum(vec_r2p ** 2, axis=1))

        # 选取距离小于40 大于0.3的mask
        mask_greater = dis_r2p < 40
        mask_smaller = 0.3 < dis_r2p
        mask = mask_greater & mask_smaller

        # 选取距离小于40 大于0.3的点
        path_loc_m = path_loc_m[mask]

        # 判断是否携带物品，并计算相应半径
        if self.robots[idx_robot].item_type == 0:
            carry_flag = False
            width = 0.46
        else:
            carry_flag = True
            width = 0.54

        # robot_loc_m 指向各个点的方向
        theta_set = np.arctan2(
            path_loc_m[:, 1] - robot_loc_m[1], path_loc_m[:, 0] - robot_loc_m[0]).reshape(-1, 1)

        # 方向左旋90度
        theta_l_set = theta_set + math.pi / 2

        # 方向右旋90度
        theta_r_set = theta_set - math.pi / 2

        # 计算左右两边的点 4个点把宽度3等分
        robot_loc_l1 = robot_loc_m + 1 / 3 * width * \
                       np.concatenate((np.cos(theta_l_set), np.sin(theta_l_set)), axis=1)
        robot_loc_l2 = robot_loc_m + width * \
                       np.concatenate((np.cos(theta_l_set), np.sin(theta_l_set)), axis=1)
        robot_loc_r1 = robot_loc_m + 1 / 3 * width * \
                       np.concatenate((np.cos(theta_r_set), np.sin(theta_r_set)), axis=1)
        robot_loc_r2 = robot_loc_m + width * \
                       np.concatenate((np.cos(theta_r_set), np.sin(theta_r_set)), axis=1)

        path_loc_l1 = path_loc_m + 1 / 3 * width * \
                      np.concatenate((np.cos(theta_l_set), np.sin(theta_l_set)), axis=1)
        path_loc_l2 = path_loc_m + width * \
                      np.concatenate((np.cos(theta_l_set), np.sin(theta_l_set)), axis=1)
        path_loc_r1 = path_loc_m + 1 / 3 * width * \
                      np.concatenate((np.cos(theta_r_set), np.sin(theta_r_set)), axis=1)
        path_loc_r2 = path_loc_m + width * \
                      np.concatenate((np.cos(theta_r_set), np.sin(theta_r_set)), axis=1)
        len_path = path_loc_m.shape[0]

        detect_l1 = np.full(len_path, False)
        detect_l2 = np.full(len_path, False)
        detect_r1 = np.full(len_path, False)
        detect_r2 = np.full(len_path, False)

        for idx_point in range(len_path):
            # 共4条线的判断
            loc0 = robot_loc_l1[idx_point, :]
            loc1 = path_loc_l1[idx_point, :]
            detect_l1[idx_point] = self.obt_detect(loc0, loc1)

            loc0 = robot_loc_l2[idx_point, :]
            loc1 = path_loc_l2[idx_point, :]
            detect_l2[idx_point] = self.obt_detect(loc0, loc1)

            loc0 = robot_loc_r1[idx_point, :]
            loc1 = path_loc_r1[idx_point, :]
            detect_r1[idx_point] = self.obt_detect(loc0, loc1)

            loc0 = robot_loc_r2[idx_point, :]
            loc1 = path_loc_r2[idx_point, :]
            detect_r2[idx_point] = self.obt_detect(loc0, loc1)


        detect_all = detect_l1 & detect_l2 & detect_r1 & detect_r2
        detect_m = detect_l1 | detect_r1

        if detect_all.any():
            m_index = np.where(detect_all)[0]
            idx_target = m_index[len(m_index) - 1]
            target_point = path_loc_m[idx_target, :]
            idx_point = np.where(mask)[0][idx_target]
        # elif detect_2.any():
        #     m_index = np.where(detect_2)[0]
        #     idx_target = m_index[len(m_index) - 1]
        #     target_point = path_loc_m[idx_target, :]
        #     idx_point = np.where(mask)[0][idx_target]
        else:
            idx_target = robot.find_temp_tar_idx()
            target_point = robot.path[idx_target, :]
            idx_point = idx_target
        # ,
        # 修正select导致在墙角导致回头的现象：
        # 如果机器人已经选中的点下一个点并接近选中点的下一个点
        # 强条件修正
        if idx_point < len(robot.path) - 1:
            # 取出机器人坐标
            point_robot = np.array(robot.loc)

            # 取出当前点
            target_point_this = robot.path[idx_target, :]

            # 取出下一个点
            target_point_next = robot.path[idx_target + 1, :]

            # 下一个点和当前点的距离
            dis_this2next = np.sqrt(np.sum((target_point_next - target_point_this) ** 2))

            # 下一个点和机器人的距离
            dis_robot2next = np.sqrt(np.sum((target_point_next - point_robot) ** 2))

            if dis_robot2next < dis_this2next:
                # 如果下一个点和机器人的距离小于当前点和下一个点的距离
                # 则将下一个点作为目标点
                target_point = target_point_next
                idx_point = idx_target + 1

        # 修正结束

        return target_point, idx_point


    def get_other_col_info2(self, idx_robot, idx_other, t_max=0.3):
        # 返回t_max时间内 最短质心距离
        obt_flag = self.obt_detect(
            np.array(self.robots[idx_robot].loc), np.array(self.robots[idx_other].loc))
        if not obt_flag:
            # 有障碍
            return 100
        robot_this = self.robots[idx_robot]
        robot_other = self.robots[idx_other]
        vec_o2t = np.array(robot_this.loc) - np.array(robot_other.loc)
        theta_o2t = np.arctan2(vec_o2t[1], vec_o2t[0])
        theta_other = robot_other.toward
        delta_theta = theta_other - theta_o2t
        delta_theta = (delta_theta +
                       math.pi) % (2 * math.pi) - math.pi

        vx_robot = robot_this.speed[0]
        vy_robot = robot_this.speed[1]
        x_robot = robot_this.loc[0]
        y_robot = robot_this.loc[1]

        vx_other = robot_other.speed[0]
        vy_other = robot_other.speed[1]
        x_other = robot_other.loc[0]
        y_other = robot_other.loc[1]

        # 判断是否路上正向对撞
        d = tools.will_collide2(
            x_robot, y_robot, vx_robot, vy_robot, x_other, y_other, vx_other, vy_other, t_max)
        # 判断是否路上侧向撞上其他机器人
        # 判断是否同时到终点僵持
        return d

    def direct_colli(self, idx_robot, idx_other, thr_dis=4, thr_theta=math.pi / 5):
        robot_this = self.robots[idx_robot]
        robot_other = self.robots[idx_other]

        raw, col = self.m_map.loc_float2int(*robot_this.loc)

        # 要在大空地 且 离目标工作台足够远
        if self.m_map_arr[raw, col] == Workmap.SUPER_BROAD_ROAD and self.dis2target(idx_robot) > 5:

            loc_this = np.array(robot_this.loc)
            loc_other = np.array(robot_other.loc)
            vec_this2other = loc_other - loc_this
            vec_other2this = loc_this - loc_other

            dis = np.sqrt(np.dot(vec_this2other, vec_this2other))

            # 距离要足够近
            if dis < thr_dis:
                #############################
                # 本机器人头朝向
                theta_toward_this = robot_this.toward

                # 本机器人速度朝向
                speed_vec_this = np.array(robot_this.speed)
                theta_speed_this = np.arctan2(
                    speed_vec_this[1], speed_vec_this[0])

                # 本机器人 对方相对于自身朝向
                theta_robot_this = np.arctan2(
                    vec_this2other[1], vec_this2other[0])

                #############################
                # 其他机器人头朝向
                theta_toward_other = robot_other.toward

                # 其他机器人速度朝向
                speed_vec_other = np.array(robot_other.speed)
                theta_speed_other = np.arctan2(
                    speed_vec_other[1], speed_vec_other[0])

                # 其他机器人 自身相对于对方朝向
                theta_robot_other = np.arctan2(
                    vec_other2this[1], vec_other2this[0])

                # 撞向对方的可能
                if max(abs(theta_toward_this - theta_robot_this), abs(theta_speed_this - theta_robot_this), abs(theta_toward_other - theta_robot_other), abs(theta_speed_other - theta_robot_other)) < thr_theta:
                    return True
            return False

    def AF(self, loc_robot):
        row, col = self.m_map.loc_float2int(loc_robot[0], loc_robot[1])
        row_start = max(row - 2, 0)
        row_stop = min(row + 2, 99)
        col_start = max(col - 2, 0)
        col_stop = min(col + 2, 99)
        offset = np.array([0, 0])
        for i_row in range(row_start, row_stop + 1):
            for i_col in range(col_start, col_stop + 1):
                if self.m_map_arr[i_row, i_col] == 0:
                    # 这是一个障碍物
                    loc_obt = np.array(self.m_map.loc_int2float(i_row, i_col))
                    dis2loc = tools.np_norm(loc_robot, loc_obt)
                    if dis2loc < 1:
                        theta = tools.np_theta(loc_obt, loc_robot)

                        offset = offset + 0.5 * \
                            np.array([np.cos(theta), np.cos(theta)])
        # sys.stderr.write(f"势场修正:{offset[0]},{offset[1]}\n")
        return offset

    def obt_near(self, robot):
        # 更新实现，利用官方雷达
        # 提取打到障碍物激光点的距离
        dis_obt = robot.radar_info_dis[robot.radar_info_obt]

        # 判定小于阈值的点
        judge = dis_obt < 0.85  # 原来是0.7

        # 存在 返回
        if judge.any():
            # 周围有障碍物
            return False
        else:
            # 周围没有障碍物
            return True
    # 尝试找更好的路径

    def obt_near_path(self, robot):
        # 更新实现，利用官方雷达
        # 提取打到障碍物激光点的距离
        dis_obt = robot.radar_info_dis[robot.radar_info_obt]

        # 判定小于阈值的点
        judge = dis_obt < 0.7

        # 存在 返回
        if judge.any():
            # 周围有障碍物
            return False
        else:
            # 周围没有障碍物
            return True

    def obt_near_count(self, robot):
        row, col = self.m_map.loc_float2int(*robot.loc)
        count = 0
        for row_offset, col_offset in Workmap.TURNS:
            new_row = row + row_offset
            new_col = col + col_offset
            if new_col < 0 or new_col > 99 or new_row < 0 or new_col > 99 or self.m_map.map_gray[new_row][
                    new_col] == Workmap.BLOCK:
                count += 1
        return count

    def rival_filter(self, radar_x, radar_y, rival_list):
        # 已扫描到的敌方机器人
        num_rival = len(rival_list)

        # 可优化 我不知道怎么直接生成全true
        mask_filter = radar_x > -1
        if num_rival > 0:
            for idx in range(num_rival):
                # 敌方机器人坐标
                loc_filter = rival_list[idx][0]  # [[(x, y), r], [(x, y), r]]
                r_filter = rival_list[idx][1]

                # 相对于敌方机器人的xy
                radar_rel_x = radar_x - loc_filter[0]
                radar_rel_y = radar_y - loc_filter[1]

                # 相对于敌方机器人的距离
                dis_rel = np.sqrt(radar_rel_x ** 2 + radar_rel_y ** 2)

                # 距离已知敌方机器人远的才是潜在的敌方机器人
                mask_filter_temp = dis_rel > r_filter + 0.01

                # 按位与更新
                mask_filter = mask_filter & mask_filter_temp

        return mask_filter

    def our_filter(self, radar_x, radar_y):
        # 排除扫描到己方机器人的情况

        # 可优化 我不知道怎么直接生成全true
        mask_filter = radar_x > -1

        for robot_filter in self.robots:
            # 己方机器人坐标
            loc_filter = robot_filter.loc

            # 己方机器人半径
            if robot_filter.item_type == 0:
                r_filter = 0.45
            else:
                r_filter = 0.53

            # 相对于己方机器人的xy
            radar_rel_x = radar_x - loc_filter[0]
            radar_rel_y = radar_y - loc_filter[1]

            # 相对于己方机器人的距离
            dis_rel = np.sqrt(radar_rel_x ** 2 + radar_rel_y ** 2)

            # 距离己方机器人远的才是潜在的敌方机器人
            mask_filter_temp = dis_rel > r_filter + 0.001

            # 按位与更新
            mask_filter = mask_filter & mask_filter_temp

        return mask_filter

    def detect_rival_item(self, robot, thr_dis, rival_list):
        loc_this = robot.loc
        radar_x = robot.radar_info_x
        radar_y = robot.radar_info_y

        # mask为true表示为潜在的机器人 通过已知障碍物排除
        # mask = tools.is_multiple_of_half(radar_y) & tools.is_multiple_of_half(radar_x)
        mask = np.logical_not(robot.radar_info_obt)
        # 相对于当前机器人的xy
        radar_rel_x = radar_x - loc_this[0]
        radar_rel_y = radar_y - loc_this[1]

        # 相对于当前机器人的距离
        dis_rel = np.sqrt(radar_rel_x ** 2 + radar_rel_y ** 2)

        # 通过相对于当前机器人距离排除 越远探测效果越差
        mask_dis = dis_rel < thr_dis

        # 从己方坐标排除
        mask_our = self.our_filter(radar_x, radar_y)

        for _ in range(4):
            # 敌方最多4个机器人

            # 从对方坐标排除
            mask_rival = self.rival_filter(radar_x, radar_y, rival_list)

            # 更新
            mask = mask & mask_dis & mask_our & mask_rival

            if sum(mask) > 0:
                # 存在有效过滤数据

                # 建立布尔表与原索引的映射
                index_map = [i for i, m in enumerate(mask) if m]

                # 取出经过过滤后的雷达数据
                radar_x_filter = radar_x[mask]
                radar_y_filter = radar_y[mask]

                # 找出过滤后最近点
                radar_x_l2r = radar_x_filter - loc_this[0]
                radar_y_l2r = radar_y_filter - loc_this[1]
                dis_l2r = np.sqrt(radar_x_l2r ** 2 + radar_y_l2r ** 2)

                # 取出距离最近的索引（不易与其他机器人误判）
                try:
                    idx_min = np.argmin(dis_l2r)
                except:
                    raise Exception(dis_l2r)

                # 还原到过滤前的索引
                idx_ori = index_map[idx_min]
                if idx_ori == 1:
                    idx_set = [359, 0, 1, 2, 3]
                elif idx_ori == 0:
                    # 边界值
                    idx_set = [358, 359, 0, 1, 2]
                elif idx_ori == 359:
                    # 边界值
                    idx_set = [357, 358, 359, 0, 1]
                elif idx_ori == 358:
                    # 边界值
                    idx_set = [356, 357, 358, 359, 0]
                else:
                    # 其他值
                    idx_set = [idx_ori - 2, idx_ori - 1,
                               idx_ori, idx_ori + 1, idx_ori + 2]

                # 从360个点中选取5个点（增强鲁棒性）
                mask_try = mask[idx_set]

                # 从5个点中选取3个点（圆心定位只能3个点）
                if len(mask_try) > 0 and sum(mask_try) >= 3:
                    true_indices = []
                    count = 0
                    for index in idx_set:
                        if mask[index]:
                            true_indices.append(index)
                            count += 1
                            # 如果已经找到了3个True，则退出循环
                            if count == 3:
                                break
                    # 取出前三个，定位算法只能取三个点。
                    idx_set = true_indices
                    if count == 3:
                        # 最近点的邻近3点都符合
                        x_circle = radar_x[idx_set]
                        y_circle = radar_y[idx_set]
                        x0, y0, r = tools.calculate_circle(x_circle, y_circle)
                        # time.sleep(1)
                        if r < 0.543:
                            rival_list.append([(x0, y0), r])

        robot.radar_info_rival = np.logical_not(mask_rival)


    def get_target(self, idx_robot):
        robot = self.robots[idx_robot]
        # True 周围无障碍物 False 周围有障碍物
        flag_obt_near = self.obt_near(robot)
        if flag_obt_near:
            target_loc, target_idx = self.select_target_4(idx_robot)
        else:
            target_idx = robot.find_temp_tar_idx()
            target_loc = robot.path[target_idx, :]
        robot.target_loc = target_loc
        robot.target_idx = target_idx
        return target_loc, target_idx

    def get_temp_loc(self, idx_robot):
        # 获取指定机器人的临时目标点
        # 根据机器人距离当前临时目标点距离决定继续追踪或是重新规划并选择目标点
        robot = self.robots[idx_robot]

        if robot.target_loc is None:
            # 没有临时目标点则重新规划
            self.re_path(robot)
            target_loc, target_idx = self.get_target(idx_robot)
            robot.repath_wait = 30
        else:
            # 有临时目标点
            dis_temp_target = np.sqrt(
                np.sum((robot.target_loc - np.array(robot.loc)) ** 2))
            # 原本是 dis_temp_target > 0.35
            if robot.repath_wait > 0:
                robot.repath_wait -= 1
            # 足够接近或者在非避让状态下已经等待了一段时间
            if (robot.frame_wait > 0 and dis_temp_target < 2) or (robot.frame_wait == 0 and dis_temp_target < 2) or (
                    robot.frame_wait == 0 and robot.repath_wait <= 0):
                self.re_path(robot)
                target_loc, target_idx = self.get_target(idx_robot)
                robot.repath_wait = 30
            else:
                target_loc = robot.target_loc
                target_idx = robot.target_idx

        return target_loc, target_idx

    def avoid_our(self, idx_robot, dis2workbench, target_loc, target_idx):
        # 避让我方机器人

        robot = self.robots[idx_robot]
        # 因为移动过程中可能导致阻塞而避让, 可以解除顶牛, 可能导致HUQ
        col_flag = False

        # 因为买卖而产生的避让
        sb_flag = False
        # 是否要采取保持距离的方式
        sb_safe_dis = False
        d = 100
        # 要避让的机器人序号
        idx_huq = -1
        for idx_other in range(4):
            if not idx_other == idx_robot:
                d = min(self.get_other_col_info2(
                    idx_robot, idx_other), d)
                if d < self.WILL_CLASH_DIS:
                    col_flag = True
                    idx_huq = idx_other
                    break
        robot_target = robot.target
        # 初始化一个较大值
        other_dis2workbench = self.WILL_HUQ_DIS

        if dis2workbench < self.WILL_HUQ_DIS and not col_flag and robot.status in [Robot.MOVE_TO_BUY_STATUS,
                                                                                   Robot.WAIT_TO_BUY_STATUS]:
            for idx_other in range(4):
                # 锐总说这不合适吧
                if (not idx_other == idx_robot) and self.robots[idx_other].frame_wait == 0 and robot_target == \
                        self.robots[idx_other].target:
                    # 另一个机器人到工作台的距离
                    other_dis2workbench = self.dis2target(idx_other)
                    status_other = self.robots[idx_other].status
                    if other_dis2workbench > self.WILL_HUQ_DIS:
                        continue
                    # 买的让卖的
                    if status_other in [Robot.MOVE_TO_SELL_STATUS, Robot.WAIT_TO_SELL_STATUS]:
                        sb_flag = True
                    # 同买, 近的让远的
                    elif status_other in [Robot.MOVE_TO_BUY_STATUS, Robot.WAIT_TO_BUY_STATUS]:
                        if dis2workbench > other_dis2workbench:
                            sb_flag = True
                        elif dis2workbench == other_dis2workbench and idx_robot > idx_other:
                            sb_flag = True
                    if sb_flag:
                        idx_huq = idx_other
                        break
        if sb_flag and dis2workbench > other_dis2workbench:
            sb_safe_dis = True

        if col_flag or (sb_flag and not sb_safe_dis):
            priority_idx = -1
            if col_flag:
                status_huq = self.robots[idx_huq].status
                huq_dis2workbench = self.dis2target(idx_huq)
                if robot_target == self.robots[idx_huq].target:
                    # 我买对方卖
                    if robot.status in [Robot.MOVE_TO_BUY_STATUS, Robot.WAIT_TO_BUY_STATUS] and status_huq in [
                            Robot.MOVE_TO_SELL_STATUS, Robot.WAIT_TO_SELL_STATUS]:
                        priority_idx = idx_robot
                    # 我卖对方买
                    elif robot.status in [Robot.MOVE_TO_SELL_STATUS, Robot.WAIT_TO_SELL_STATUS] and status_huq in [
                            Robot.MOVE_TO_BUY_STATUS, Robot.WAIT_TO_BUY_STATUS]:
                        priority_idx = idx_huq
                    # 同买同卖
                    else:
                        if dis2workbench > huq_dis2workbench:
                            priority_idx = idx_robot
                        elif dis2workbench < huq_dis2workbench:
                            priority_idx = idx_huq
            else:
                priority_idx = idx_robot
            self.re_path(robot)
            self.re_path(self.robots[idx_huq])
            avoid_idx, avoid_path = self.process_deadlock(
                idx_robot, idx_huq, priority_idx)
            # sys.stderr.write(f"avoid_idx: {avoid_idx}\n")
            if avoid_idx == -1:
                # target_loc, target_idx = self.get_target(idx_robot)
                sb_safe_dis = True
                pass
            elif avoid_idx == idx_robot:
                self.robots[idx_robot].set_path(avoid_path)
                self.robots[idx_robot].frame_wait = self.AVOID_FRAME_WAIT
                target_loc, target_idx = self.get_target(idx_robot)
        return col_flag, sb_flag, sb_safe_dis, d, target_loc, target_idx

    def move(self, idx_robot):
        # 新版move

        # 控制参数：
        k_r = 8  # 定位旋转时的比例控制系数
        k_f = 3  # 定位前进时的比例控制系数
        thr_near_target = 5  # 小于此角度不避让对方机器人

        #  取出机器人的引用
        robot = self.robots[idx_robot]

        # 到工作台距离 用于判定是否接近目标工作台
        dis2workbench = self.dis2target(idx_robot)

        # 获取临时目标点
        target_loc, target_idx = self.get_temp_loc(idx_robot)

        # 获取对我方机器人的避让信息 此处可能更新path以及targe_loc 因此放在self.get_temp_loc_bck(idx_robot)的后面
        col_flag, sb_flag, sb_safe_dis, d, target_loc, target_idx = self.avoid_our(
            idx_robot, dis2workbench, target_loc, target_idx)

        # 障碍物避让方法：
        # True：避让敌方和障碍物
        # False:仅避让静态障碍物

        # 高鹏注意这里的修改
        row, col = self.m_map.loc_float2int(*robot.loc)
        road_level = self.m_map_arr[row, col]
        flag_avoid_rival = dis2workbench > thr_near_target and robot.status != Robot.BLOCK_OTRHER and road_level == Workmap.SUPER_BROAD_ROAD and robot.item_type > 3
        # 获取障碍物避让控制信息
        # flag：True 有障碍物 False 无障碍物
        # theta_avoid_obt：避让障碍物的角度 偏移角度
        # try:
        # flag_avoid_obt, d_theta_avoid_obt = robot.avoid_obt(t=0.5, target_loc=target_loc,
                                                            # flag_avoid_rival=flag_avoid_rival)
        flag_avoid_obt = False
        d_theta_avoid_obt = 0
        # except:
        #     import debug
        #     debug.save_controller(self)
        #     raise Exception("temp_target", target_loc, "loc", robot.loc)
        # 根据给定目标点计算角度偏移和
        target_vec = [target_loc[0] - robot.loc[0],
                      target_loc[1] - robot.loc[1]]
        dis_target = np.sqrt(np.dot(target_vec, target_vec))

        target_theta = np.arctan2(
            target_vec[1], target_vec[0])

        robot_theta = self.robots[idx_robot].toward
        delta_theta = target_theta - robot_theta
        delta_theta = (delta_theta +
                       math.pi) % (2 * math.pi) - math.pi
        if robot.status == Robot.BLOCK_OTRHER:
            # 干扰敌人的机器人
            if robot.attack_status == Robot.MOV_TO_ATTACK:
                # 前往干扰工作台的路上
                if abs(delta_theta) > math.pi / 6:
                    # 角度相差较大 原地转向
                    robot.forward(0)
                elif self.target_slow(idx_robot, target_idx, target_loc, col_flag, sb_flag, sb_safe_dis):
                    # 慢速行驶至目标
                    robot.forward(dis_target * k_f)
                else:
                    # 高速行驶至目标
                    robot.forward(9)

                robot.rotate(delta_theta * k_r)
                if dis2workbench < 0.2:
                    # 到达敌方工作台，切换为等待攻击状态
                    robot.attack_status = Robot.WAIT_TO_ATTACK

            if robot.attack_status == Robot.WAIT_TO_ATTACK:
                # 等待敌人接近工作台

                # 检查是否有敌人靠近工作台
                if self.rivals_on_targets(idx_robot, 9):
                    # 瞄准敌人的方向

                    # 获取距离工作台最近的敌人位置
                    dis_min, rival_loc, rival_r, theta_rival = self.get_nearst_rival2workbench(
                        idx_robot)

                    # 本机机器人指向敌人位置的向量
                    vec_robot2rival = np.array(rival_loc) - np.array(robot.loc)

                    # 向量的角度
                    rival_theta = np.arctan2(
                        vec_robot2rival[1], vec_robot2rival[0])

                    # 机器人指向敌人的角度偏移
                    delta_theta = rival_theta - robot_theta

                    # 映射到0-2pi区间
                    delta_theta = (delta_theta +
                                   math.pi) % (2 * math.pi) - math.pi

                    robot.forward(0)

                    # 原地旋转预瞄准
                    robot.rotate(delta_theta * k_r)

                    if dis_min < 4:
                        # 到达敌人位置，切换为攻击状态
                        robot.attack_status = Robot.ATTACK

            if robot.attack_status == Robot.ATTACK:
                # 攻击敌人 金钟罩

                # 获取距离工作台最近的敌人位置
                dis_min, rival_loc, rival_r, theta_rival = self.get_nearst_rival2workbench(
                    idx_robot)
                if robot.item_type == 0:
                    my_r = 0.45
                else:
                    my_r = 0.53
                if theta_rival is not None:
                    theta_rival = theta_rival + math.pi
                    offset = 0.2
                    target_loc = np.array(
                        rival_loc) - offset * np.array([np.cos(theta_rival), np.sin(theta_rival)])
                    target_vec = [target_loc[0] - robot.loc[0],
                                  target_loc[1] - robot.loc[1]]
                    target_theta = np.arctan2(
                        target_vec[1], target_vec[0])
                    robot_theta = self.robots[idx_robot].toward
                    delta_theta = target_theta - robot_theta
                    robot.forward((dis_target - (rival_r + my_r - offset)))
                    robot.rotate(delta_theta * k_r)

                if not self.rivals_on_targets(idx_robot, 6):
                    # 没有敌人了，切换为前往工作台的路上
                    robot.attack_status = Robot.MOV_TO_ATTACK

            if robot.attack_status == Robot.BCK_TO_ATTACK:
                # 回防工作台的路上
                if self.target_slow(idx_robot, target_idx, target_loc, col_flag, sb_flag, sb_safe_dis):
                    # 慢速行驶至目标
                    robot.forward(dis_target * k_f)
                else:
                    # 高速行驶至目标

                    robot.forward(9)

                robot.rotate(delta_theta * k_r)

                if self.rivals_on_targets(idx_robot, 6):
                    # 到达敌方工作台，切换为攻击状态
                    robot.attack_status = Robot.ATTACK
        else:
            # 正常的机器人

            if flag_avoid_obt:
                delta_theta = d_theta_avoid_obt

            if self.target_slow(idx_robot, target_idx, target_loc, col_flag, sb_flag, sb_safe_dis):
                # 慢速行驶至目标
                robot.forward(dis_target * k_r)
            else:
                # 高速行驶至目标
                robot.forward(9)

            if sb_safe_dis:
                # 和我方正在买卖的机器人保持安全距离
                robot.forward((d - self.WILL_CLASH_DIS-0.1) * 6)
            if self.target_slow(idx_robot, target_idx, target_loc, col_flag, sb_flag, sb_safe_dis):
                # 慢速行驶至目标
                robot.forward(dis_target * k_r)
            else:
                # 高速行驶至目标
                robot.forward(9)

            if abs(delta_theta) > math.pi / 6:
                # 角度相差较大 原地转向
                robot.forward(0)



            robot.rotate(delta_theta * k_r)

    def target_slow(self, idx_robot, target_idx, target_loc, col_flag, sb_flag, sb_safe_dis):
        # 判断目标点是否需要减速
        robot = self.robots[idx_robot]

        # 到工作台距离 用于判定是否接近目标工作台
        dis2workbench = self.dis2target(idx_robot)
        if robot.target_idx == len(robot.path) - 1 or dis2workbench < 5:
            # 到达终点或者接近终点
            if self.rivals_on_targets(idx_robot, 0.7):
                # 有敌人 撞！
                return False
            else:
                return True

        # 机器人自身指向目标点的向量
        vec_robot2target = np.array(target_loc) - np.array(robot.loc)

        if not (col_flag or (sb_flag and not sb_safe_dis)):
            # 目标点指向路径上下一个路径点的向量
            try:
                vec_target2next = np.array(
                    robot.path[target_idx + 1, :]) - np.array(target_loc)
            except:
                raise Exception(target_loc)
            # 两个向量的夹角
            theta = np.arccos(np.dot(vec_robot2target, vec_target2next) /
                              (np.linalg.norm(vec_robot2target) * np.linalg.norm(vec_target2next)))
            if theta > math.pi * 0.4:
                return True
            else:
                return False
        else:
            # 这是啥？
            return False

    def move_bck(self, idx_robot):

        # 360雷达前的版本
        robot = self.robots[idx_robot]
        k_r = 8
        flag_obt_near = self.obt_near(robot)

        # 到工作台距离 用于判定是否接近道路终点
        dis2workbench = self.dis2target(idx_robot)

        # 判定是否有临时目标点
        if self.robots[idx_robot].target_loc is None:
            # 没有临时目标点则重新规划
            self.re_path(robot)
            if flag_obt_near:
                target_loc, target_idx = self.select_target(idx_robot)
            else:
                target_idx = robot.find_temp_tar_idx()
                target_loc = robot.path[target_idx, :]
            robot.target_loc = target_loc
            robot.target_idx = target_idx
        else:
            # 有临时目标点
            dis_temp_target = np.sqrt(
                np.sum((robot.target_loc - np.array(robot.loc)) ** 2))
            if (robot.frame_wait > 0 and dis_temp_target > 0.35) or (robot.frame_wait == 0 and dis_temp_target > 2):
                # 距离大于给定值时 继续追踪
                target_loc = robot.target_loc
            else:
                self.re_path(robot)
                # 足够接近时 重新选择
                if flag_obt_near:
                    target_loc, target_idx = self.select_target(idx_robot)
                else:
                    target_idx = robot.find_temp_tar_idx()
                    target_loc = robot.path[target_idx, :]
                robot.target_loc = target_loc
                robot.target_idx = target_idx
        # 因为移动过程中可能导致阻塞而避让, 可以解除顶牛, 可能导致HUQ
        col_flag = False
        # 因为买卖而产生的避让
        sb_flag = False
        # 是否要采取保持距离的方式
        sb_safe_dis = False
        d = 100
        # 要避让的机器人序号
        idx_huq = -1
        for idx_other in range(4):
            if not idx_other == idx_robot:
                d = min(self.get_other_col_info2(
                    idx_robot, idx_other), d)
                if d < self.WILL_CLASH_DIS:
                    col_flag = True
                    idx_huq = idx_other
                    break
        robot_target = robot.target
        # 初始化一个较大值
        other_dis2workbench = self.WILL_HUQ_DIS

        if dis2workbench < self.WILL_HUQ_DIS and not col_flag and robot.status in [Robot.MOVE_TO_BUY_STATUS,
                                                                                   Robot.WAIT_TO_BUY_STATUS]:
            for idx_other in range(4):
                # 锐总说这不合适吧
                if (not idx_other == idx_robot) and self.robots[idx_other].frame_wait == 0 and robot_target == \
                        self.robots[idx_other].target:
                    # 另一个机器人到工作台的距离
                    other_dis2workbench = self.dis2target(idx_other)
                    status_other = self.robots[idx_other].status
                    if other_dis2workbench > self.WILL_HUQ_DIS:
                        continue
                    # 买的让卖的
                    if status_other in [Robot.MOVE_TO_SELL_STATUS, Robot.WAIT_TO_SELL_STATUS]:
                        sb_flag = True
                    # 同买, 近的让远的
                    elif status_other in [Robot.MOVE_TO_BUY_STATUS, Robot.WAIT_TO_BUY_STATUS]:
                        if dis2workbench > other_dis2workbench:
                            sb_flag = True
                        elif dis2workbench == other_dis2workbench and idx_robot > idx_other:
                            sb_flag = True
                    if sb_flag:
                        idx_huq = idx_other
                        break
        if sb_flag and dis2workbench > other_dis2workbench:
            sb_safe_dis = True

        if col_flag or (sb_flag and not sb_safe_dis):
            priority_idx = -1
            if col_flag:
                status_huq = self.robots[idx_huq].status
                huq_dis2workbench = self.dis2target(idx_huq)
                if robot_target == self.robots[idx_huq].target:
                    # 我买对方卖
                    if robot.status in [Robot.MOVE_TO_BUY_STATUS, Robot.WAIT_TO_BUY_STATUS] and status_huq in [
                            Robot.MOVE_TO_SELL_STATUS, Robot.WAIT_TO_SELL_STATUS]:
                        priority_idx = idx_robot
                    # 我卖对方买
                    elif robot.status in [Robot.MOVE_TO_SELL_STATUS, Robot.WAIT_TO_SELL_STATUS] and status_huq in [
                            Robot.MOVE_TO_BUY_STATUS, Robot.WAIT_TO_BUY_STATUS]:
                        priority_idx = idx_huq
                    # 同买同卖
                    else:
                        if dis2workbench > huq_dis2workbench:
                            priority_idx = idx_robot
                        elif dis2workbench < huq_dis2workbench:
                            priority_idx = idx_huq
            else:
                priority_idx = idx_robot
            self.re_path(robot)
            self.re_path(self.robots[idx_huq])
            avoid_idx, avoid_path = self.process_deadlock(
                idx_robot, idx_huq, priority_idx)
            # sys.stderr.write(f"avoid_idx: {avoid_idx}\n")
            if avoid_idx == -1:
                # sys.stderr.write(
                #     f"REVERSE idx_robot: {idx_robot}\n")
                # 如果出现可能有坑 一个机器人堵了两个机器人
                sb_safe_dis = True
                pass
            elif avoid_idx == idx_robot:
                # sys.stderr.write(f"idx_robot{idx_robot}, robot.item{robot.item_type}, avoid_path{avoid_path}\n")
                self.robots[idx_robot].set_path(avoid_path)
                self.robots[idx_robot].frame_wait = self.AVOID_FRAME_WAIT
                # sys.stderr.write(f"idx_robot: {idx_robot}\n")
                flag_obt_near = self.obt_near(robot)
                if flag_obt_near:
                    target_loc, target_idx = self.select_target(
                        idx_robot)
                else:
                    target_idx = robot.find_temp_tar_idx()
                    target_loc = robot.path[target_idx, :]
                robot.target_loc = target_loc
                robot.target_idx = target_idx

        # 根据给定目标点修正
        target_vec = [target_loc[0] - robot.loc[0],
                      target_loc[1] - robot.loc[1]]
        dis_target = np.sqrt(np.dot(target_vec, target_vec))

        target_theta = np.arctan2(
            target_vec[1], target_vec[0])

        robot_theta = self.robots[idx_robot].toward
        delta_theta = target_theta - robot_theta

        # 不确定用处大不大，暂时保留
        # for idx_other in range(4):
        #     if not idx_other == idx_robot:
        #         if self.direct_colli(idx_robot, idx_other, thr_dis=6):
        #             delta_theta -= math.pi / 5
        #             break

        if dis2workbench < 3 and robot.status == Robot.BLOCK_OTRHER and 0:
            print("forward", idx_robot, dis2workbench * 3)
            self.robots[idx_robot].rotate(delta_theta * k_r)
            # sys.stderr.write("干死他\n")
            pass
        else:
            delta_theta = (delta_theta +
                           math.pi) % (2 * math.pi) - math.pi
            if sb_safe_dis:
                # 保持安全车距等待买卖
                print("forward", idx_robot, (d - self.WILL_CLASH_DIS-0.1) * 6)
            # elif abs(delta_theta) > math.pi * 5 / 6 and dis_target < 2:
            #     # 角度相差太大倒车走
            #     print("forward", idx_robot, -2)
            #     # delta_theta += math.pi
            elif abs(delta_theta) > math.pi / 6:
                # 角度相差较大 原地转向
                print("forward", idx_robot, 0)
            elif dis2workbench < 1.5:

                print("forward", idx_robot, dis2workbench * 6)

            else:
                print("forward", idx_robot, (dis_target) * 10)

            delta_theta = (delta_theta +
                           math.pi) % (2 * math.pi) - math.pi

            self.robots[idx_robot].rotate(delta_theta * k_r)

    def get_time_rate(self, frame_sell: float) -> float:
        # 计算时间损失
        if frame_sell >= 9000:
            return 0.8
        sqrt_num = math.sqrt(1 - (1 - frame_sell / 9000) ** 2)
        return (1 - sqrt_num) * 0.2 + 0.8

    def choise(self, frame_id: int, robot: Robot) -> bool:
        # 进行一次决策
        max_radio = 0  # 记录最优性价比
        for idx_workbench_to_buy in robot.target_workbench_list:
            if idx_workbench_to_buy in self.can_not_reach_workbenchs:
                continue
            workbench_buy = self.workbenchs[idx_workbench_to_buy]
            if workbench_buy.product_time == -1 and workbench_buy.product_status == 0 or workbench_buy.product_pro == 1:  # 被预定了,后序考虑优化
                continue
            # 生产所需时间，如果已有商品则为0
            frame_wait_buy = workbench_buy.product_time if workbench_buy.product_status == 0 else 0
            # if frame_wait_buy > self.MAX_WAIT: 由于移动时间更长了，所以直接生产等待时间比较不合理
            #     continue
            frame_move_to_buy = len(self.m_map.get_path(
                robot.loc, idx_workbench_to_buy, self.blue_flag)) * self.MOVE_SPEED
            if frame_wait_buy - frame_move_to_buy > self.MAX_WAIT:  # 等待时间超出移动时间的部分才有效
                continue
            # 需要这个产品的工作台
            for idx_workbench_to_sell in workbench_buy.target_workbench_list:
                if idx_workbench_to_sell in self.can_not_reach_workbenchs:
                    continue
                workbench_sell = self.workbenchs[idx_workbench_to_sell]
                if workbench_sell.check_material_pro(workbench_buy.typeID):
                    # sys.stderr.write(f"idx_workbench_to_sell:{idx_workbench_to_sell}\n")
                    continue
                # 格子里有这个原料
                # 判断是不是8或9 不是8或9 且这个原料格子已经被占用的情况, 生产完了并不一定能继续生产
                frame_wait_sell = 0
                if workbench_sell.check_material(workbench_buy.typeID):
                    continue
                    # 阻塞或者材料格没满
                    if workbench_sell.product_time in [-1, 0] or not workbench_sell.check_materials_full():
                        continue
                    elif workbench_sell.product_status == 1:  # 到这里说明材料格和产品格都满了不会再消耗原料格了
                        continue
                    else:
                        frame_wait_sell = workbench_sell.product_time
                # frame_move_to_buy, frame_move_to_sell= self.get_time_rww(idx_robot, idx_workstand, idx_worksand_to_sell)
                frame_move_to_sell = len(self.m_map.get_path(workbench_buy.loc, idx_workbench_to_sell, self.blue_flag,
                                                             True)) * self.MOVE_SPEED
                frame_buy = max(frame_move_to_buy, frame_wait_buy)  # 购买时间
                frame_sell = max(frame_move_to_sell,
                                 frame_wait_sell - frame_buy)  # 出售时间
                total_frame = frame_buy + frame_sell  # 总时间
                if total_frame * self.CONSERVATIVE + frame_id > self.TOTAL_FRAME:  # 完成这套动作就超时了
                    continue
                time_rate = self.get_time_rate(
                    frame_move_to_sell)  # 时间损耗
                # sell_weight = self.SELL_WEIGHT**workbench_sell.get_materials_num # 已经占用的格子越多优先级越高
                sell_weight = self.SELL_WEIGHT if workbench_sell.material else 1  # 已经占用格子的优先级越高
                sell_debuff = self.SELL_DEBUFF if workbench_sell.typeID == 9 and workbench_sell.typeID != 7 else 1
                strave_wight = self.STARVE_WEIGHT if self.starve.get(
                    workbench_sell.typeID, 0) > 0 else 1  # 鼓励生产急需商品
                radio = (workbench_buy.sell_price * time_rate -
                         workbench_buy.buy_price) / total_frame * sell_weight * sell_debuff * strave_wight
                # sys.stderr.write(f"radio:{radio} strave_wight{strave_wight}\n")
                if radio > max_radio:
                    max_radio = radio
                    robot.set_plan(idx_workbench_to_buy, idx_workbench_to_sell)
                    robot.frame_reman_buy = frame_buy
                    robot.frame_reman_sell = frame_sell
        if max_radio > 0:  # 开始执行计划
            return True
        return False

    def re_path(self, robot: Robot):
        '''
        为机器人重新规划路劲
        '''
        # 判断周围是否有障碍
        loc = robot.loc
        if robot.status in [Robot.MOVE_TO_BUY_STATUS, Robot.WAIT_TO_BUY_STATUS]:
            # 重新规划路径
            robot.set_path(self.m_map.get_float_path(
                loc, robot.get_buy(), self.blue_flag))
            robot.status = Robot.MOVE_TO_BUY_STATUS
        elif robot.status in [Robot.MOVE_TO_SELL_STATUS, Robot.WAIT_TO_SELL_STATUS]:
            robot.set_path(self.m_map.get_float_path(
                loc, robot.get_sell(), self.blue_flag, True))
            robot.status = Robot.MOVE_TO_SELL_STATUS
        # 这一状态最好老老实实追点, 少用re_path
        elif robot.status == Robot.AVOID_CLASH:
            other_locs = [self.robots[idx].loc for idx in range(
                len(self.robots)) if idx != robot.ID]
            if self.rival_list:
                other_locs.extend(list(zip(*self.rival_list))[0])
            target_loc = self.workbenchs[robot.target].loc
            new_way = self.m_map.get_a_new_way(
                robot.loc, target_loc, other_locs, robot.item_type > 0)
            if new_way:
                robot.set_path(new_way)
            else:
                # 再次a_star失败直接改机器人状态，repath
                if robot.item_type > 0:
                    robot.status = Robot.MOVE_TO_SELL_STATUS
                else:
                    robot.status = Robot.MOVE_TO_BUY_STATUS
                self.re_path(robot)
        elif robot.status == Robot.BLOCK_OTRHER:
            robot.set_path(self.m_map.get_float_path(
                loc, robot.target, not self.blue_flag, robot.item_type > 0))

    def process_deadlock(self, robot1_idx, robot2_idx, priority_idx=-1, safe_dis: float = None):
        '''
        处理死锁
        robot1_idx, robot2_idx 相互死锁的两个机器人ID
        priority_idx 优先要避让的机器人ID
        返回 避让的id及路径
        如果都无路可退 id为-1 建议让两个直接倒车
        '''
        r1, r2 = min(robot1_idx, robot2_idx), max(robot1_idx, robot2_idx)
        if (r1, r2) in self.tmp_avoid:
            return self.tmp_avoid[(r1, r2)]
        robot1, robot2 = self.robots[robot1_idx], self.robots[robot2_idx]
        other_locs = []  # 记录另外两个机器人的坐标
        for idx, robot in enumerate(self.robots):
            if idx not in [robot1_idx, robot2_idx]:
                other_locs.append(robot.loc)
        # 机器人1的退避路径
        avoid_path1 = self.m_map.get_avoid_path(
            robot1.loc, robot2.path, other_locs + [robot2.loc], robot1.item_type != 0, safe_dis)
        # 机器人2的退避路径
        avoid_path2 = self.m_map.get_avoid_path(
            robot2.loc, robot1.path, other_locs + [robot1.loc], robot2.item_type != 0, safe_dis)
        # 记录一下要避让的机器人
        avoid_robot = -1
        avoid_path = []
        if priority_idx == robot1_idx and avoid_path1:
            return robot1_idx, avoid_path1
        elif priority_idx == robot2_idx and avoid_path2:
            return robot2_idx, avoid_path2
        if not avoid_path1 and avoid_path2:
            avoid_robot = robot2_idx
            avoid_path = avoid_path2
        elif not avoid_path2 and avoid_path1:
            avoid_robot = robot1_idx
            avoid_path = avoid_path1
        elif avoid_path1 and avoid_path2:
            # 1 要走得路多，2让
            if len(avoid_path1) > len(avoid_path2):
                avoid_robot = robot2_idx
                avoid_path = avoid_path2
            elif len(avoid_path1) < len(avoid_path2):
                avoid_robot = robot1_idx
                avoid_path = avoid_path1
            else:
                robot1_frame_reman = robot1.get_frame_reman()
                robot2_frame_reman = robot2.get_frame_reman()
                if robot1_frame_reman < robot2_frame_reman or robot1_frame_reman == robot2_frame_reman and robot1_idx < robot2_idx:
                    avoid_robot = robot2_idx
                    avoid_path = avoid_path2
                else:
                    avoid_robot = robot1_idx
                    avoid_path = avoid_path1
        # sys.stderr.write(f"avoid_robot: {avoid_robot} avoid_path{avoid_path}\n")
        self.tmp_avoid[(r1, r2)] = (avoid_robot, avoid_path)
        return avoid_robot, avoid_path

    def process_long_deadlock(self, frame_id):
        '''
        处理长时间死锁，如果死锁处理失败会出现这种情况
        '''
        # 在这里执行冲突检测和化解并记得记录上一个机器人的状态
        # 如果冲突无法化解，让每个机器人都倒一下车
        self.detect_deadlock(frame_id)
        locs = [robot.loc for robot in self.robots]
        for idx, robot in enumerate(self.robots):
            if not robot.is_stuck or robot.status in [Robot.BLOCK_OTRHER, Robot.FREE_STATUS]:
                continue
            other_locs = locs[:]
            other_locs.pop(idx)
            if self.rival_list:
                other_locs.extend(list(zip(*self.rival_list))[0])
            # sys.stderr.write(f"other_locs:{other_locs}\n")
            target_loc = self.workbenchs[robot.target].loc
            new_way = self.m_map.get_a_new_way(
                robot.loc, target_loc, other_locs, robot.item_type > 0)
            if new_way:  # 切换机器人状态
                robot.set_path(new_way)
                robot.status = robot.AVOID_CLASH
                robot.is_stuck = False
                if robot.item_type > 0:
                    robot.frame_reman_sell = len(new_way)*self.MOVE_SPEED
                else:
                    robot.frame_reman_buy = len(new_way)*self.MOVE_SPEED

    def have_rival_robot(self, robot, min_dis=1.2):
        '''
        判断己方机器人附近是否有敌方机器人
        '''
        r_loc = np.array(robot.loc)
        for i_loc in self.rival_list:
            # sys.stderr.write(f"r_loc:{r_loc}, i_loc:{i_loc[0]}, dis:{tools.np_norm(r_loc, i_loc[0])}\n")
            if tools.np_norm(r_loc, i_loc[0]) < min_dis:
                return True
        return False

    def detect_rival(self):
        # 对手机器人坐标 半径
        rival_list = []

        # 遍历己方机器人
        for robot in self.robots:
            self.detect_rival_item(robot, 20, rival_list)

        self.rival_list = rival_list

    def rivals_on_targets(self, idx_robot, thr_dis):
        # 检测是否有对手在目标点
        robot = self.robots[idx_robot]
        idx_workbench = robot.target

        # 目标工作台坐标
        target_loc = self.workbenchs[idx_workbench].loc if self.robots[
            idx_robot].status != Robot.BLOCK_OTRHER else self.rival_workbenchs[idx_workbench].loc
        for rival in self.rival_list:
            # 取出敌人列表 的 坐标
            loc_rival, _ = rival
            if np.sqrt((loc_rival[0] - target_loc[0]) ** 2 + (loc_rival[1] - target_loc[1]) ** 2) < thr_dis:
                # 距离过近
                # sys.stderr.write(f'{idx_workbench}')
                return True
        # 不存在距离过近的敌人
        return False

    def get_nearst_rival2workbench(self, idx_robot):
        # 检测是否有对手在目标点
        robot = self.robots[idx_robot]
        idx_workbench = robot.target

        # 目标工作台坐标
        target_loc = self.workbenchs[idx_workbench].loc if self.robots[idx_robot].status != Robot.BLOCK_OTRHER else \
            self.rival_workbenchs[idx_workbench].loc

        dis_min = 1000
        loc_rival_min = None
        theta_min = None
        r_min = None
        for rival in self.rival_list:
            # 取出敌人列表 的 坐标
            loc_rival, r = rival
            vec = np.array(loc_rival) - np.array(target_loc)
            dis = np.sqrt(np.dot(vec, vec))
            theta = np.arctan2(vec[1], vec[0])
            if dis < dis_min:
                dis_min = dis
                loc_rival_min = loc_rival
                theta_min = theta
                r_min = r
        # 不存在距离过近的敌人
        return dis_min, loc_rival_min, r_min, theta_min

    def control(self, frame_id: int, money: int):
        # 三个问题 1 A_star算路时间太久 2 AVOID状态下尽量避免re_path 3 使这个的触发条件更严格一些
        # self.process_long_deadlock(frame_id)
        self.detect_rival()
        print(frame_id)
        sell_out_list = []  # 等待处理预售的机器人列表
        idx_robot = 0
        while idx_robot < 4:
            robot = self.robots[idx_robot]
            robot_status = robot.status
            if robot_status == Robot.FREE_STATUS:
                # 判断是否出现了因跳帧导致的出售失败, 不太对, 预售预购的处理
                if robot.item_type != 0:
                    robot.status = Robot.MOVE_TO_SELL_STATUS
                    # 重新预售，有极低概率出问题，重复预售
                    workbench_sell = self.workbenchs[robot.get_sell()]
                    workbench_sell.pro_sell(robot.item_type)
                    continue
                # 【空闲】执行调度策略
                if self.choise(frame_id, robot):
                    # 设置机器人移动目标
                    idx_workbench_to_buy = robot.get_buy()
                    idx_workbench_to_sell = robot.get_sell()
                    robot.target = idx_workbench_to_buy
                    robot.set_path(self.m_map.get_float_path(
                        robot.loc, idx_workbench_to_buy, self.blue_flag))
                    # 预定工作台
                    self.workbenchs[idx_workbench_to_buy].pro_buy()
                    self.workbenchs[idx_workbench_to_sell].pro_sell(
                        self.workbenchs[idx_workbench_to_buy].typeID)
                    robot.status = Robot.MOVE_TO_BUY_STATUS
                    robot.free_frames = 0
                    continue
                else:
                    robot.free_frames += 1
                    if self.TOTAL_FRAME-frame_id < 1000:
                        self.attack_one(robot, True)
                    elif robot.free_frames > self.MAX_FREE:
                        self.attack_one(robot)
                    else:
                        robot.forward(9)
                        robot.rotate(4)

            elif robot_status == Robot.MOVE_TO_BUY_STATUS:
                # 【购买途中】
                self.move(idx_robot)
                # 判断距离是否够近

                # 判定是否进入交互范围
                if robot.workbench_ID == robot.target:
                    # 记录任务分配时的时间 距离 角度相差
                    robot.status = Robot.WAIT_TO_BUY_STATUS  # 切换为 【等待购买】
                    continue
            elif robot_status == Robot.WAIT_TO_BUY_STATUS:
                # 【等待购买】
                self.move(idx_robot)
                idx_workbench_to_buy = robot.get_buy()
                product_status = self.workbenchs[idx_workbench_to_buy].product_status
                # self.pre_rotate(idx_robot, next_walkstand)
                # 如果在等待，提前转向
                if product_status == 1:  # 这里判定是否生产完成可以购买
                    # 可以购买
                    if robot.buy():  # 防止购买失败
                        # 取消预售
                        self.workbenchs[idx_workbench_to_buy].pro_buy(False)
                        idx_workbench_to_sell = robot.get_sell()
                        robot.target = idx_workbench_to_sell  # 更新目标到卖出地点
                        # 取消拉黑
                        if idx_workbench_to_buy in self.black_workbenchs:
                            self.black_workbenchs.pop(idx_workbench_to_buy)
                        if robot.block_model:
                            robot.status = Robot.BLOCK_OTRHER  # 切换为崽人
                            self.rival_workbenchs[robot.target].attack_value = Workbench.MAX_ATTCK_VALUE
                            # 攻击状态设一下
                            robot.attack_status = Robot.MOV_TO_ATTACK
                        else:
                            robot.status = Robot.MOVE_TO_SELL_STATUS  # 切换为 【出售途中】
                        robot.set_path(self.m_map.get_float_path(
                            robot.loc, idx_workbench_to_sell, self.blue_flag, True))
                        # continue
                    else:
                        robot.status = Robot.MOVE_TO_BUY_STATUS
                        robot.set_path(self.m_map.get_float_path(
                            robot.loc, robot.get_buy(), self.blue_flag))
                        continue
            elif robot_status == Robot.MOVE_TO_SELL_STATUS:
                # 【出售途中】
                # 判断是否出现了因跳帧导致的购买失败，有极低的概率重复预购
                if robot.item_type == 0:
                    robot.status = Robot.MOVE_TO_BUY_STATUS
                    self.workbenchs[robot.get_buy()].pro_buy()
                    continue
                # 移动
                self.move(idx_robot)
                # 判断距离是否够近

                # 判定是否进入交互范围
                if robot.workbench_ID == robot.target:
                    robot.status = Robot.WAIT_TO_SELL_STATUS  # 切换为 【等待出售】
                    # logging.debug(f"{idx_robot}->ready to sell")
                    # 记录任务分配时的时间 距离 角度相差
                    # 记录任务分配时的时间 距离 角度相差
                    continue

            elif robot_status == Robot.WAIT_TO_SELL_STATUS:
                # 【等待出售】
                self.move(idx_robot)
                idx_workbench_to_sell = robot.get_sell()
                workbench_sell = self.workbenchs[idx_workbench_to_sell]
                # 如果在等待，提前转向
                # raise Exception(F"{material},{material_type}")
                # 这里判定是否生产完成可以出售 不是真的1
                if not Workbench.WORKSTAND_OUT[workbench_sell.typeID] or not workbench_sell.check_material(
                        robot.item_type):
                    # 可以出售
                    if robot.sell():  # 防止出售失败
                        # 维护一下急需列表
                        # 7且不等于0 说明有材料且没满
                        if workbench_sell.typeID == 7 and workbench_sell.material != 0:
                            # sys.stderr.write(f"material: {workbench_sell.material}\n")
                            if workbench_sell.material in Workbench.WORKSTAND_STARVE:  # 就差一个了
                                self.starve[robot.item_type] -= 1
                            else:
                                self.starve[Workbench.WORKSTAND_STARVE[workbench_sell.material + (
                                    1 << robot.item_type)]] += 1
                            # sys.stderr.write(f"material: {self.starve}\n")
                        # 取消拉黑
                        if idx_workbench_to_sell in self.black_workbenchs:
                            self.black_workbenchs.pop(idx_workbench_to_sell)
                        # 取消预定
                        sell_out_list.append(idx_robot)
                        robot.status = Robot.FREE_STATUS
                        # 这个机器人不能再决策了, 不然目标更新，状态会被清错
                        # continue
                    else:
                        robot.status = Robot.MOVE_TO_SELL_STATUS  # 购买失败说明位置不对，切换为 【出售途中】
                        robot.set_path(self.m_map.get_float_path(
                            robot.loc, idx_workbench_to_sell, self.blue_flag, True))
                        continue
            elif robot_status == Robot.AVOID_CLASH:
                # 解除此状态
                if self.dis2target(idx_robot) < 1:
                    if robot.item_type > 0:
                        robot.status = Robot.MOVE_TO_SELL_STATUS
                    else:
                        robot.status = Robot.MOVE_TO_BUY_STATUS
                    self.set_robot_state_undeadlock(idx_robot, frame_id)
                    self.re_path(robot)
                else:
                    self.move(idx_robot)
            elif robot_status == Robot.BLOCK_OTRHER:
                # 这里可以后续加点长时间没人的处理策略
                self.move(idx_robot)
                r_w = self.rival_workbenchs[robot.target]
                if self.have_rival_robot(robot):
                    r_w.attack_value += Workbench.ATTCK_VALUE
                    if self.rival_workbenchs[robot.target].attack_value > Workbench.MAX_ATTCK_VALUE:
                        r_w.attack_value = Workbench.MAX_ATTCK_VALUE
                # 在工作台附近才减分
                elif robot.block_type == Robot.BLOCK_TYPE_SENTINEL and self.dis2target(idx_robot) < 1.5:
                    r_w.attack_value -= 1
                    if r_w.attack_value == 0:
                        self.attack_one(robot)
                elif robot.block_type == Robot.BLOCK_TYPE_PATORL and self.dis2target(idx_robot) < 0.5:
                    sys.stderr.write(f'robotID:{robot.ID}, block_workbench_index{robot.block_workbench_index}\n')
                    for _ in range(len(self.other_workbenchs_list)):
                        robot.block_workbench_index = (
                            robot.block_workbench_index + 1) % len(self.other_workbenchs_list)
                        if self.get_attack_path(robot, self.other_workbenchs_list[robot.block_workbench_index]):
                            robot.target = robot.get_buy()
                            robot.set_path(self.m_map.get_float_path(
                                robot.loc, robot.target, not self.blue_flag, robot.item_type > 0))
                            break
                # sys.stderr.write(f'robotID:{robot.ID}, attack_value{r_w.attack_value}\n')

            # 根据状态更新预估时间
            self.tmp_avoid.clear()
            robot.update_frame_reman()
            # 严重超时, 重新规划, 对手里有东西的稍微宽容一点
            if robot.status in [Robot.MOVE_TO_BUY_STATUS, Robot.MOVE_TO_SELL_STATUS]:
                move_reman = self.MOVE_SPEED * \
                    len(self.m_map.get_path(robot.loc, robot.target,
                        self.blue_flag, robot.item_type > 0))  # 预估到达时间
                # and robot.is_stuck:
                if robot.get_frame_reman() - move_reman < self.MAX_TIME_OUT * (1 if robot.item_type == 0 else 1.5):
                    self.re_mession(robot, self.AVOID_FRAME_WAIT)
            idx_robot += 1
        for idx_robot in sell_out_list:
            robot = self.robots[idx_robot]
            workbench_sell = self.workbenchs[robot.get_sell()]
            workbench_sell.pro_sell(robot.item_type, False)
        # 处理一下工作台
        recorve_w_idx = []  # 记录可以恢复的工作台
        for w_idx, frames in self.can_not_reach_workbenchs.items():
            self.can_not_reach_workbenchs[w_idx] -= 1
            if frames <= 0:
                recorve_w_idx.append(w_idx)
        # 取消不可达状态
        for w_idx in recorve_w_idx:
            self.can_not_reach_workbenchs.pop(w_idx)
        # sys.stderr.write(f"can_not_reach_workbenchs:{self.can_not_reach_workbenchs}\n")
