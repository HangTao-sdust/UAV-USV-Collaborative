import random
import numpy as np
from multiagent.core import World, Agent, Landmark
from multiagent.scenario import BaseScenario
import math
import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns
class Scenario(BaseScenario):

    def __init__(self):
        # 创建保存图片的文件夹
        self.save_dir = 'output_images'
        self.uncertainty_history = []  # 用于记录全局不确定性变化
        os.makedirs(self.save_dir, exist_ok=True)

    def calculate_entropy(self, p_value):
        """计算单个概率值的信息熵"""
        if p_value == 0 or p_value == 1:
            return 0
        return -p_value * math.log2(p_value) - (1 - p_value) * math.log2(1 - p_value)

    def calculate_grid_uncertainty(self, world, x, y):
        """计算单个网格的不确定性"""
        p = world.p[x][y]
        return self.calculate_entropy(p)

    def calculate_global_uncertainty(self, world):
        """计算全局不确定性（所有网格的平均熵）"""
        total_entropy = 0
        count = 0
        for i in range(1, world.grid_num + 1):
            for j in range(1, world.grid_num + 1):
                total_entropy += self.calculate_entropy(world.p[i][j])
                count += 1
        return total_entropy / count if count > 0 else 0



    # def plot_and_save_probability_map(self, world, step):
    #     """
    #     绘制并保存概率图，并标记网格坐标
    #     """
    #     plt.figure(figsize=(10, 10))
    #     plt.imshow(world.p, cmap='viridis', origin='lower', vmin=0, vmax=1)
    #     plt.colorbar(label='Probability')
    #     #plt.title(f'Probability Map at Step {step}')
    #     #plt.xlabel('X Grid')
    #     #plt.ylabel('Y Grid')
    #
    #     # 添加网格线
    #     plt.grid(True, which='both', color='black', linestyle='--', linewidth=0.5)
    #
    #     # 标记网格坐标（1-20）
    #     plt.xticks(np.arange(0.5, world.grid_num + 0.5, 1), np.arange(1, world.grid_num + 1, 1))
    #     plt.yticks(np.arange(0.5, world.grid_num + 0.5, 1), np.arange(1, world.grid_num + 1, 1))
    #
    #     # 保存为 .png 格式
    #     plt.savefig(os.path.join(self.save_dir, f'probability_map_step_{step}.png'), format='png')
    #     plt.close()
    #
    # def save_probability_values(self, world, step):
    #     """
    #     保存概率数值到文件
    #     :param world: World 对象
    #     :param step: 当前步骤编号
    #     """
    #     with open(os.path.join(self.save_dir, f'probability_values_step_{step}.txt'), 'w') as f:
    #         for i in range(1, world.grid_num + 1):
    #             for j in range(1, world.grid_num + 1):
    #                 f.write(f"Grid ({i}, {j}): {world.p[i][j]:.4f}\n")

    def update_Nmap(self,agent,world):

        x_grid,y_grid = agent.cur_grid
        target_x_grid = math.ceil(world.landmarks[0].state.p_pos[0] / world.map_int)
        target_y_grid = math.ceil(world.landmarks[0].state.p_pos[1] / world.map_int)


        '''
        p:目标和无人机在同一个网格 并且无人机成功检测到目标的概率
        1-p:目标和无人机在同一个网格 但是无人机没有检测到目标的概率
        q:目标和无人机不在同一个网格 无人机检测到目标的概率
        '''
        # d是检测概率 f是虚报概率 原文0.8 0.2

        if target_x_grid == x_grid and target_y_grid == y_grid:  # 如果无人机和目标在同一个网格内
            rand = random.random()
            if rand < agent.d:  # p的概率无人机遇到了目标并且检测到目标
                world.p[x_grid][y_grid] = agent.d * world.p[x_grid][y_grid] / (agent.d * world.p[x_grid][y_grid] + agent.f * (1 - world.p[x_grid][y_grid]))
                agent.pd_time[x_grid][y_grid] += 1
            else:  # 1-p的概率无人机遇到了目标但是未检测到目标
                world.p[x_grid][y_grid] = (1 - agent.d) * world.p[x_grid][y_grid] / ((1 - agent.d) * world.p[x_grid][y_grid] + (1 - agent.f) * (1 - world.p[x_grid][y_grid]))
                agent.nd_time[x_grid][y_grid] += 1
        else:
            rand = random.random()
            if rand < agent.f:  # q的概率无人机未遇到目标但是虚报了
                world.p[x_grid][y_grid] = agent.d * world.p[x_grid][y_grid] / (agent.d * world.p[x_grid][y_grid] + agent.f * (1 - world.p[x_grid][y_grid]))
                agent.pd_time[x_grid][y_grid] += 1
            else:  # 1-q的概率无人机未遇到目标并且不会虚报
                world.p[x_grid][y_grid] = (1 - agent.d) * world.p[x_grid][y_grid] / ((1 - agent.d) * world.p[x_grid][y_grid] + (1 - agent.f) * (1 - world.p[x_grid][y_grid]))
                agent.nd_time[x_grid][y_grid] += 1
        # 新增：更新全局不确定性记录
        current_uncertainty = self.calculate_global_uncertainty(world)
        self.uncertainty_history.append(current_uncertainty)
        print(current_uncertainty)

    def share_Qmap(self, agent, world):

        x_grid, y_grid = agent.cur_grid

        world.pd[x_grid][y_grid] = max(world.pd[x_grid][y_grid], agent.pd_time[x_grid][y_grid])  # 更新positive detection times
        world.nd[x_grid][y_grid] = max(world.nd[x_grid][y_grid], agent.nd_time[x_grid][y_grid])  # 更新negative detection times

        # 更新智能体的N表
        agent.pd_time[x_grid][y_grid] = world.pd[x_grid][y_grid]
        agent.nd_time[x_grid][y_grid] = world.nd[x_grid][y_grid]
        #计算置信度
        world.Qmap[x_grid][y_grid] = world.pd[x_grid][y_grid] * math.log(agent.f / agent.d) + world.nd[x_grid][y_grid] * math.log((1 - agent.f) / (1 - agent.d))



    def make_world(self):
        world = World()
        # 保存 Scenario 实例的引用
        world.scenario = self
        # set any world properties first
        world.dim_c = 2
        num_UAVs = 3 #5、6、7
        num_landmarks = 1
        # add agents
        world.agents = [Agent() for i in range(num_UAVs)]

        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
            agent.collide = False
            agent.silent = True
            agent.type = 'UAV'
            agent.size = 0.9

        world.landmarks = [Landmark() for i in range(num_landmarks)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = 'landmark %d' % i
            landmark.collide = False
            landmark.movable = False
            landmark.size = 0.9
            # make initial conditions
        self.reset_world(world)
        return world



    def reset_world(self, world):

        self.step_count = 0  # 重置步骤计数器
        world.pd = [[0 for _ in range(world.grid_num + 1)] for _ in range(world.grid_num + 1)]
        world.nd = [[0 for _ in range(world.grid_num + 1)] for _ in range(world.grid_num + 1)]

        self.uncertainty_history = []

        cnt = 0
        for i in range(len(world.st_map[0])):
            for st in world.st_map[i]:
                if st > 0:
                    cnt += 1
        #print(cnt)
        world.st_map = [[0 for _ in range(world.grid_num + 1)] for _ in range(world.grid_num + 1)]
        world.Qmap = [[0 for _ in range(world.grid_num + 1)] for _ in range(world.grid_num + 1)]
        world.p = [[0.5 for _ in range(world.grid_num + 1)] for _ in range(world.grid_num + 1)]
        #print(world.p)


        for agent in world.agents:
            agent.pd_time = [[0 for _ in range(agent.grid_num + 1)] for _ in range(agent.grid_num + 1)]
            agent.nd_time = [[0 for _ in range(agent.grid_num + 1)] for _ in range(agent.grid_num + 1)]
            agent.collision = False
        for i, agent in enumerate(world.agents):
            c_x = random.choice(range(1, world.boundary[0] * 2, 2))
            c_y = random.choice(range(1, world.boundary[1] * 2, 2))
            agent.st_reward = [[0 for _ in range(agent.grid_num + 1)] for _ in range(agent.grid_num + 1)]

            agent.state.p_pos = [c_x * world.map_int * 0.5, c_y * world.map_int * 0.5]
            x, y = agent.state.p_pos
            x_grid = math.ceil(x / world.map_int)
            y_grid = math.ceil(y / world.map_int)
            agent.cur_grid = [x_grid, y_grid]
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)

            agent.color = np.array([0.35, 0.35, 0.85])  # 设置 UAV 的颜色
        # Random properties for landmarks
        for landmark in world.landmarks:
            landmark.state.p_pos = np.random.uniform(0, world.boundary[0], world.dim_p)
            landmark.state.p_vel = np.zeros(world.dim_p)
            landmark.color = np.array([0.35, 0.35, 0.35])  # 设置地标的颜色


    def UAVs(self,world):
        return [agent for agent in world.agents if agent.type == 'UAV']

    def reward(self, agent, world):

        reward = self.UAV_reward(agent,world)

        return reward

    def UAV_reward(self,agent, world):

        x, y = agent.cur_grid
        #更新概率图和置信度
        self.update_Nmap(agent,world)
        self.share_Qmap(agent,world)

        #1.某坐标概率图概率大于阈值的奖励
        Reward_Prob = 0
        if world.p[x][y] > world.p_threshold and agent.st_reward[x][y] == 0:
            Reward_Prob = 1
            agent.st_reward[x][y] = 1


        #2.探索奖励
        Reward_Search =  1 / (world.st_map[x][y] + 1)
        world.st_map[x][y] += 1


        #3.碰撞奖励
        Reward_Collision = 0
        for other in world.agents:
            if agent.name == other.name: continue

            other_x,other_y = agent.cur_grid
            if x == other_x and y == other_y:
                Reward_Collision -= 10

        alpha = 0.6
        beta = 0.2
        gamma = 0.2
        total_reward = alpha * Reward_Search + beta * Reward_Prob + gamma * Reward_Collision
        #更新概率图

        return total_reward


    def observation(self, agent, world):     # 观测

        directions = [(0, -1), (-1, 0), (0, 1), (1, 0)]
        obs = []
        for direction in directions:
            x_grid = agent.cur_grid[0] + direction[0]
            y_grid = agent.cur_grid[1] + direction[1]

            if(x_grid <= 0 or x_grid > world.grid_num or y_grid <= 0 or y_grid > world.grid_num):
                obs.append(-1)
            else: obs.append(world.st_map[x_grid][y_grid])

        obs.append(world.st_map[agent.cur_grid[0]][agent.cur_grid[1]])

        min_x = min_y = 0
        min_delta = 100
        for other in world.agents:
            if agent.name == other.name:
                continue
            delta_x = agent.cur_grid[0] - other.cur_grid[0]
            delta_y = agent.cur_grid[1] - other.cur_grid[1]
            if(math.hypot(delta_x,delta_y) < min_delta):
                min_delta = math.hypot(delta_x,delta_y)
                min_x = agent.cur_grid[0] - other.cur_grid[0]
                min_y = agent.cur_grid[1] - other.cur_grid[1]
        obs.append(min_x)
        obs.append(min_y)
        return np.concatenate([obs])




