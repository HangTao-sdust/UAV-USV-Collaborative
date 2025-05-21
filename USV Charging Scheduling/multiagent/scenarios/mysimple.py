import numpy as np
from multiagent.core import World, Agent, Landmark
from multiagent.scenario import BaseScenario
import math

class Scenario(BaseScenario):

    def make_world(self):
        world = World()
        # set any world properties first
        world.dim_c = 2
        num_USVs = 3
        num_UAVs = 5
        world.num_agents = num_USVs + num_UAVs
        num_landmarks = 1
        # add agents
        world.agents = [Agent() for i in range(world.num_agents)]

        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
            agent.collide = False
            agent.silent = True
            agent.accel = 2
            agent.max_speed = 2
            # 根据代理的索引 i 判断代理类型并设置特定属性
            if i < num_USVs:  # 如果代理索引小于 USVs 数量，则为 USV
                agent.type = 'USV'
                agent.size = 0.05
                agent.charge_rate = 50  # 设置 USV 的充电速度
                agent.energy = 0
            else:  # 否则为 UAV
                agent.type = 'UAV'
                agent.size = 0.1
                agent.Rs = 3
                agent.power_remain = np.random.uniform(30, 100)
                agent.threshold = 20
                agent.max_capacity = 100
                agent.power_consumption = 10
                agent.area = 0

        world.landmarks = [Landmark() for i in range(num_landmarks)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = 'landmark %d' % i
            landmark.collide = False
            landmark.movable = False
            landmark.size = 0.05
            # make initial conditions
        self.reset_world(world)
        return world

    def is_collision(self, agent1, agent2):
        delta_pos = agent1.state.p_pos - agent2.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        dist_min = agent1.size + agent2.size
        return True if dist < dist_min else False


    def reset_world(self, world):
        # Random properties for agents
        for i, agent in enumerate(world.agents):
            agent.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)
            # 设置代理的颜色和特定属性
            if agent.type == 'USV':  # 检查代理是否为 USV 类型
                agent.color = np.array([0.85, 0.35, 0.35])  # 设置 USV 的颜色
            elif agent.type == 'UAV':  # 检查代理是否为 UAV 类型
                agent.color = np.array([0.35, 0.35, 0.85])  # 设置 UAV 的颜色
                agent.power_remain = np.random.uniform(30, 100)
                if agent.power_remain < 50:
                    agent.color = np.array([1, 0, 0])
        # Random properties for landmarks
        for landmark in world.landmarks:
            landmark.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
            landmark.state.p_vel = np.zeros(world.dim_p)
            landmark.color = np.array([0.35, 0.35, 0.35])  # 设置地标的颜色

    def USVs(self,world):
        return [agent for agent in world.agents if agent.type == 'USV']
    def UAVs(self,world):
        return [agent for agent in world.agents if agent.type == 'UAV']


    def charge(self, agent, world):
        max_threshold = 100  # 充电阈值
        min_threshold = 20
        tdelta = 1  # 时间槽的持续时间 1s
        xi = 0.9
        charged_power = 0
        h = 1
        charge_rate = 50
        for usv_agent in self.USVs(world):
            for other_agent in self.UAVs(world):
                remaining_capacity = max_threshold - other_agent.power_remain
                # 计算USV和UAV之间的距离
                delta = usv_agent.state.p_pos - other_agent.state.p_pos
                dist_squared = np.sum(np.square(delta))
                distance = np.sqrt(dist_squared + h ** 2)
                if other_agent.power_remain < 50:
                    charge_amount = (xi * tdelta * charge_rate) / distance
                    if charge_amount <= remaining_capacity:
                        other_agent.power_remain += charge_amount
                        charged_power += charge_amount
                        # 确保充电后的电量在合理范围内
                        if other_agent.power_remain > max_threshold:
                            other_agent.power_remain = max_threshold
                    elif other_agent.power_remain < min_threshold:
                        other_agent.power_remain = min_threshold
                #print(other_agent.power_remain)
        return charged_power

    def USV_energy(self, agent,world):
        total_energy_consumption = 0
        sum_energy_squared = 0
        n = len(self.USVs(world))
        for agent in self.USVs(world):
            co_efficient = 0.5
            Res = 0.5
            agent.state.p_pos += agent.state.p_vel * world.dt
            delta_dist = np.linalg.norm(agent.state.p_vel * world.dt)
            agent_energy_consumption = co_efficient * delta_dist + Res
            agent.energy += agent_energy_consumption
            total_energy_consumption += agent_energy_consumption
            # total_energy_consumption += agent_energy_consumption
            sum_energy_squared += agent_energy_consumption ** 2

        return agent.energy



    def UAV_energy(self, world):
        for agent in self.UAVs(world):
            c = 1
            total_efficiency = 0
            kappa1 = 2.803
            kappa2 = 0.3177
            kappa3 = 0.0296
            nu = 2.5
            rho = 1 - (agent.power_remain - agent.threshold) / agent.max_capacity
            energy_consumption = kappa1 * agent.max_speed ** 3 + (kappa2 + kappa3) * nu ** (3 / 2)
            agent.power_remain -= energy_consumption
            #agent.power_remain -= agent.power_consumption
            if agent.power_remain < 30:
                self.charge(agent,world)
            if agent.power_remain < agent.threshold:
                agent.movable = False
            #print(agent.power_remain)
            beta_value = 1 if agent.power_remain > agent.threshold else 0
            #agent.area += math.pi * Rs ** 2
            efficiency = (beta_value * agent.power_remain + c) / (math.pi * agent.Rs ** 2)
            #efficiency = agent.area / world.size**2
            total_efficiency += efficiency
            #print("UAV搜索效率:", efficiency)
            # 计算平均搜索效率
        avg_efficiency = total_efficiency / len(self.UAVs(world))
        #print("平均搜索效率:", avg_efficiency)

        return rho



    def reward(self, agent, world):
        # Agents are rewarded based on minimum agent distance to each landmark
        main_reward = self.USVs_reward(agent,world) if agent.type == 'USV' else self.UAV_reward(agent,world)
        return main_reward


    def USVs_reward(self, agent, world):   #mysimple1 run1
        # Initialize reward variables
        char_rew = 0  # Total adversary reward
        pos_rew = 0  # Total positive reward for agents
        wd = -1  # 负系数
        wr = -1.5  # 负系数-1.33
        k = 10
        e = self.USV_energy(agent,world)
        # Get all UAV agents in the world
        UAV_agents = self.UAVs(world)
        USV_agents = self.USVs(world)

        # Calculate negative reward for adversaries and positive reward for agents
        for s in USV_agents:
            for a in UAV_agents:
                if a.power_remain < 50:
                    charged_power = self.charge(agent, world)
                    rho = self.UAV_energy(world)
                    char_rew += (charged_power + k) * rho  # Considering charged power
                    distance_to_uav = np.sqrt(np.sum(np.square(s.state.p_pos - a.state.p_pos)))
                    pos_rew += wd * distance_to_uav + wr * a.power_remain  # Encourage proximity to low-battery adversary
        # Combine adversary reward and positive agent reward
        total_reward = pos_rew + char_rew
        if agent.collide:
            for a in USV_agents:
                    if self.is_collision(a, agent):
                        total_reward -= 2 * char_rew
        for b in UAV_agents:
            if b.power_remain < b.threshold:
                total_reward -= char_rew
        total_reward /= e
        return total_reward



    def UAV_reward(self,agent, world):
        # Rewarded based on proximity to the goal landmark
        total_adv_rew = 0  # 初始化总奖励
        UAV_agents = self.UAVs(world)
        for uav_agent in UAV_agents:
                # 计算代理与目标地标之间的距离
                total_adv_rew -= 0.1 * min([np.sum(np.square(uav_agent.state.p_pos - l.state.p_pos))for l in world.landmarks])
        #total_adv_rew -= self.bound_reward(agent,world)
        return total_adv_rew  # 返回总奖励

    # @staticmethod
    # def bound_reward(agent,world):
    #     def bound(x):
    #         if x < 0.9:
    #             return 0
    #         if x < 1.0:
    #             return (x - 0.9) * 10
    #         return min(np.exp(2 * x - 2), 10)
    #
    #     total_reward = 0
    #     for p in range(world.dim_p):
    #         x = abs(agent.state.p_pos[p])
    #         total_reward -= bound(x)
    #     return total_reward

    def observation(self, agent, world):     # 观测
        # get positions of all entities in this agent's reference frame
        entity_pos = []
        for entity in world.landmarks:  # world.entities:
            entity_pos.append(entity.state.p_pos - agent.state.p_pos)

        entity_color = []
        for entity in world.landmarks:  # world.entities:
            entity_color.append(entity.color)
        # communication of all other agents
        comm = []
        other_pos = []
        for other in world.agents:
            if other is agent: continue
            comm.append(other.state.c)
            other_pos.append(other.state.p_pos - agent.state.p_pos)

        return np.concatenate([agent.state.p_vel] + [agent.state.p_pos] + entity_pos + other_pos + comm)








