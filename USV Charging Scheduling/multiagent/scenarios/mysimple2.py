import numpy as np
from multiagent.core import World, Agent, Landmark
from multiagent.scenario import BaseScenario
import math

class Scenario(BaseScenario):

    def make_world(self):
        world = World()
        # set any world properties first
        world.dim_c = 2
        num_USVs = 3 #3、4
        num_UAVs = 6 #5、6、7
        world.num_agents = num_USVs + num_UAVs
        num_landmarks = 1

        # 初始化时间
        world.time = 0  # 仿真时间从 0 开始

        # add agents
        world.agents = [Agent() for i in range(world.num_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
            agent.collide = False
            agent.silent = True
            agent.accel = 2
            agent.max_speed = 1
            agent.velocity = 2
            # 根据代理的索引 i 判断代理类型并设置特定属性
            if i < num_USVs:  # 如果代理索引小于 USVs 数量，则为 USV
                agent.type = 'USV'
                agent.size = 0.05
                agent.energy = 0
                agent.movable = True
            else:  # 否则为 UAV
                agent.type = 'UAV'
                agent.size = 0.1
                agent.Rs = 3
                agent.power_remain = np.random.uniform(30, 100)
                agent.threshold = 20
                agent.max_capacity = 100
                agent.power_consumption = 10
                agent.area = 0
                agent.ran = np.random.uniform(20,30)
                agent.total_area = 0
                agent.total_energy_consumption = 0
                agent.request_time = None  # 记录请求时间
                agent.charge_start_time = None  # 记录充电开始时间
                agent.charge_end_time = None  # 记录充电结束时间
                agent.movable = True  # UAV 可以移动
                agent.response_time = -1.

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
        # 重置仿真时间
        world.time = 0
        # Random properties for agents
        for i, agent in enumerate(world.agents):
            agent.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)
            agent.state.energy = 0
            # 设置代理的颜色和特定属性
            if agent.type == 'USV':  # 检查代理是否为 USV 类型
                agent.color = np.array([0.85, 0.35, 0.35])  # 设置 USV 的颜色
            elif agent.type == 'UAV':  # 检查代理是否为 UAV 类型
                agent.color = np.array([0.35, 0.35, 0.85])  # 设置 UAV 的颜色
                agent.power_remain = np.random.uniform(30, 100)
                if agent.power_remain < agent.threshold:
                    agent.color = np.array([1, 0, 0])

                # 重置充电相关属性
                agent.request_time = None
                agent.charge_start_time = None
                agent.charge_end_time = None
                agent.response_time = -1.
        # Random properties for landmarks
        for landmark in world.landmarks:
            landmark.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
            landmark.state.p_vel = np.zeros(world.dim_p)
            landmark.color = np.array([0.35, 0.35, 0.35])  # 设置地标的颜色

    def USVs(self,world):
        return [agent for agent in world.agents if agent.type == 'USV']
    def UAVs(self,world):
        return [agent for agent in world.agents if agent.type == 'UAV']

    def USV_energy(self, world):  # 能耗
        total_energy_consumption = 0

        for agent in world.agents:
            if agent.type == 'USV':
                co_efficient = 0.5
                Res = 0.5
                agent.state.p_pos += agent.state.p_vel * world.dt
                delta_dist = np.sqrt(np.sum(np.square(agent.state.p_vel * world.dt)))
                agent_energy_consumption = co_efficient * delta_dist + Res
                agent.state.energy += agent_energy_consumption
                total_energy_consumption += agent_energy_consumption
        return total_energy_consumption



    def UAV_energy(self, world, agent):
        rho = None  # 设置默认值
        # 计算 rho
        rho = 1 - (agent.power_remain - agent.threshold) / agent.max_capacity
        # 飞行功率计算
        air_density = 1.225
        weight = 2  # UAV 的重量 (kg)
        rotor_area = 0.18  # 旋翼面积 (m^2)
        velocity = 2
        # 计算飞行功率
        V_h = np.sqrt(weight / (2 * air_density * rotor_area))
        P_fly = weight / (np.sqrt(2) * air_density * rotor_area) * \
                1 / np.sqrt(velocity ** 2 + np.sqrt(velocity ** 4 + 4 * V_h ** 4))
        # 能量消耗计算
        time_step = 5  # 时间步长 (s)
        E_fly = P_fly * time_step  # 飞行能量消耗
        E_com = 0.1 * time_step  # 通信能量消耗
        E_op = 0.05 * time_step  # 操作能量消耗
        # 计算总能量消耗
        energy_consumption = (E_fly + E_com + E_op) * (rho**2)  # 总能量消耗
        # print(f"{agent.type}{agent.name}能耗:{energy_consumption}")
        agent.power_remain -= energy_consumption
        agent.total_energy_consumption += energy_consumption
        if agent.power_remain < agent.ran:
            agent.color = np.array([0.35, 0.35, 0.35])  # 设置 USV 的颜色
        if agent.power_remain <= 0:
            agent.power_remain = 0
        # print(f"{agent.type} {agent.name}充电阈值:{agent.ran},剩余电量:{agent.power_remain}")
        # 检查电量是否低于阈值
        if agent.power_remain < agent.ran:
            if agent.request_time is None:  # 如果还未发送请求
                agent.request_time = world.time  # 记录请求时间
                # print(f"{agent.type} {agent.name}发出充电请求时间: {agent.request_time}")
            self.charge(world,agent)
        if agent.power_remain < agent.threshold:
            agent.movable = False
            agent.color = np.array([0.35, 0.35, 0.35])  # 设置 UAV 的颜色
        #print(agent.power_remain)
        return rho  # 返回 rho 的值




    def charge(self, world, agent):
        V_in = 5.0  # 输入电压 (V)
        I_in = 1.0  # 输入电流 (A)
        eta_boost = 0.9  # 升压效率
        eta_coupling_max = 0.85  # 磁耦合效率
        eta_rect = 0.95  # 整流效率
        t = 100  # 充电时间 (s)
        charged_power = 0
        max_threshold = 100  # 充电阈值
        k = 0.2

        # 找到最近的USV
        nearest_usv = None
        min_distance = float('inf')
        for usv in self.USVs(world):
            distance = np.sqrt(np.sum(np.square(usv.state.p_pos - agent.state.p_pos)))
            if distance < min_distance:
                min_distance = distance
                nearest_usv = usv
        # print(f"{agent.type} {agent.name}'s min_distance is {min_distance}")
        if nearest_usv is not None and min_distance < 0.3:
            # 记录充电开始时间
            if agent.charge_start_time is None:
                agent.charge_start_time = world.time
                # print(f"{agent.type} {agent.name}开始充电时间: {agent.charge_start_time}")

            # 计算充电量
            eta_coupling = eta_coupling_max * math.exp(-k * agent.power_remain)

            P_in = V_in * I_in
            P_boost = P_in * eta_boost
            P_recv = P_boost * eta_coupling * eta_rect
            charge_amount = P_recv * t

            if agent.power_remain + charge_amount > max_threshold:
                agent.power_remain = max_threshold
            else:
                agent.power_remain += charge_amount

            if agent.power_remain >= max_threshold:
                agent.charge_end_time = world.time

        return charged_power

    def calculate_response_time(self, uav_agent):
        if uav_agent.charge_start_time is not None and uav_agent.request_time is not None:
            response_time = uav_agent.charge_start_time - uav_agent.request_time
            return response_time
        return None



    def reward(self, agent, world):
        if agent.type == 'UAV':
            self.UAV_energy(world,agent)
        # Agents are rewarded based on minimum agent distance to each landmark
        main_reward = 0.
        if agent.type == 'USV':
            main_reward = self.new_USV_reward(agent, world)
        else:
            main_reward = self.UAV_reward(agent,world)

        # 如果是 UAV 并且已经开始充电，计算并输出响应时间
        if agent.type == 'UAV' and agent.charge_start_time is not None:
            response_time = self.calculate_response_time(agent)
            if response_time is not None:
                # print(f"UAV {agent.name} 的响应时间为: {response_time} 秒")
                agent.response_time = response_time

        return main_reward

    def new_USV_reward(self, agent, world):
        total_left_energy = 0.
        min_dist = 9999999.
        # 所有UAV的剩余电量（求平均） + 离agent最近的需要充电的UAV的距离
        uav = self.UAVs(world)
        for uav_agent in uav:
            total_left_energy += uav_agent.power_remain
            if uav_agent.power_remain < uav_agent.ran:
                dist = np.sqrt(np.sum(np.square(uav_agent.state.p_pos - agent.state.p_pos)))
                if dist < min_dist:
                    min_dist = dist
        total_left_energy = total_left_energy / len(uav)
        dist_rew = 0.5/(min_dist+0.1)
        return dist_rew + total_left_energy

    def USVs_reward(self, agent, world):
        # Initialize reward variables
        char_rew = 0  # Total adversary reward
        pos_rew = 0  # Total positive reward for agents
        wd = -1  # 负系数
        wr = -1.5  # 负系数-1.33
        k = 10
        e = self.USV_energy(world)
        target_uav = None
        min_distance = float('inf')
        for uav in self.UAVs(world):
            distance = np.sqrt(np.sum(np.square(uav.state.p_pos - agent.state.p_pos)))
            if distance < min_distance:
                min_distance = distance
                target_uav = uav
        rho = 1 - (target_uav.power_remain - target_uav.threshold) / target_uav.max_capacity
        # Get all UAV agents in the world
        UAV_agents = self.UAVs(world)
        USV_agents = self.USVs(world)

        # Calculate negative reward for adversaries and positive reward for agents
        for s in USV_agents:
            for a in UAV_agents:
                char_rew += (0 + k) * rho  # Considering charged power
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
        # R_e
        total_reward /= e
        return total_reward


    def UAV_reward(self,agent, world):
        # Rewarded based on proximity to the goal landmark
        total_adv_rew = 0  # 初始化总奖励
        UAV_agents = self.UAVs(world)
        for uav_agent in UAV_agents:
                # 计算代理与目标地标之间的距离
            total_adv_rew -= 0.1 * min([np.sum(np.square(uav_agent.state.p_pos - l.state.p_pos))for l in world.landmarks])

        return total_adv_rew  # 返回总奖励


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
