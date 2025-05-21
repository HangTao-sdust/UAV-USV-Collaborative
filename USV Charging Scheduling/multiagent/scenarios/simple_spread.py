# import numpy as np
# from multiagent.core import World, Agent, Landmark
# from multiagent.scenario import BaseScenario
#
#
# class Scenario(BaseScenario):
#     def make_world(self):
#         world = World()
#         # set any world properties first
#         world.dim_c = 2
#         num_agents = 3
#         num_landmarks = 3
#         # add agents
#         world.agents = [Agent() for i in range(num_agents)]
#         for i, agent in enumerate(world.agents):
#             agent.name = 'agent %d' % i
#             agent.collide = True
#             agent.silent = True
#             agent.size = 0.05
#             agent.accel = 2
#             agent.max_speed = 2
#         # add landmarks
#         world.landmarks = [Landmark() for i in range(num_landmarks)]
#         for i, landmark in enumerate(world.landmarks):
#             landmark.name = 'landmark %d' % i
#             landmark.collide = False
#             landmark.movable = False
#             #landmark.power_remain = np.random.uniform(30, 100)
#             landmark.max_capacity = 351288
#             landmark.threshold = 0.2 * landmark.max_capacity
#             landmark.energy = np.random.uniform(10386,351288)
#             landmark.weight = 4
#             landmark.rotor_area = 0.18
#             landmark.is_covered = False
#         # make initial conditions
#         self.reset_world(world)
#         return world
#
#     def reset_world(self, world):
#         # random properties for agents
#         for i, agent in enumerate(world.agents):
#             agent.color = np.array([0.35, 0.35, 0.85])
#         # random properties for landmarks
#         for i, landmark in enumerate(world.landmarks):
#             landmark.color = np.array([0.25, 0.25, 0.25])
#         # set random initial states
#         for agent in world.agents:
#             agent.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
#             agent.state.p_vel = np.zeros(world.dim_p)
#             agent.state.c = np.zeros(world.dim_c)
#         for i, landmark in enumerate(world.landmarks):
#             landmark.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
#             landmark.state.p_vel = np.zeros(world.dim_p)
#
#     def calculate_flight_power(self, landmark):
#         # Calculate V_h
#         air_density=1.225
#         velocity = 11
#         V_h = np.sqrt(landmark.weight / (2 * air_density * landmark.rotor_area))
#         # Calculate P_fly
#         P_fly = landmark.weight / (np.sqrt(2) * air_density * landmark.rotor_area) * \
#                 1 / np.sqrt(velocity ** 2 + np.sqrt(velocity ** 4 + 4 * V_h ** 4))
#         return P_fly
#
#     def calculate_energy_consumption(self, landmark):
#         # Energy consumption during flight
#         time_step = 100
#         P_fly = self.calculate_flight_power(landmark)
#         E_fly = P_fly * time_step
#         # Energy consumption during communication (assumed constant)
#         E_com = 0.1 * time_step
#         # Energy consumption during operation (assumed constant)
#         E_op = 0.05 * time_step
#         # Total energy consumption
#         E_total = (E_fly + E_com + E_op) * time_step
#         landmark.energy -= E_total
#         return landmark.energy
#
#     # def charge(self,agent,world):
#     #     V_in = 12.0  # 输入电压 (V)
#     #     I_in = 2.0  # 输入电流 (A)
#     #     eta_boost = 0.9  # 升压效率
#     #     eta_coupling = 0.85  # 磁耦合效率
#     #     eta_rect = 0.95  # 整流效率
#     #     t = 360  # 充电时间 (s)
#     #     for landmark in world.landmarks:
#     #         E_max = landmark.max_capacity  # 最大能量容量 (J)
#     #         E_re_m_prev = self.calculate_energy_consumption(landmark)
#     #         if E_re_m_prev < landmark.threshold:
#     #             # 计算输入功率
#     #             P_in = V_in * I_in
#     #                 # 计算升压后的输出功率
#     #             P_boost = P_in * eta_boost
#     #                 # 计算接收端的直流功率
#     #             P_recv = P_boost * eta_coupling * eta_rect
#     #                 # 计算充电能量
#     #             E_har_t = P_recv * t
#     #
#     #                 # 更新无人机的剩余能量
#     #             if E_re_m_prev + E_har_t >= E_max:
#     #                 E_re_m_t = E_max
#     #             else:
#     #                 E_re_m_t = E_re_m_prev + E_har_t
#     #
#     #     #landmark.energy = E_re_m_t
#     #     print(E_re_m_t)
#     #     return E_har_t
#
#     def charge(self, agent, world):
#         V_in = 12.0  # 输入电压 (V)
#         I_in = 2.0  # 输入电流 (A)
#         eta_boost = 0.9  # 升压效率
#         eta_coupling = 0.85  # 磁耦合效率
#         eta_rect = 0.95  # 整流效率
#         t = 360  # 充电时间 (s)
#         for landmark in world.landmarks:
#             E_max = landmark.max_capacity  # 最大能量容量 (J)
#             E_re_m_prev = self.calculate_energy_consumption(landmark)
#             print(f"Previous energy: {E_re_m_prev}, Threshold: {landmark.threshold}")  # 调试输出
#             if E_re_m_prev < landmark.threshold:
#                 # 计算输入功率
#                 P_in = V_in * I_in
#                 print(f"Input power: {P_in}")  # 调试输出
#                 # 计算升压后的输出功率
#                 P_boost = P_in * eta_boost
#                 print(f"Boosted power: {P_boost}")  # 调试输出
#                 # 计算接收端的直流功率
#                 P_recv = P_boost * eta_coupling * eta_rect
#                 print(f"Received power: {P_recv}")  # 调试输出
#                 # 计算充电能量
#                 E_har_t = P_recv * t
#                 print(f"Harvested energy: {E_har_t}")  # 调试输出
#                 # 更新无人机的剩余能量
#                 if E_re_m_prev + E_har_t >= E_max:
#                     E_re_m_t = E_max
#                 else:
#                     E_re_m_t = E_re_m_prev + E_har_t
#                 print(f"Updated energy: {E_re_m_t}")  # 调试输出
#         return E_har_t
#
#
#
#     def USV_energy(self, world):  # 能耗
#         total_energy_consumption = 0
#         for agent in world.agents:
#             co_efficient = 0.5
#             Res = 0.5
#             agent.state.p_pos += agent.state.p_vel * world.dt
#             delta_dist = np.sqrt(np.sum(np.square(agent.state.p_vel * world.dt)))
#             agent_energy_consumption = co_efficient * delta_dist + Res
#             agent.energy += agent_energy_consumption
#             total_energy_consumption += agent_energy_consumption
#         return total_energy_consumption
#
#
#
#     def benchmark_data(self, agent, world):
#         rew = 0
#         collisions = 0
#         occupied_landmarks = 0
#         min_dists = 0
#         for l in world.landmarks:
#             dists = [np.sqrt(np.sum(np.square(a.state.p_pos - l.state.p_pos))) for a in world.agents]
#             min_dists += min(dists)
#             rew -= min(dists)
#             if min(dists) < 0.1:
#                 occupied_landmarks += 1
#         if agent.collide:
#             for a in world.agents:
#                 if self.is_collision(a, agent):
#                     rew -= 1
#                     collisions += 1
#         return (rew, collisions, min_dists, occupied_landmarks)
#
#
#     def is_collision(self, agent1, agent2):
#         delta_pos = agent1.state.p_pos - agent2.state.p_pos
#         dist = np.sqrt(np.sum(np.square(delta_pos)))
#         dist_min = agent1.size + agent2.size
#         return True if dist < dist_min else False
#
#     def reward(self, agent, world):
#         rew = 0
#         # 1. Energy Consumption Reward (Re_i^t)
#         energy_cost = self.USV_energy(world)
#         if energy_cost > 0:
#             Re = 1 / energy_cost
#         else:
#             Re = 0
#
#         # 2. Charging Reward (Rc_i^t)
#         Rc = 0
#         for landmark in world.landmarks:
#             if landmark.energy < landmark.threshold:  # If landmark energy is below threshold
#                 dist = np.sqrt(np.sum(np.square(agent.state.p_pos - landmark.state.p_pos)))
#                 if dist < 0.1:  # If agent is close enough to cover the landmark
#                     landmark.is_covered = True  # Mark the landmark as covered
#
#                 if landmark.is_covered and landmark.energy < landmark.threshold:  # If landmark is covered and energy is below threshold
#                     E_har = self.charge(agent, landmark)  # Charge the landmark
#                     rho = 1 - (landmark.energy - landmark.threshold) / landmark.max_capacity  # Charging urgency
#                     Rc += (E_har + 1) * rho  # k = 1
#
#         # 3. Penalty Term (R_l)
#         Rl = 0
#         for landmark in world.landmarks:
#             if landmark.energy < landmark.threshold:  # If landmark energy is below threshold
#                 Rl -= Rc  # Apply penalty
#
#         # 4. Collision Penalty
#         if agent.collide:
#             for other_agent in world.agents:
#                 if other_agent is not agent and self.is_collision(agent, other_agent):
#                     rew -= 10  # Collision penalty
#
#         # Total reward
#         total_reward = Re * Rc + Rl
#         return total_reward
#
#     def observation(self, agent, world):
#         # get positions of all entities in this agent's reference frame
#         entity_pos = []
#         for entity in world.landmarks:  # world.entities:
#             entity_pos.append(entity.state.p_pos - agent.state.p_pos)
#         # entity colors
#         entity_color = []
#         for entity in world.landmarks:  # world.entities:
#             entity_color.append(entity.color)
#         # communication of all other agents
#         comm = []
#         other_pos = []
#         for other in world.agents:
#             if other is agent: continue
#             comm.append(other.state.c)
#             other_pos.append(other.state.p_pos - agent.state.p_pos)
#         return np.concatenate([agent.state.p_vel] + [agent.state.p_pos] + entity_pos + other_pos + comm)
#
#     def done(self, agent, world):
#         for p in range(world.dim_p):
#             x = abs(agent.state.p_pos[p])
#             if (x>1.0):
#                 return True
#         return False
#



import numpy as np
from multiagent.core import World, Agent, Landmark
from multiagent.scenario import BaseScenario


class Scenario(BaseScenario):
    def make_world(self):
        world = World()
        # set any world properties first
        world.dim_c = 2
        num_agents = 3
        num_landmarks = 3
        # add agents
        world.agents = [Agent() for i in range(num_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
            agent.collide = True
            agent.silent = True
            agent.size = 0.05
            agent.accel = 2
            agent.max_speed = 2
        # add landmarks
        world.landmarks = [Landmark() for i in range(num_landmarks)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = 'landmark %d' % i
            landmark.collide = False
            landmark.movable = False
            landmark.max_capacity = 351288
            landmark.threshold = 0.2 * landmark.max_capacity
            landmark.energy = np.random.uniform(10386, 351288)
            landmark.weight = 4
            landmark.rotor_area = 0.18
            landmark.is_covered = False
        # make initial conditions
        self.reset_world(world)
        return world

    def reset_world(self, world):
        # random properties for agents
        for i, agent in enumerate(world.agents):
            agent.color = np.array([0.35, 0.35, 0.85])
        # random properties for landmarks
        for i, landmark in enumerate(world.landmarks):
            landmark.color = np.array([0.25, 0.25, 0.25])
        # set random initial states
        for agent in world.agents:
            agent.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)
        for i, landmark in enumerate(world.landmarks):
            landmark.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
            landmark.state.p_vel = np.zeros(world.dim_p)

    def calculate_flight_power(self, landmark):
        # Calculate V_h
        air_density = 1.225
        velocity = 11
        V_h = np.sqrt(landmark.weight / (2 * air_density * landmark.rotor_area))
        # Calculate P_fly
        P_fly = landmark.weight / (np.sqrt(2) * air_density * landmark.rotor_area) * \
                1 / np.sqrt(velocity ** 2 + np.sqrt(velocity ** 4 + 4 * V_h ** 4))
        return P_fly

    def calculate_energy_consumption(self, landmark):
        # Energy consumption during flight
        time_step = 100
        P_fly = self.calculate_flight_power(landmark)
        E_fly = P_fly * time_step
        # Energy consumption during communication (assumed constant)
        E_com = 0.1 * time_step
        # Energy consumption during operation (assumed constant)
        E_op = 0.05 * time_step
        # Total energy consumption
        E_total = (E_fly + E_com + E_op) * time_step
        landmark.energy -= E_total
        return landmark.energy

    def charge(self, agent, landmark):
        V_in = 12.0  # 输入电压 (V)
        I_in = 2.0  # 输入电流 (A)
        eta_boost = 0.9  # 升压效率
        eta_coupling = 0.85  # 磁耦合效率
        eta_rect = 0.95  # 整流效率
        t = 360  # 充电时间 (s)

        E_max = landmark.max_capacity  # 最大能量容量 (J)
        E_re_m_prev = landmark.energy  # 当前能量
        print(f"Previous energy: {E_re_m_prev}, Threshold: {landmark.threshold}")  # 调试输出

        if E_re_m_prev < landmark.threshold:
            # 计算输入功率
            P_in = V_in * I_in
            print(f"Input power: {P_in}")  # 调试输出
            # 计算升压后的输出功率
            P_boost = P_in * eta_boost
            print(f"Boosted power: {P_boost}")  # 调试输出
            # 计算接收端的直流功率
            P_recv = P_boost * eta_coupling * eta_rect
            print(f"Received power: {P_recv}")  # 调试输出
            # 计算充电能量
            E_har_t = P_recv * t
            print(f"Harvested energy: {E_har_t}")  # 调试输出
            # 更新无人机的剩余能量
            if E_re_m_prev + E_har_t >= E_max:
                E_re_m_t = E_max
            else:
                E_re_m_t = E_re_m_prev + E_har_t
            print(f"Updated energy: {E_re_m_t}")  # 调试输出
            landmark.energy = E_re_m_t  # 更新 landmark 的能量
        else:
            E_har_t = 0  # 如果能量高于阈值，不充电

        return E_har_t

    def USV_energy(self, world):  # 能耗
        total_energy_consumption = 0
        for agent in world.agents:
            co_efficient = 0.5
            Res = 0.5
            agent.state.p_pos += agent.state.p_vel * world.dt
            delta_dist = np.sqrt(np.sum(np.square(agent.state.p_vel * world.dt)))
            agent_energy_consumption = co_efficient * delta_dist + Res
            agent.energy += agent_energy_consumption
            total_energy_consumption += agent_energy_consumption
        return total_energy_consumption

    def is_collision(self, agent1, agent2):

        delta_pos = agent1.state.p_pos - agent2.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        dist_min = agent1.size + agent2.size
        return True if dist < dist_min else False

    def reward(self, agent, world):
        rew = 0
        # 1. Energy Consumption Reward (Re_i^t)
        energy_cost = self.USV_energy(world)
        if energy_cost > 0:
            Re = 1 / energy_cost
        else:
            Re = 0

        # 2. Charging Reward (Rc_i^t)
        Rc = 0
        for landmark in world.landmarks:
            if landmark.energy < landmark.threshold:  # If landmark energy is below threshold
                dist = np.sqrt(np.sum(np.square(agent.state.p_pos - landmark.state.p_pos)))
                if dist < 0.1:  # If agent is close enough to cover the landmark
                    landmark.is_covered = True  # Mark the landmark as covered

                if landmark.is_covered and landmark.energy < landmark.threshold:  # If landmark is covered and energy is below threshold
                    E_har = self.charge(agent, landmark)  # Charge the landmark
                    rho = 1 - (landmark.energy - landmark.threshold) / landmark.max_capacity  # Charging urgency
                    Rc += (E_har + 1) * rho  # k = 1

        # 3. Penalty Term (R_l)
        Rl = 0
        for landmark in world.landmarks:
            if landmark.energy < landmark.threshold:  # If landmark energy is below threshold
                Rl -= Rc  # Apply penalty

        # 4. Collision Penalty
        if agent.collide:
            for other_agent in world.agents:
                if other_agent is not agent and self.is_collision(agent, other_agent):
                    rew -= 10  # Collision penalty

        # Total reward
        total_reward = Re * Rc + Rl
        return total_reward

    def observation(self, agent, world):
        # get positions of all entities in this agent's reference frame
        entity_pos = []
        for entity in world.landmarks:  # world.entities:
            entity_pos.append(entity.state.p_pos - agent.state.p_pos)
        # entity colors
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

    def done(self, agent, world):
        for p in range(world.dim_p):
            x = abs(agent.state.p_pos[p])
            if x > 1.0:
                return True
        return False