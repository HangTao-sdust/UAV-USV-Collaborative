# import numpy as np
# import math
#
# # 定义UAV类
# class UAV:
#     def __init__(self, power_remain):
#         self.power_remain = power_remain
#         self.Rs = 3  # 搜索半径，默认为10
#         self.charging = False  # 是否在充电中，默认为否
#         self.charge_time = 0  # 充电时间，默认为0
#
# # 定义搜索效率函数
# def efficiency(agent, beta_value, c):
#     if agent.power_remain < 20 or agent.charging:
#         return 0
#     else:
#
#         return (beta_value * agent.power_remain + c) / (math.pi * agent.Rs ** 2)
#
# # 归一化搜索效率函数
# def normalize_efficiency(efficiency_value, max_efficiency):
#     return efficiency_value / max_efficiency
#
# # 定义充电函数
# def charge(agent):
#     agent.charging = True
#     agent.charge_time += 1
#     if agent.charge_time == 5:  # 假设充电持续时长为2步
#         agent.power_remain = 100  # 充电满后电量为100
#         agent.charging = False
#         agent.charge_time = 0
#
# # 初始化5个UAV，电量在[30, 100]之间随机分布
# uavs = [UAV(np.random.uniform(30, 100)) for _ in range(7)]
#
# # 定义搜索效率参数
# beta_value = 1
# c = 1
#
# # 计算最大搜索效率
# max_efficiency = efficiency(UAV(100), beta_value, c)
#
# # 模拟搜索和充电过程
# for step in range(50):
#     print("Step:", step+1)
#     for i, uav in enumerate(uavs):
#         print("UAV {}: Power remaining = {}".format(i+1, uav.power_remain))
#         if uav.power_remain < 20:
#             print("Low power, returning to charging station...")
#             charge(uav)
#             print("UAV {}: Fully charged. Resuming search.".format(i+1))
#         else:
#             search_efficiency = efficiency(uav, beta_value, c)
#             normalized_efficiency = normalize_efficiency(search_efficiency, max_efficiency)
#             print("UAV {}: Search efficiency = {}".format(i+1, normalized_efficiency))
#             # 消耗电量
#             uav.power_remain -= 10
#     print()
#
#
#
#
#
#
#
import numpy as np
import math

# 定义UAV类
class UAV:
    def __init__(self, power_remain):
        self.power_remain = power_remain
        self.Rs = 3  # 搜索半径，默认为10
        self.charging = False  # 是否在充电中，默认为否
        self.charge_time = 0  # 充电时间，默认为0
        self.total_energy = 0  # 总能量消耗
        self.total_area = 0  # 总搜索面积
        self.power_consume = 10


# 计算效率
    def efficiency(self):
        rho = 1-(uav.power_remain-20)/100
        Rs = -2*rho + 3
        uav.total_area += math.pi * Rs ** 2
        uav.total_energy += uav.power_consume
        efficiency = uav.total_energy/uav.total_area
        return efficiency

# 归一化搜索效率函数
def normalize_efficiency(efficiency_value, max_efficiency):
    return efficiency_value / max_efficiency

# 定义充电函数
def charge(agent):
    agent.charging = True
    agent.charge_time += 1
    if agent.charge_time == 5:  # 假设充电持续时长为2步
        agent.power_remain = 100  # 充电满后电量为100
        agent.charging = False
        agent.charge_time = 0

# 初始化5个UAV，电量在[30, 100]之间随机分布
uavs = [UAV(np.random.uniform(30, 100)) for _ in range(5)]

# 模拟搜索和充电过程
for step in range(50):
    print("Step:", step+1)
    for i, uav in enumerate(uavs):
        rho = 1 - (uav.power_remain - 20) / 100
        Rs = -2 * rho + 3
        uav.total_area += math.pi * Rs ** 2
        uav.total_energy += uav.power_consume
        # search_efficiency = uav.total_energy / uav.total_area

        search_efficiency = uav.efficiency()
        # print(search_efficiency)
        print("UAV {}: Search efficiency = {}".format(i + 1, search_efficiency / 2))
        # max_efficiency = math.pi * (uav.Rs ** 2) / 10  # 计算最大搜索效率
        # normalized_efficiency = normalize_efficiency(search_efficiency, max_efficiency)
        # print("UAV {}: Search efficiency = {}".format(i, normalized_efficiency))

        #print("UAV {}: Power remaining = {}".format(i+1, uav.power_remain))
        if uav.power_remain < 20:
            print("Low power, returning to charging station...")
            charge(uav)
            print("UAV {}: Fully charged. Resuming search.".format(i+1))
        else:
            # 消耗电量
            uav.power_remain -= 10
    print()
