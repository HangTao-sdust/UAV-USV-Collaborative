import numpy as np


# 无人机携带水听器移动能耗
def hydrophone_energy_consumption(v):
    M = 11.5  # 无人机质量(Kg)
    m = None  # 水听器质量
    g = 9.8  # m/s^2
    H1 = 10  # 绳子长度（m）,无人机距水面10m
    p = 1.02  # 1.01g/m^3 ~ 1.03g/m^3
    C_d = 0.5  # 阻力系数 （球形水听器阻力为0.5）
    A = None  # 阻力面积
    # v = None  # 飞行速度
    F = 0.5 * p * C_d * A * (v ** 2)
    u = 0.1  # 绳子的摩擦系数(0.1~0.3)
    T = F / u
    hydrophone_energy = (M + m) * g * H1 + T * H1
    return hydrophone_energy

    pass


# # 水听器移动时所受阻力
# def hydrophone_resistance():
#     p = None  # 海水密度
#     C_d = 0.5  # 阻力系数 （球形水听器阻力为0.5）
#     A = None  # 阻力面积
#     v = None  # 飞行速度
#     F = 0.5 * p * C_d * A * (v ** 2)
#     return F
#     pass
#
#
# # 绳子拉力
# def hydrophone_tension():
#     u = 0.1  # 绳子的摩擦系数(0.1~0.3)
#     F = hydrophone_resistance()
#     T = F / u
#     return T
#     pass


def energy_consumption(delta_dist, v):  # 能耗
    hover_energy_cost = 0.5
    co_efficient = 0.5
    hydrophone_energy = hydrophone_energy_consumption(v)
    agent_energy_consumption = hover_energy_cost + co_efficient * delta_dist + hydrophone_energy
    return agent_energy_consumption

"""
def cal_fairness():
    K = None
    
"""



