import random
import math
import matplotlib.pyplot as plt

# 常量定义
V_in = 5.0  # 输入电压 (V)
I_in = 2.0  # 输入电流 (A)
eta_boost = 0.9  # 升压效率
eta_coupling = 0.85  # 磁耦合效率
eta_rect = 0.9  # 整流效率
E_max = 100.0  # UAV 最大电量
W = 1.5  # UAV 重量 (kg)
rho = 1.225  # 空气密度 (kg/m^3)
A = 0.1  # 转子总面积 (m^2)
dt = 1  # 时间步长 (s)

# 充电过程计算
def calculate_charging_power(current_energy):
    # 充电功率非线性：电量越少充电越快，电量接近 E_max 时充电越慢
    charging_rate = 1 - (current_energy / E_max)  # 电量越少，充电速率越高
    P_in = V_in * I_in  # 输入功率
    P_boost = P_in * eta_boost  # 升压后功率
    P_recv = P_boost * eta_coupling * eta_rect  # 接收功率
    return P_recv * charging_rate  # 非线性充电功率

# 能量消耗计算
def calculate_energy_consumption(current_energy):
    # 能量消耗非线性：电量越多消耗越快，电量越少消耗越慢
    consumption_rate = current_energy / E_max  # 电量越多，消耗速率越高
    base_consumption = 1.0  # 基础消耗
    return base_consumption * consumption_rate  # 非线性消耗

# UAV 类
class UAV:
    def __init__(self, id, initial_energy):
        self.id = id
        self.energy = initial_energy
        self.charging = False
        self.energy_history = []  # 记录电量变化

    def update_energy(self, E_harvest, E_consumption):
        if self.charging:
            self.energy += E_harvest  # 充电
            if self.energy >= 99:  # 电量达到 99，停止充电
                self.energy = 99
                self.charging = False
        self.energy -= E_consumption  # 无论是否充电，都会消耗能量
        if self.energy < 20:  # 电量低于阈值，开始充电
            self.charging = True
        self.energy_history.append(self.energy)
        print(self.energy)

# 初始化 UAV
uavs = [UAV(i, random.uniform(30, 100)) for i in range(5)]

# 模拟过程
time_steps = 300
for t in range(time_steps):
    for uav in uavs:
        if uav.charging:
            E_harvest = calculate_charging_power(uav.energy) * dt  # 非线性充电能量
        else:
            E_harvest = 0  # 不充电时无充电能量
        E_consumption = calculate_energy_consumption(uav.energy) * dt  # 非线性消耗能量
        uav.update_energy(E_harvest, E_consumption)

# 绘制电量变化图
plt.figure(figsize=(12, 6))
colors = ['b', 'g', 'r', 'c', 'm']  # 每个 UAV 的颜色
for uav in uavs:
    plt.plot(range(time_steps), uav.energy_history, label=f"UAV {uav.id}", color=colors[uav.id], linewidth=2)

# 图表美化
plt.title("UAV 电量变化 (非线性充电和消耗)", fontsize=16)
plt.xlabel("时间步长", fontsize=14)
plt.ylabel("电量", fontsize=14)
plt.legend(loc="upper right", fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.ylim(0, 110)  # 设置 y 轴范围
plt.tight_layout()  # 调整布局
plt.show()