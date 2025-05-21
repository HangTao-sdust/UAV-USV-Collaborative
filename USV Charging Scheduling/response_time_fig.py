import matplotlib.pyplot as plt
import numpy as np

# 数据
uav_types = ['5UAV', '6UAV', '7UAV']
maddpg = [282, 303, 324]
random = [371, 384, 392]
# td3 = [0, 0, 0]

# 绘制折线图
plt.figure(figsize=(10, 6))
plt.plot(uav_types, maddpg, marker='o', label='MADDPG')
plt.plot(uav_types, random, marker='s', label='Random')
# plt.plot(uav_types, td3, marker='^', label='TD3')

# 添加图例
plt.legend()

# 添加标题和轴标签
plt.title('Response Time for Different UAV Types and Algorithms')
plt.xlabel('UAV number')
plt.ylabel('Response Time (s)')

# 显示图表
plt.grid(True)
plt.show()