import numpy as np
import random
import math
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment
import argparse
service_centers = 3
data = []
with open("pro_map.txt", "r") as file:
    for line in file:
        # 去除换行符和多余的空格
        line = line.strip()
        # 去除方括号
        line = line.strip("[]")
        # 按逗号分割
        row = line.split(", ")
        # 清理每个元素中的多余字符（如逗号和方括号）
        row = [x.strip(",]") for x in row]
        # 将字符串转换为浮点数
        row = [float(x) for x in row]
        # 将行添加到 data 中
        data.append(row)



point = []
threshold = 0.8

for i in range(len(data[0])):
        for j in range(len(data[0])):
            if(data[i][j] > threshold):
                point.append([i + 1,j + 1])
point_number = len(point)
points_locations = []

parser = argparse.ArgumentParser()
parser.add_argument("--seed",type=int)
config = parser.parse_args()
random.seed(2)

for i in range(service_centers * 2):
    points_locations.append(random.randint(0,20))

for p in point:
    p[0] -= 0.5
    p[1] -= 0.5
    points_locations.append(p[1])
    points_locations.append(p[0])
points_locations = np.array(points_locations)


points_locations = points_locations.reshape((service_centers+point_number, 2))

st = [False for i in range(service_centers+point_number)]


path = [[] for i in range(service_centers)]

for i in range(service_centers):
    path[i].append(i)
    st[i] = True

kmeans = KMeans(n_clusters=3, random_state=0)

# 对点进行聚类
kmeans.fit(points_locations[service_centers:])

# 获取每个点的聚类标签
labels = kmeans.labels_

# 获取聚类中心
centroids = kmeans.cluster_centers_


dis = []
for point in points_locations[0:service_centers]:
    for i in range(len(centroids)):
        dis.append(math.hypot(point[0] - centroids[i][0], point[1] - centroids[i][1]))



new_st = [False for i in range(service_centers)]


print(dis)
dis = [dis[i:i + service_centers] for i in range(0, len(dis), service_centers)]
row_ind, col_ind = linear_sum_assignment(dis)
print(col_ind)

for i in range(len(col_ind)):
    labels = np.insert(labels, 0, col_ind[len(col_ind) - i - 1])

print("聚类标签:", labels)

backup = points_locations.copy()



while(True):
    flag = True
    for i,usv in enumerate(backup[0:service_centers]):
        min_dis = 1e5
        min_idx = -1
        for j,point in enumerate(backup[service_centers:]):
            dis = math.hypot(usv[0] - point[0], usv[1] - point[1])
            if(dis < min_dis and st[j + service_centers] == False and labels[j + service_centers] == labels[i]):
                min_dis = dis
                min_idx = j + service_centers

        if min_idx != -1:
            st[min_idx] = True
            path[i].append(min_idx)
            backup[i] = backup[min_idx]
    for st_ in st:
        if st_ == False: flag = False

    if flag == True: break

res = 0

for p in path:
    for i in range(len(p)):
        if i == 0: continue
        res += math.hypot(points_locations[p[i]][0] - points_locations[p[i - 1]][0],points_locations[p[i]][1] - points_locations[p[i - 1]][1])

print(points_locations)

print(path)

print(res)
plt.figure(figsize=(10, 10))
plt.scatter(points_locations[:, 0], points_locations[:, 1], c='blue', label='Points')

# 绘制服务中心的路径
colors = ['red', 'green', 'purple']  # 为每个服务中心分配不同的颜色
for i, p in enumerate(path):
    x = [points_locations[idx][0] for idx in p]
    y = [points_locations[idx][1] for idx in p]
    plt.plot(x, y, marker='o', color=colors[i], label=f'Service Center {i+1}')

# 标记服务中心的起点
for i in range(service_centers):
    plt.scatter(points_locations[i][0], points_locations[i][1], c='black', marker='s', s=100, label=f'Start {i+1}' if i == 0 else "")

# 添加图例和标题
plt.legend()
plt.title('Service Centers Paths')
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.show()

with open('data_kmeans.txt','a') as file:
    file.write(f"{res}\n")