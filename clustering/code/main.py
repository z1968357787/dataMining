import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import matplotlib.pyplot as plt
import sklearn.datasets as datasets
import DBSCAN
import KMeansRelevant


def show_cluster_result(cluster, normal_data, cluster_method):
    # 第一幅图展示原始图
    plt.figure(figsize=(24, 10), dpi=80)
    plt.subplot(1, 2, 1)
    plt.scatter(normal_data[:, 0], normal_data[:, 1], color='black')#列号0为X轴，列号1位Y轴
    plt.title('raw graph')

    # 第二幅图聚类结果
    plt.subplot(1, 2, 2)
    plt.scatter(normal_data[:, 0], normal_data[:, 1], c=cluster, marker='o')#根据聚类结果为不同的簇分配不同的颜色
    plt.title(cluster_method)
    plt.legend(labels=['x', 'y'])
    plt.show()

#图形化展示每一个样本距离相差第k的距离
def show_K_dis(data, K):
    distance = DBSCAN.k_nearest_neighbour_distance(dataSet, K)#k个近邻点
    x = [i for i in range(distance.shape[0])]
    plt.scatter(x, distance)
    plt.title("Points Sorted by Distance to %ith Nearest Neighbor" % K)
    plt.legend()
    plt.show()


scutJPG = np.load("scutVec.npy")  # 华工校徽
n_samples = 2500 #2500行噪声数据
noisy_circles, _ = datasets.make_circles(n_samples=n_samples, factor=0.5, noise=0.05,random_state=8)#圆形数据
noisy_moons, _ = datasets.make_moons(n_samples=n_samples, noise=0.05,random_state=8)#生成半环状数据
blobs, _ = datasets.make_blobs(n_samples=n_samples, random_state=8)#生成团状数据

# 1,选择不同的数据集
dataSet = scutJPG  # ['noisy_circles', 'noisy_moons', 'blobs', 'scutJPG']
# 2, 对数据进行处理
scaler = StandardScaler()
normal_data = scaler.fit_transform(dataSet)

# 3, 选择并运行聚类算法
cluster_method = "kMeans"  # ['kMeans', 'biKMeans', 'kMeans++', 'DBSCAN']

if cluster_method == "kMeans":
    # DBSCAN的参数设置与运行, 请根据K-dis图像来获得大致的eps值
    min_pts = 8#最大点是8个
    # 查看k-dist
    show_K_dis(normal_data, min_pts)
    eps = 0.08# 0.09 距离范围为0.09
    cluster = DBSCAN.dbscan(normal_data, eps, min_pts)
elif cluster_method == "biKMeans":  # 运行二分K-means算法，请指定k值
    cluster = KMeansRelevant.biKMeans(normal_data, 10)
elif cluster_method == "kMeans++":  # 运行K-means++算法，请指定k值
    cluster = KMeansRelevant.kMeansPP(normal_data, 10)
else: # 默认其他跑KMeans
    cluster = KMeansRelevant.origin_kMeans(normal_data, 10)

# 4, 输出聚类结果
num = cluster.max(0)
print("cluster num is:", num + 1)
show_cluster_result(cluster, normal_data, cluster_method)


