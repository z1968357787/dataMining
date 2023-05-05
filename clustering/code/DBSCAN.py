import numpy as np
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform
import torch

#返回距离矩阵
def get_distance_mat(data_set):
    return squareform(pdist(data_set, metric='euclidean'))

# 请补全
# DBSCAN算法模型
def dbscan(data_set, eps, min_pts):
    """
    DBSCAN聚类算法,返回每个样本的cluster类别
    """
    #获得距离矩阵
    distance_mat=get_distance_mat(data_set)
    #样本数量
    example_nums=np.shape(data_set)[0]
    #获得核心点坐标索引
    #print(distance_mat)，寻找周围有min_pts个点以上的核心点能在范围内的，即为核心点,最外层的np.where返回的就是行号，即点的编号
    #核心点选取法则，在所有样本点当中，假如某一个点再周围eps范围内，有八个以上近邻点，则该点就是核心店
    core_points=np.where(np.sum(np.where(distance_mat<=eps,1,0),axis=1)>=min_pts)[0] #axis是指对列操作
    #类别标签，如果为-1表示没有被分类
    cluster=np.full(example_nums,-1)#初始将所有点都记为未分类
    k=0#初始化一个类别为0
    for core_point in core_points:
        #如果核心点没有被分类，就先将其作为种子点，并入种子集合
        if(cluster[core_point]==-1):
            cluster[core_point]=k
            #将该核心点周围eps范围内的所有点全都标为种子点，进行BFS算法
            core_point_neighbour=np.where(distance_mat[:,core_point]<=eps)[0]
            seeds=set(core_point_neighbour)#先将核心点附近的点分类

            while(len(seeds)>0): #如果存在种子点
                point_in_seeds=seeds.pop()#BFS算法开始
                cluster[point_in_seeds]=k#先将种子点分为原先的核心点的簇中
                point_in_seeds_neighbour=np.where(distance_mat[:,point_in_seeds]<=eps)[0]#搜寻该种子点周围eps范围内的所有店
                if(len(point_in_seeds_neighbour)>=min_pts):#如果种子附近的点的数量有min_pts个以上，就说明这些点都属于这个种子点的簇中，否则，则不属于这个簇
                    for point in point_in_seeds_neighbour:
                        if(cluster[point]==-1):#对于该范围内的点，如果没有被分类，就加入种子点序列中，最终也会分配给这个簇
                            seeds.add(point)

            #寻找下一个类别
            k=k+1
    return cluster

#展示每一个样本距离相差第k的距离
def k_nearest_neighbour_distance(data_set, k):
    # 获取距离矩阵
    distance_mat = get_distance_mat(data_set)

    # 对各个点，获得其第K个最近邻的距离值(可使用torch的topk方法)
    value, _ = torch.topk(torch.from_numpy(distance_mat), k+1 , largest=False)
    distance = value[:, k]
    distance = distance.numpy()

    # 排序
    distance = np.sort(distance)

    return distance
