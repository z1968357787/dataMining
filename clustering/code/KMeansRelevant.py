import numpy as np
import sys


def distEclud(vecA, vecB):
    """
    计算两个向量的欧式距离
    """
    return np.sqrt(np.sum(np.power(vecA - vecB, 2)))


# 请补全
def biKMeans(dataSet, k, distMeas=distEclud):
    """
    二分k-Means聚类算法,返回最终的k各质心和点的分配结果
    """
    m = dataSet.shape[0]  # 获取样本数量
    # 构建一个簇分配结果矩阵，共两列，第一列为样本所属的簇类值，第二列为样本到簇质心的误差
    clusterAssment = np.mat(np.zeros((m, 2)))  # m行2列

    #获取数据集中所有数据坐标的均值，生成一个初始质心，所有样本点都分配给它
    centroid0=np.mean(dataSet,axis=0).tolist()[0]
    centList=[centroid0]#可变数组长度

    #计算每一个点到质心的距离
    for j in range(m):
        clusterAssment[j,1]=distMeas(np.mat(centroid0),dataSet[j,:])**2

    #选出k个质心
    while len(centList)<k:
        lowestSSE=np.inf;#初始化误差总和为无穷大
        for i in range(len(centList)):
            pstInCurrCluster=dataSet[np.nonzero(clusterAssment[:,0].A==i)[0],:]#对于第i个簇中的所有点，全部赋值，然后二分
            centroidMat,splitClustAss=kMeans(pstInCurrCluster,2,distMeas)#对第i个簇进行K-Means算法
            sseSplit=sum(splitClustAss[:,1])#计算需要分裂的簇与不需要分裂的簇之间的误差之和
            sseNotSplit=sum(clusterAssment[np.nonzero(clusterAssment[:,0].A!=i)[0],1])
            if(sseSplit+sseNotSplit<lowestSSE):#选出效果最好的，即误差最小的
                bestCentToSplit=i #选出最适合分裂的簇号
                bestNewCents=centroidMat #两个新簇的质心
                bestClustAss=splitClustAss.copy() #复制新簇的点集归属
                lowestSSE=sseSplit+sseNotSplit #更新lowestSSE
        #更新簇编号 0更新为划分簇，1更新为新簇编号，确实要先赋值len(centList)，否则先如果先赋值bestCentToSplit，
        #当bestCentToSplit=1时，会导致可能分配的簇是1的簇又分配给len(centList)
        bestClustAss[np.nonzero(bestClustAss[:, 0].A == 1)[0], 0] = len(centList)
        bestClustAss[np.nonzero(bestClustAss[:,0].A==0)[0],0]=bestCentToSplit


        print("the bestCentToSplit is:",bestCentToSplit)
        print("the len of bestClustAss is ",len(centList))

        #增加质心
        centList[bestCentToSplit]=bestNewCents[0,:]
        centList.append(bestNewCents[1,:])

        #更新簇分配结果(更新原来分配编号是i的簇)
        clusterAssment[np.nonzero(clusterAssment[:,0].A==bestCentToSplit)[0],:]=bestClustAss
        #bestClustAss里面已经包含了簇的分配编号以及误差，分配编号以及误差同时更新

    #print(clusterAssment[:, 0])
    #print(clusterAssment[:, 0].A[:, 0])
    return clusterAssment[:,0].A[:,0]#将二维数组转换成一维数组






2
# 请补全
def kpp_initialize(dataSet, k):
    """
    用于K-Means++的初始化质心
    """
    centroids=[]#质心矩阵
    #随机选择一个质心
    centroids.append(dataSet[np.random.randint(dataSet.shape[0]),:])#随机选取一个样本作为质心

    for c_id in range(k-1):#随机生成k个质心
        dist=[]
        for i in range(dataSet.shape[0]):
            point=dataSet[i,:]#选取第i个样本，这里要对每一个样本都要遍历一次
            d=sys.maxsize
            for j in range(len(centroids)):
                temp_dist=distEclud(point,centroids[j])#第j个质心与该样本的距离
                d=min(d,temp_dist)#选取该样本与所有质心之中的最小距离的一个
            dist.append(d)#将该距离加入距离矩阵中
        dist=np.array(dist)
        next_centroid=dataSet[np.argmax(dist),:]#选取距离矩阵中距离最大的一个，通过argmax获取该最大值的下标
        centroids.append(next_centroid)#更新质心集合

    centroids=np.array(centroids)#将最终的所有质心转换成nparray并返回
    return centroids


def randCent(dataSet, k):
    """
    随机生成k个点作为质心，其中质心均在整个数据数据的边界之内
    """
    n = dataSet.shape[1]  # 获取数据的维度
    centroids = np.mat(np.zeros((k, n)))
    for j in range(n):
        minJ = np.min(dataSet[:, j])
        rangeJ = np.float(np.max(dataSet[:, j]) - minJ)
        centroids[:, j] = minJ + rangeJ * np.random.rand(k, 1)
    return centroids





def kMeans(dataSet, k, distMeas=distEclud, createCent=randCent):
    """
    k-Means聚类算法,返回最终的k各质心和点的分配结果
    """
    m = dataSet.shape[0]  # 获取样本数量
    # 构建一个簇分配结果矩阵，共两列，第一列为样本所属的簇类值，第二列为样本到簇质心的误差
    clusterAssment = np.mat(np.zeros((m, 2)))#m行2列
    # 1. 初始化k个质心
    centroids = createCent(dataSet, k)
    clusterChanged = True#判断簇分配是否继续分配
    while clusterChanged:
        clusterChanged = False
        for i in range(m):
            minDist = np.inf#无穷大，计算其到k个质心的距离
            minIndex = -1#设置该簇为未分配状态
            # 2. 找出最近的质心
            for j in range(k):
                distJI = distMeas(centroids[j, :], dataSet[i, :])#获取此样本与每一个质心之间的距离
                if distJI < minDist:
                    minDist = distJI#更新距离
                    minIndex = j#更新分配编号
            # 3. 更新每一行样本所属的簇
            if clusterAssment[i, 0] != minIndex:#判断是否有簇分配发生变化
                clusterChanged = True
            clusterAssment[i, :] = minIndex, minDist ** 2#质心，到质心的距离
        print(centroids)  # 打印质心
        # 4. 更新质心
        for cent in range(k):
            ptsClust = dataSet[np.nonzero(clusterAssment[:, 0].A == cent)[0]]
            # 获取给定簇的所有点，A应该是该下标下点的值，这里获取的是行号，np.nonzero(clusterAssment[:, 0].A == cent)返回的是矩阵对象
            centroids[cent, :] = np.mean(ptsClust, axis=0)  # 沿矩阵列的方向求均值
    return centroids, clusterAssment#centroids为所有质心的坐标，clusterAssment为每一个点所分配的质心以及距离

#createCent为生成质心的方法，K-Means++的方式选取质心
#distMeas为计算距离的方法
def kMeansPP(dataSet, k, disMeas=distEclud):
    _, clusterAssment = kMeans(dataSet, k, distMeas=disMeas, createCent=kpp_initialize)
    return clusterAssment[:, 0].A[:, 0]#这里是将二维数组clusterAssment[:, 0]转换为一维数组

#createCent为生成质心的方法，此处通过随机生成的方式生成质心
#distMeas为计算距离的方法
def origin_kMeans(dataSet, k, disMeas=distEclud):
    _, clusterAssment = kMeans(dataSet, k, distMeas=disMeas, createCent=randCent)
    return clusterAssment[:, 0].A[:, 0]#这里是将二维数组clusterAssment[:, 0]转换为一维数组

