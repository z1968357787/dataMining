
import numpy as np
import pandas as pd
import sklearn.datasets as sd
import sklearn.model_selection as sms
import matplotlib.pyplot as plt
import math
import random

# 读取实验数据
X, y = sd.load_svmlight_file('housing_scale.txt',n_features = 13)

# 将数据集切分为训练集和验证集
X_train, X_valid, y_train, y_valid = sms.train_test_split(X, y)

# 将稀疏矩阵转为ndarray类型
X_train = X_train.toarray()
X_valid = X_valid.toarray()
y_train = y_train.reshape(len(y_train),1)
y_valid = y_valid.reshape(len(y_valid),1)#转化为1列



# 选取一个Loss函数，计算训练集的Loss函数值，记为loss
def compute_loss(X, y, theta):
    hx = X.dot(theta)#w点乘X(矩阵乘法)
    error = np.power((hx - y), 2).mean()
    return hx,error

#让自变量添加一列，作为偏置量，即b，然后统一处理
X_train = np.concatenate((np.ones((X_train.shape[0],1)), X_train), axis = 1)
X_valid = np.concatenate((np.ones((X_valid.shape[0],1)), X_valid), axis = 1)

# 闭式解函数
def normal_equation(X, y):
    return (np.linalg.inv(X.T.dot(X))).dot(X.T).dot(y)

theta = normal_equation(X_train, y_train)

hx,loss_train = compute_loss(X_train, y_train, theta)


hv,loss_valid = compute_loss(X_valid, y_valid, theta)

print(loss_train)
print(loss_valid)
plt.plot(range(len(y_valid)), y_valid, label="true_y", color="blue")
plt.plot(range(len(hv)), hv, label="pred_y", color="red")  # 红线表示预测值
plt.show()

##ques2


#梯度函数
def gradient(X, y, theta):
    #np.dot(X,theta)
    return X.T.dot(X.dot(theta) - y)



#随机梯度下降
# @X 训练集自变量
# @Y 训练集因变量
# @theta 模型权重系数
# @alpha 学习率
# @iters 迭代次数
# @X_valid 验证集自变量
# @y_valid 验证集因变量
def random_descent(X, y, theta, alpha, iters, X_valid, y_valid):
    # n个特征
    n=X.shape#行为样本，列为特征
    loss_train = np.zeros((iters,1))#初始化训练集损失数组
    loss_valid = np.zeros((iters,1))#初始化验证集损失数组
    loss=float('inf')
    hv_loss=[]
    for i in range(iters):
        #随机选择一个样本
        num=np.random.randint(n,size=1)#选择一个样本，包含所有特征
        x_select=X[num,:]#获取该样本的所有特征值
        y_select=y[num,0]#获取该样本的标签
        grad = gradient(x_select, y_select, theta)#进行梯度计算
        theta = theta - alpha * grad#模型参数优化
        hx,loss_train[i] = compute_loss(X, y, theta)#记录训练集中每一次迭代的损失
        hv,loss_valid[i] = compute_loss(X_valid, y_valid, theta)#记录验证集中每一次迭代的损失
        if loss>loss_valid[i]:
            loss=loss_valid[i]
            hv_loss=hv

    return theta, loss_train, loss_valid,hv_loss



#全梯度下降
#随机梯度下降
# @X 训练集自变量
# @Y 训练集因变量
# @theta 模型权重系数
# @alpha 学习率
# @iters 迭代次数
# @X_valid 验证集自变量
# @y_valid 验证集因变量
def descent(X, y, theta, alpha, iters, X_valid, y_valid):
    loss_train = np.zeros((iters,1))#初始化训练集损失数组
    loss_valid = np.zeros((iters,1))#初始化验证集损失数组
    loss = float('inf')
    hv_loss = []
    for i in range(iters):#全部样本都进行一次迭代
        grad = gradient(X, y, theta)#对每一个样本都进行梯度计算
        theta = theta - alpha * grad#参数优化
        hx,loss_train[i] = compute_loss(X, y, theta)#记录训练集中每一次迭代的损失
        hv,loss_valid[i] = compute_loss(X_valid, y_valid, theta)#记录验证集中每一次迭代的损失
        if loss>loss_valid[i]:
            loss=loss_valid[i]
            hv_loss=hv
    return theta, loss_train, loss_valid,hv_loss


# 线性模型参数初始化，可以考虑全零初始化，随机初始化或者正态分布初始化。
theta = np.zeros((14, 1))

# 随机梯度下降
alpha = 1e-3
iters = 30
opt_theta, loss_train, loss_valid,hv = random_descent(X_train, y_train, theta, alpha, iters, X_valid, y_valid)
plt.plot(range(len(y_valid)), y_valid, label="true_y", color="blue")
plt.plot(range(len(hv)), hv, label="pred_y", color="red")  # 红线表示预测值
plt.legend(loc='best')
plt.show()
#选取矩阵中最小的值
print(loss_train.min())
print(loss_valid.min())



iteration = np.arange(0, iters, step = 1)
fig, ax = plt.subplots(figsize = (12,8))
ax.set_title('Train')
ax.set_xlabel('iteration')
ax.set_ylabel('loss')
plt.plot(iteration, loss_train, 'b', label='Train')
#plt.plot(iteration, loss_valid, 'r', label='Valid')
plt.legend()
plt.show()


# 线性模型参数初始化，可以考虑全零初始化，随机初始化或者正态分布初始化。
theta = np.zeros((14, 1))#14行1列

# 全批量梯度下降
alpha = 0.0001
iters = 30
opt_theta, loss_train, loss_valid,hv2 = descent(X_train, y_train, theta, alpha, iters, X_valid, y_valid)
#选取矩阵中最小的值
#plt.plot(range(len(y_valid)), y_valid, label="true_y", color="blue")
#plt.plot(range(len(hv2)), hv2, label="pred_y", color="red")  # 红线表示预测值
#plt.legend(loc='best')
#plt.show()
print(loss_train.min())
print(loss_valid.min())



iteration = np.arange(0, iters, step = 1)
fig, ax = plt.subplots(figsize = (12,8))
ax.set_title('Train')
ax.set_xlabel('iteration')
ax.set_ylabel('loss')
plt.plot(iteration, loss_train, 'b', label='Train')
# plt.plot(iteration, loss_valid, 'r', label='Valid')
plt.legend()
plt.show()
