import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# 读数据
# delimiter分隔符
data = np.loadtxt('./boston_house_price.csv', float, delimiter=",", skiprows=1)
#X为前13列数据，y为最后一列数据
X, y = data[:, :13], data[:, 13]
# Z-score归一化
for i in range(X.shape[1]):
    X[:, i] = (X[:, i] - np.mean(X[:, i])) / np.std(X[:, i])
# 划分训练集、测试集，将数据集的0.2作为测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
# 将训练集、测试集改为列向量的形式
y_train = y_train.reshape((-1, 1))
y_test = y_test.reshape((-1, 1))

# 初始化模型参数
def initialize_params(feature_num):
    #初始化w与b
    w=np.random.rand(feature_num,1)
    b=0
    return w,b

#定义损失函数以及梯度求解函数
def forward(X, y, w, b):
    #定义loss函数
    #w的梯度以及b的梯度
    num_train=X.shape[0]
    y_hat=np.dot(X,w)+b
    loss=np.sum((y_hat-y)**2)/num_train
    dw=np.dot(X.T,(y_hat-y))/num_train
    db=np.sum((y_hat-y))/num_train
    return y_hat,loss,dw,db


# 定义线性回归模型的训练过程
def my_linear_regression(X, y, learning_rate, epochs):
    #定义模型训练过程
    loss_his=[]#保存训练误差
    w,b=initialize_params(X.shape[1])#初始化模型参数

    #开始训练
    for i in range(epochs):
        y_hat, loss, dw, db=forward(X,y,w,b)
        #参数优化
        w+=-learning_rate*dw
        b+=-learning_rate*db

        loss_his.append(loss)

        if i%100==0:
            print(' epoch %d loss %f' %(i,loss))

    return loss_his,w,b



# 线性回归模型训练，获取数据
loss_his, w, b = my_linear_regression(X_train, y_train, 0.01, 5000)
# 打印loss曲线
plt.plot(range(len(loss_his)), loss_his, linewidth=1, linestyle="solid", label="train loss")
plt.show()
# 打印训练后得到的模型参数
print("w:", w, "\nb", b)

# 定义MSE函数
def MSE(y_test, y_pred):
    return np.sum(np.square(y_pred - y_test)) / y_pred.shape[0]

# 定义R系数函数
def r2_score(y_test, y_pred):
    # 测试集标签均值
    y_avg = np.mean(y_test)
    # 总离差平方和
    ss_tot = np.sum((y_test - y_avg) ** 2)
    # 残差平方和
    ss_res = np.sum((y_test - y_pred) ** 2)
    # R计算
    r2 = 1 - (ss_res / ss_tot)
    return r2

# 在测试集上预测
y_pred = np.dot(X_test, w) + b
# 计算测试集的MSE
print("测试集的MSE: {:.2f}".format(MSE(y_test, y_pred)))
# 计算测试集的R方系数
print("测试集的R2: {:.2f}".format(r2_score(y_test, y_pred)))