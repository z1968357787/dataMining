import pandas as pd
import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import nn

FEATURE_NUMBER = 18
HOUR_PER_DAY = 24

def DataProcess(df):
    #df = pd.read_csv('dataset2.csv')  # 读入股票数据
    data1 = np.array(df['dt'])  # 获取收盘价序列
    data2 = np.array(df['Latitude'])
    data3 = np.array(df['Longitude'])
    label=np.array(df['AverageTemperature_1'])
    # print(data)
    #data = data[::-1]  # 反转，使数据按照日期先后顺序排列
    #data2 = data2[::-1]
    #data3 = data3[::-1]
    normalize_data1=data1
    #normalize_data1 = (data - np.mean(data)) / np.std(data)  # 标准化
    normalize_data1 = normalize_data1[:, np.newaxis]  # 增加维度
    normalize_data2=data2
    #normalize_data2 = (data2 - np.mean(data2)) / np.std(data2)  # 标准化
    normalize_data2 = normalize_data2[:, np.newaxis]  # 增加维度
    normalize_data3=data3
    #normalize_data3 = (data3 - np.mean(data3)) / np.std(data3)  # 标准化
    normalize_data3 = normalize_data3[:, np.newaxis]  # 增加维度
    normalize_data = np.c_[normalize_data1, normalize_data2, normalize_data3]
    normalize_label=label
    #normalize_label=(label-np.mean(label))/np.std(label)
    x_list, y_list = [], []

    for i in range(len(normalize_data)):#每七天数据预测第八天数据
        _x = normalize_data[i,:]
        _y = normalize_label[i]
        x_list.append(_x.tolist())
        y_list.append(_y.tolist())
    """
    array = np.array(df).astype(float)#设置数据类型

    for i in range(0, array.shape[0], FEATURE_NUMBER):
        for j in range(HOUR_PER_DAY - 9):
            mat = array[i:i+18, j:j+9]
            label = array[i+9,j+9] # 用PM2.5作为标签
            x_list.append(mat)#作为自变量
            y_list.append(label)#作为因变量
    """
    x = np.float32(np.array(x_list))#设置浮点数精度为32bits
    y = np.float32(np.array(y_list))
    return x, y

class NeuralNetwork(nn.Module):
    def __init__(self, input_size):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()#允许维度变换
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(input_size, 1024),
            nn.ReLU(),#激活函数
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

    def forward(self, x):#forward就是专门用来计算给定输入，得到神经元网络输出的方法
        y_pred = self.linear_relu_stack(x)
        y_pred = y_pred.squeeze()#这个函数主要对数据的维度进行压缩，去掉维数为1的的维度，比如是一行或者一列这种，一个一行三列（1,3）的数去掉第一个维数为一的维度之后就变成（3）行
        #y_pred本事一个1行n列的数据，squeeze后就变成了n行
        return y_pred

if __name__ == '__main__':
    #df = pd.read_csv('data.csv', usecols=range(2,26)) #去2~25列
    df = pd.read_csv('dataset2.csv')
    # 将RAINFALL的空数据用0进行填充
    #df[df == 'NR'] = 0
    x, y = DataProcess(df)#数据预处理
    # 输出（3，4）表示矩阵为3行4列
    # shape[0]输出3，为矩阵的行数
    # 同理shape[1]输出列数
    #x = x.reshape(x.shape[0], -1)#矩阵转置
    #arr.reshape(m, -1)  # 改变维度为m行、d列 （-1表示列数自动计算，d= a*b /m ）
    #np.arange(16).reshape(2, 8)  # 生成16个自然数，以2行8列的形式显示
    x = torch.from_numpy(x)#用来将数组array转换为张量Tensor（多维向量）
    y = torch.from_numpy(y)
    
    # 划分训练集和测试集
    x_train = x[:840]
    y_train = y[:840]
    x_test = x[840:]
    y_test = y[840:]
    
    model =  NeuralNetwork(x.shape[1])#shape[1]是获取矩阵的列数，由于是转置之后，原本是行数，样本数

    criterion = torch.nn.MSELoss(reduction='mean')#损失函数的计算方法
    optimizer = torch.optim.SGD(model.parameters(), lr=5e-4)#定义SGD随机梯度下降法，学习率

    # train
    print('START TRAIN')
    for t in range(2000):
        
        y_pred = model(x_train)

        loss = criterion(y_pred, y_train)#获取偏差
        if (t+1) % 50 == 0:
            print(t+1, loss.item())

        optimizer.zero_grad()#在运行反向通道之前，将梯度归零。
        loss.backward()#反向传播计算梯度，否则梯度可能会叠加计算
        optimizer.step()#更新参数
    
    # test
    with torch.no_grad():
        y_pred_test = model(x_test)
    loss_test = criterion(y_pred_test, y_test)#计算误差
    #print(y_test)
    #print(y_pred_test)
    #print(y_test.item())
    #print(y_pred_test.item())
    result=y_pred_test.unsqueeze(1)
    #print(result)
    plt.plot(range(len(y_test)), y_test, label="true_y", color="blue")  # 蓝线表示真实值
    plt.plot(range(len(y_pred_test)), result, label="pred_y", color="red")  # 红线表示预测值
    plt.legend(loc='best')
    plt.show()
    print('TEST LOSS:', loss_test.item())
    




