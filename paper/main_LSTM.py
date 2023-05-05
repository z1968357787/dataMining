import pandas as pd
import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import nn
import sklearn.datasets as sd
import sklearn.model_selection as sms
import torch.nn.functional as F

FEATURE_NUMBER = 18
HOUR_PER_DAY = 24
time_step = 14

def DataProcess(X_train,y_train):
    #df = pd.read_csv('dataset2.csv')  # 读入股票数据
    #data=np.array(df['AverageTemperature_1'])

    #normalize_data = (data - np.mean(data)) / np.std(data)  # 标准化
    normalize_data=X_train
    normalize_data = normalize_data[:, np.newaxis]  # 增加维度
    normalize_label=y_train
    normalize_label= normalize_label[:, np.newaxis]
    x_list, y_list = [], []

    for i in range(len(normalize_data)-time_step):#每七天数据预测第八天数据
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
    #print(x_list)
    x = np.float32(np.array(x_list))#设置浮点数精度为32bits
    y = np.float32(np.array(y_list))
    return x, y

class NeuralNetwork(nn.Module):
    def __init__(self, input_size):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()#允许维度变换
        self.LSTM=nn.LSTM(input_size,64,1)
        self.linear_relu_stack = nn.Sequential(
            nn.ReLU(),#激活函数
            nn.Linear(64, 1),
            # nn.ReLU(),
            # nn.Linear(1024, 1)
        )
    def forward(self, x):#forward就是专门用来计算给定输入，得到神经元网络输出的方法
        temp = self.LSTM(x)
        temp=temp[0]
        y_pred = self.linear_relu_stack(temp)
        y_pred = y_pred.squeeze()
        return y_pred

def process_predict(df):
    # df = pd.read_csv('dataset2.csv')  # 读入股票数据
    data = np.array(df['AverageTemperature_1'])

    # normalize_data = (data - np.mean(data)) / np.std(data)  # 标准化
    normalize_data = data
    normalize_data = normalize_data[:, np.newaxis]  # 增加维度

    #x_list= []

    #for i in range(len(normalize_data) - time_step):  # 每七天数据预测第八天数据
    _x = normalize_data[0:time_step]
    #x_list.append(_x.tolist())


    x = np.float32(np.array(_x.tolist()))  # 设置浮点数精度为32bits

    return x,_x.tolist()
def predict(x_list,y):
    x=x_list[1:]
    temp=[]
    temp.append(y.numpy().tolist())
    x.append(temp)
    X = np.float32(np.array(x))
    return X,x

if __name__ == '__main__':
    #df = pd.read_csv('data.csv', usecols=range(2,26)) #去2~25列
    #df = pd.read_csv('C_Data.csv')
    # 将RAINFALL的空数据用0进行填充
    #df[df == 'NR'] = 0
    X, y = sd.load_svmlight_file('housing_scale.txt', n_features=13)

    # 将数据集切分为训练集和验证集
    X_train, X_valid, y_train, y_valid = sms.train_test_split(X, y)

    # 将稀疏矩阵转为ndarray类型
    X_train = X_train.toarray()
    X_valid = X_valid.toarray()
    #X_train = np.concatenate((np.ones((X_train.shape[0], 1)), X_train), axis=1)
    #X_valid = np.concatenate((np.ones((X_valid.shape[0], 1)), X_valid), axis=1)
    x, y = DataProcess(X_train,y_train)#数据预处理
    x_test,y_test=DataProcess(X_valid,y_valid)
    # 输出（3，4）表示矩阵为3行4列
    # shape[0]输出3，为矩阵的行数
    # 同理shape[1]输出列数
    #x = x.reshape(x.shape[0], -1)#矩阵转置
    #arr.reshape(m, -1)  # 改变维度为m行、d列 （-1表示列数自动计算，d= a*b /m ）
    #np.arange(16).reshape(2, 8)  # 生成16个自然数，以2行8列的形式显示
    x = torch.from_numpy(x)#用来将数组array转换为张量Tensor（多维向量）
    y = torch.from_numpy(y)
    x=x.squeeze()
    y=y.squeeze()
    x_test = torch.from_numpy(x_test)  # 用来将数组array转换为张量Tensor（多维向量）
    y_test = torch.from_numpy(y_test)
    x_test = x_test.squeeze()
    y_test = y_test.squeeze()
    x_train=x
    y_train=y
    # 划分训练集和测试集
    #x_train = x[:3000]
    #y_train = y[:3000]
    #x_test = x[3000:]
    #y_test = y[3000:]
    
    model =  NeuralNetwork(x.shape[1])#shape[1]是获取矩阵的列数，由于是转置之后，原本是行数，样本数

    criterion = torch.nn.MSELoss(reduction='mean')#损失函数的计算方法
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)#定义SGD随机梯度下降法，学习率

    # train
    loss_train=[]
    print('START TRAIN')
    for t in range(2000):
        
        y_pred = model(x_train)

        loss = criterion(y_pred, y_train)#获取偏差
        if (t+1) % 50 == 0:
            print(t+1, loss.item())
        loss_train.append(loss.item())
        optimizer.zero_grad()#在运行反向通道之前，将梯度归零。
        loss.backward()#反向传播计算梯度，否则梯度可能会叠加计算
        optimizer.step()#更新参数
    
    # test
    with torch.no_grad():
        y_pred_test = model(x_test)
    loss_test = criterion(y_pred_test, y_test)#计算误差

    #torch.save(model.state_dict(), "model.pth")
    #print("Saved PyTorch Model State to model.pth")

    result=y_pred_test.unsqueeze(1)
    plt.plot(range(len(loss_train)), loss_train, label="training_loss", color="red")  # 红线表示预测值
    plt.legend(loc='best')
    plt.show()
    #print(result)
    plt.plot(range(len(y_test)), y_test, label="true_y", color="blue")  # 蓝线表示真实值
    plt.plot(range(len(y_pred_test)), result, label="pred_y", color="red")  # 红线表示预测值
    plt.legend(loc='best')
    plt.show()
    print('TEST LOSS:', loss_test.item())








    




