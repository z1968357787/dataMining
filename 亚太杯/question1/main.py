import pandas as pd
import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import nn

FEATURE_NUMBER = 18
HOUR_PER_DAY = 24
time_step = 12

def DataProcess(df):
    #df = pd.read_csv('dataset2.csv')  # 读入股票数据
    data=np.array(df['AverageTemperature_1'])

    #normalize_data = (data - np.mean(data)) / np.std(data)  # 标准化
    normalize_data=data
    normalize_data = normalize_data[:, np.newaxis]  # 增加维度

    x_list, y_list = [], []

    for i in range(len(normalize_data)-time_step):#每七天数据预测第八天数据
        _x = normalize_data[i:i + time_step]
        _y = normalize_data[i + time_step]
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
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.ReLU(),#激活函数
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )

    def forward(self, x):#forward就是专门用来计算给定输入，得到神经元网络输出的方法
        y_pred = self.linear_relu_stack(x)
        y_pred = y_pred.squeeze()#这个函数主要对数据的维度进行压缩，去掉维数为1的的维度，比如是一行或者一列这种，一个一行三列（1,3）的数去掉第一个维数为一的维度之后就变成（3）行
        #y_pred本事一个1行n列的数据，squeeze后就变成了n行
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
    df = pd.read_csv('C_Data.csv')
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
    x=x.squeeze()
    y=y.squeeze()
    
    # 划分训练集和测试集
    x_train = x[:3000]
    y_train = y[:3000]
    x_test = x[3000:]
    y_test = y[3000:]
    
    model =  NeuralNetwork(x.shape[1])#shape[1]是获取矩阵的列数，由于是转置之后，原本是行数，样本数

    criterion = torch.nn.MSELoss(reduction='mean')#损失函数的计算方法
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-6)#定义SGD随机梯度下降法，学习率

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

    df2 = pd.read_csv('C_Data_prev.csv')

    prev_x,x_list= process_predict(df2)  # 数据预处理
    prev_x = torch.from_numpy(prev_x)  # 用来将数组array转换为张量Tensor（多维向量）
    prev_x = prev_x.squeeze()
    pred_y_list=[]
    for i in range(1200):
        with torch.no_grad():
            pred_y = model(prev_x)
        prev_x,x_list=predict(x_list,pred_y)
        temp=pred_y.numpy().tolist()
        if(temp>20):
            pred_y_list.append(20+temp/25)
        elif(temp<10):
            pred_y_list.append(10+temp/25)
        else:
            pred_y_list.append(temp)
        prev_x = torch.from_numpy(prev_x)  # 用来将数组array转换为张量Tensor（多维向量）
        prev_x = prev_x.squeeze()
    #print(pred_y_list)
    pred_y_list_pre_year = []
    pred_x_list_pre_year = []
    sum=0
    #print(pred_y_list)
    for i in range(len(pred_y_list)):
        sum+=pred_y_list[i]
        if((i+1)%12==0):
            pred_x_list_pre_year.append(2014+(i+1)/12)
            pred_y_list_pre_year.append(sum/12)
            sum=0

    plt.plot(pred_x_list_pre_year, pred_y_list_pre_year, label="predict", color="red")  # 红线表示预测值
    plt.legend(loc='best')
    plt.show()







    




