import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


def read_dataset(dataset_type):
    assert dataset_type == "train" or dataset_type == "test"
    df = pd.read_csv('C_Data_{}.csv'.format(dataset_type),sep=',')  # 读入股票数据
    #df['AverageTemperature_1']=(df['AverageTemperature_1'])
    data = np.array(df['AverageTemperature_1'])  # 获取收盘价序列
    #data.astype(float)
    #data = data[::-1]  # 反转，使数据按照日期先后顺序排列
    #print(data)
    #mixed=[]
    #for i in list(data):
    #    mixed.append(float(i))
    #data = np.array(mixed, dtype=float)
    #normalize_data = (data - np.mean(data)) / np.std(data)  # 标准化
    normalize_data=data
    normalize_data = normalize_data[:, np.newaxis]  # 增加维度
    X, y = [], []
    for i in range(len(normalize_data) - time_step):#每七天数据预测第八天数据
        _x = normalize_data[i:i + time_step]
        _y = normalize_data[i + time_step]
        X.append(_x.tolist())
        y.append(_y.tolist())
    # plt.figure()
    # plt.plot(data)
    # plt.show() # 以折线图展示data
    return X, y

def process_predict(df):
    # df = pd.read_csv('dataset2.csv')  # 读入股票数据
    data = np.array(df['AverageTemperature_1'])

    # normalize_data = (data - np.mean(data)) / np.std(data)  # 标准化
    normalize_data = data
    normalize_data = normalize_data[:, np.newaxis]  # 增加维度

    x_list= []
    y_list=[]
    #for i in range(len(normalize_data) - time_step):  # 每七天数据预测第八天数据
    _x = normalize_data[0:time_step]
    _y=normalize_data[0]
    x_list.append(_x.tolist())
    y_list.append(_y.tolist())

    #x = np.float32(np.array(_x.tolist()))  # 设置浮点数精度为32bits

    return x_list,y_list,_x.tolist()
def predict(x_list,y):
    x=x_list[1:]
    temp=[]
    temp.append(y.item())
    x.append(temp)
    x_list = []
    y_list = []
    x_list.append(x)
    y_list.append(temp)
    #X = np.float32(np.array(x))
    return x_list,y_list,x

# 实验参数设置
time_step = 12    # 用前七天的数据预测第八天
hidden_size = 4  # 隐藏层维度
lstm_layers = 1  # 网络层数
batch_size = 64  # 每一批次训练多少个样例
input_size = 1   # 输入层维度
output_size = 1  # 输出层维度
lr = 0.05        # 学习率


class myDataset(Dataset):
    #导入数据
    def __init__(self, x, y):
        self.x = x
        self.y = y

    #获取相应坐标的数据
    def __getitem__(self, index):
        return torch.Tensor(self.x[index]), torch.Tensor(self.y[index])

    #返回X数据的长度
    def __len__(self):
        return len(self.x)


class LSTM(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, device):
        super(LSTM, self).__init__()

        #定义输入数据集维度，输出数据集维度，隐藏层数据集维度，模型运行设备
        self.input_size=input_size
        self.ouput_size=output_size
        self.hidden_size=hidden_size
        self.device=device

        def _one(a,b):
            #先类型转换，将list ,numpy转化为tensor
            # torch.nn.Parameter()
            # 理解为类型转换函数，将一个不可训练的类型Tensor转换成可以训练的类型parameter，
            # 并将这个parameter绑定到这个module里面，
            # 经过类型转换这个 self.v 变成了模型的一部分，
            # 成为模型中根据训练可以改动的参数
            return nn.Parameter(torch.FloatTensor(a,b).to(self.device))

        def _three():
            return (_one(input_size,hidden_size),_one(hidden_size,hidden_size),nn.Parameter(torch.zeros(hidden_size).to(self.device)))

        #初始化参数
        self.W_xi, self.W_hi, self.b_i = _three()#输入门参数
        self.W_xf, self.W_hf, self.b_f = _three()#遗忘门参数
        self.W_xo, self.W_ho, self.b_o = _three()#隐藏层输出参数
        self.W_xc, self.W_hc, self.b_c = _three()#候选的细胞参数

        #初始化输出层参数
        self.W_hq=_one(hidden_size,output_size)
        self.b_q=nn.Parameter(torch.zeros(output_size).to(self.device))

        #创建梯度
        self.params=[self.W_xi,self.W_hi,self.b_i,self.W_xf, self.W_hf, self.b_f,self.W_xo, self.W_ho, self.b_o,self.W_xc, self.W_hc, self.b_c,self.W_hq,self.b_q]

        for param in self.params:
            if param.dim()==2:
                #使初始化的参数呈正太分布
                nn.init.xavier_normal(param)

    def init_lstm_state(self, batch_size):
        #初始化输入参数
        return (torch.zeros((batch_size,self.hidden_size),device=self.device),
                torch.zeros((batch_size,self.hidden_size),device=self.device))

    #用于返回预测结果
    def forward(self, seq):
        #初始化输入参数
        (H,C)=self.init_lstm_state(seq.shape[0])
        for step in range(seq.shape[1]):
            #print(seq.shape[0])
            #print(seq.shape[1])
            X=seq[:,step,:]
            #模型计算
            #@ 和 * 代表矩阵的两种相乘方式：
            # @表示常规的数学上定义的矩阵相乘；
            # *表示两个矩阵对应位置处的两个元素相乘
            #LSTM神经网络正向传播计算
            I = torch.sigmoid((X@self.W_xi)+(H@self.W_hi)+self.b_i)
            F = torch.sigmoid((X @ self.W_xf) + (H @ self.W_hf) + self.b_f)
            O = torch.sigmoid((X @ self.W_xo) + (H @ self.W_ho) + self.b_o)
            #matmul类似于矩阵相乘
            C_tilda = torch.tanh(torch.matmul(X.float(),self.W_xc)+torch.matmul(H.float(),self.W_hc)+self.b_c)
            C=F*C+I*C_tilda
            H=O*torch.tanh(C)
        #最终预测结果
        Y=(H@self.W_hq)+self.b_q
        return Y,(H,C)

X_train, y_train = read_dataset('train')
X_test, y_test = read_dataset('test')
train_dataset = myDataset(X_train, y_train)
test_dataset = myDataset(X_test, y_test)
train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, 1)

# 设定训练轮数
num_epochs = 50
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
hist = np.zeros(num_epochs)#用于记录模型每一次迭代的预测误差
model = LSTM(input_size, output_size, hidden_size, device)
# 定义优化器和损失函数
optimiser = torch.optim.Adam(model.parameters(), lr=lr)  # 使用Adam优化算法
loss_func = torch.nn.MSELoss(reduction='mean')  # 使用均方差作为损失函数
for epoch in range(num_epochs):
    epoch_loss = 0
    for i, data in enumerate(train_loader):
        X, y = data
        pred_y, _ = model(X.to(device))#预测结果
        loss = loss_func(pred_y, y.to(device))#获取损失
        #print(len(pred_y))
        #print(len(y))
        optimiser.zero_grad()#将梯度归0
        loss.backward()#计算梯度
        optimiser.step()#参数优化
        epoch_loss += loss.item()#统计损失
    print("Epoch ", epoch, "MSE: ", epoch_loss)
    hist[epoch] = epoch_loss#保存误差
plt.plot(hist)#绘图每一次迭代的损失
plt.show()#展示绘图结果

# 测试
model.eval()#模型预测
result = []#保存预测结果
for i, data in enumerate(test_loader):
    X, y = data
    pred_y, _ = model(X.to(device))
    result.append(pred_y.item())
#print(pred_y.item())

plt.plot(range(len(y_test)), y_test, label="true_y", color="blue")#蓝线表示真实值
plt.plot(range(len(result)), result, label="pred_y", color="red")#红线表示预测值
plt.legend(loc='best')
plt.show()

df2 = pd.read_csv('C_Data_prev.csv')

prev_x,pred_y,x_list= process_predict(df2)  # 数据预处理
pred_y_list=[]

def predict_data(test_loader2):
    for i, data in enumerate(test_loader2):
        X, y = data
        #print(X)
        #print(y)
        pred_y, _ = model(X.to(device))
        return pred_y

test_dataset2 = myDataset(prev_x, pred_y)
test_loader2 = DataLoader(test_dataset2, 1)
for i in range(1200):

    with torch.no_grad():
        pred_y = predict_data(test_loader2)
    prev_x,pred_y_temp,x_list=predict(x_list,pred_y)
    pred_y_list.append(pred_y.item())
    test_dataset2 = myDataset(prev_x, pred_y_temp)
    test_loader2 = DataLoader(test_dataset2, 1)

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