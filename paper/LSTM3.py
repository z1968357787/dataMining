import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import sklearn.datasets as sd
import sklearn.model_selection as sms


def read_dataset(X_train,y_train):

    normalize_data=X_train
    normalize_label = y_train
    #normalize_data=np.c_[normalize_data,normalize_label]
    #normalize_data = (normalize_data - np.mean(normalize_data)) / np.std(normalize_data)
    normalize_data=normalize_data[:, np.newaxis]


    #normalize_label = (normalize_label - np.mean(normalize_label)) / np.std(normalize_label)
    normalize_label = normalize_label[:, np.newaxis]
    #print(normalize_data)
    X,y = [],[]
    for i in range(len(normalize_data)):#每七天数据预测第八天数据
        _x = normalize_data[i:i + time_step,:]
        _y = normalize_label[i]
        X.append(_x.tolist())
        y.append(_y.tolist())
    # plt.figure()
    # plt.plot(data)
    # plt.show() # 以折线图展示data
    return X, y


# 实验参数设置
time_step = 1    # 用前七天的数据预测第八天
hidden_size = 6  # 隐藏层维度
lstm_layers = 1  # 网络层数
batch_size = 64  # 每一批次训练多少个样例
input_size = 13   # 输入层维度
output_size = 1  # 输出层维度
lr = 100        # 学习率


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
        #seq=seq.view(1,seq.shape[0],-1)
        (H,C)=self.init_lstm_state(seq.shape[0])
        for step in range(seq.shape[1]):

            #print(seq)
            X=seq[:,step,:]
            #print(X)
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

X, y = sd.load_svmlight_file('housing_scale.txt', n_features=13)

# 将数据集切分为训练集和验证集
X_train, X_valid, y_train, y_valid = sms.train_test_split(X, y)

# 将稀疏矩阵转为ndarray类型
X_train = X_train.toarray()
X_valid = X_valid.toarray()
#X_train = np.concatenate((np.ones((X_train.shape[0], 1)), X_train), axis=1)
#X_valid = np.concatenate((np.ones((X_valid.shape[0], 1)), X_valid), axis=1)
X_train, y_train = read_dataset(X_train,y_train)
X_test, y_test = read_dataset(X_valid,y_valid)
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
print(result)
plt.plot(range(len(y_test)), y_test, label="true_y", color="blue")#蓝线表示真实值
plt.plot(range(len(result)), result, label="pred_y", color="red")#红线表示预测值
plt.legend(loc='best')
plt.show()
