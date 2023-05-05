from __future__ import division
from __future__ import print_function

import time
import argparse
import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim
from matplotlib import pyplot as plt

from utils import load_data, accuracy
from models import GCN

# Training settings
#训练参数设置
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False,
                    help='Validate during training pass.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')#随机状态
parser.add_argument('--epochs', type=int, default=200,
                    help='Number of epochs to train.')#训练论数
parser.add_argument('--lr', type=float, default=0.01,
                    help='Initial learning rate.')#学习率
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')#初始化权重
parser.add_argument('--hidden', type=int, default=16,
                    help='Number of hidden units.')#隐藏层层数
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')#定义dropout函数

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()#判断是cpu还是gpu

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

# Load data 加载数据
adj, features, labels, idx_train, idx_val, idx_test = load_data()

# Model and optimizer
#初始化模型
model = GCN(nfeat=features.shape[1],
            nhid=args.hidden,
            nclass=labels.max().item() + 1,
            dropout=args.dropout)
#定义Adam参数优化算法
optimizer = optim.Adam(model.parameters(),
                       lr=args.lr, weight_decay=args.weight_decay)

#设定数据对应设备
if args.cuda:
    model.cuda()
    features = features.cuda()
    adj = adj.cuda()
    labels = labels.cuda()
    idx_train = idx_train.cuda()
    idx_val = idx_val.cuda()
    idx_test = idx_test.cuda()

#绘图数据存储
loss_train_his=[]
loss_valid_his=[]
acc_train_his=[]
acc_valid_his=[]

def train(epoch):
    #定义初始时间
    t=time.time()
    #将模型设置为训练模式
    model.train()
    #梯度归0
    optimizer.zero_grad()
    #模型预测
    output=model(features,adj)
    #使用NLLLOSS作为损失函数
    loss_train=F.nll_loss(output[idx_train],labels[idx_train])
    #预测准确率
    acc_train=accuracy(output[idx_train],labels[idx_train])
    #梯度归0
    loss_train.backward()
    #参数优化
    optimizer.step()

    #判断模型是否需要验证集
    if not args.fastmode:
        model.eval()
        output=model(features,adj)

    #验证集损失
    loss_val=F.nll_loss(output[idx_val],labels[idx_val])
    #验证集准确率
    acc_val=accuracy(output[idx_val],labels[idx_val])

    print('Epoch: {:04d}'.format(epoch+1),
          'loss_train: {:.4f}'.format(loss_train.item()),
          'acc_train: {:.4f}'.format(acc_train.item()),
          'loss_val: {:.4f}'.format(loss_val.item()),
          'acc_val: {:.4f}'.format(acc_val.item()),
          'time: {:.4f}'.format(time.time()-t))
    loss_train_his.append(loss_train.item())
    loss_valid_his.append(loss_val.item())
    acc_train_his.append(acc_train)
    acc_valid_his.append(acc_val)
    # 请补全代码

#模型预测
def test():
    #将模型设置为测试模式
    model.eval()
    #模型预测
    output = model(features, adj)
    #计算损失
    loss_test = F.nll_loss(output[idx_test], labels[idx_test])
    #计算准确率
    acc_test = accuracy(output[idx_test], labels[idx_test])
    print("Test set results:",
          "loss= {:.4f}".format(loss_test.item()),
          "accuracy= {:.4f}".format(acc_test.item()))


# Train model
t_total = time.time()#记录初试时间
for epoch in range(args.epochs):
    train(epoch)
print("Optimization Finished!")
print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

# Testing
test()

#绘图loss
iteration = np.arange(1, args.epochs+1, step = 1)
fig, ax = plt.subplots(figsize = (6,4))
ax.set_title('Train—Valid')
ax.set_xlabel('iteration')
ax.set_ylabel('loss')
plt.plot(iteration, loss_train_his, 'b', label='Train')
plt.plot(iteration, loss_valid_his, 'r', label='Valid')
plt.legend()
plt.show(block=True)

#绘图acc
iteration = np.arange(1, args.epochs+1, step = 1)
fig, ax = plt.subplots(figsize = (6,4))
ax.set_title('Train—Valid')
ax.set_xlabel('iteration')
ax.set_ylabel('accuracy')
plt.plot(iteration, acc_train_his, 'b', label='Train')
plt.plot(iteration, acc_valid_his, 'r', label='Valid')
plt.legend()
plt.show(block=True)

