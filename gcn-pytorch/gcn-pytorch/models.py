import torch.nn as nn
import torch.nn.functional as F
from layers import GraphConvolution


class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN, self).__init__()
        #第一层GCN
        self.gc1 = GraphConvolution(nfeat, nhid)
        #第二层GCN
        self.gc2 = GraphConvolution(nhid, nclass)
        # 防止过拟合
        self.dropout = dropout
    #函数预测
    def forward(self, x, adj):
        #第一层预测
        x=F.relu(self.gc1(x,adj))
        #减少过拟合
        x=F.dropout(x,self.dropout,training=self.training)
        #第二层预测
        x=self.gc2(x,adj)
        #最后softmax
        return F.log_softmax(x,dim=1)
        # 请补全代码

