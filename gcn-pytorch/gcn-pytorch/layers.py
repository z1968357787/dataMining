import math

import torch

from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module


class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        # 请补全代码
        #输入序列
        self.in_features=in_features
        #输出序列
        self.out_features=out_features
        #初始化权重矩阵
        self.weight=Parameter(torch.FloatTensor(in_features,out_features))
        #判断是否需要偏置量
        if bias:
            self.bias=Parameter(torch.FloatTensor(out_features))
        else:
            self.reset_parameters('bias',None)
        self.reset_parameters()

    #参数初始化
    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    #模型预测
    def forward(self, input, adj):
        support=torch.mm(input,self.weight)#矩阵乘法XW
        output=torch.spmm(adj,support)#矩阵乘法AXW
        if self.bias is not None:
            return output+self.bias#输出加上偏置量
        else:
            return output
        # 请补全代码

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'
