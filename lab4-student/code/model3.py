import numpy as np
import torch.nn
import torch.nn as nn
from utils import *
from torch.nn import Module
import scipy.sparse as sp


class GCN_Layer(Module):
    #初始化
    def __init__(self,inF,outF):
        super(GCN_Layer, self).__init__()
        #两个线性层
        self.W1 = torch.nn.Linear(in_features=inF,out_features=outF)#第一层参数
        self.W2 = torch.nn.Linear(in_features=inF, out_features=outF)#第二层参数

    #模型单层预测
    def forward(self,graph,selfLoop,features):
        #part1=self.W1(torch.sparse.mm(graph+selfLoop,features))#里面包含了参数矩阵W，graph对应L，(L+I)EW1
        #part2=self.W2(torch.mul(torch.sparse.mm(graph,features),features))#mm为矩阵乘法，mul为矩阵点乘，(LE*E)W
        return torch.sparse.mm(graph,features)#激活函数



class GCN(Module):
    #初始化
    def __init__(self, args, user_feature, item_feature, rating):
        super(GCN, self).__init__()
        #参数集合
        self.args = args
        #设备
        self.device = args.device
        #用户特征
        self.user_feature = user_feature
        #物品特征
        self.item_feature = item_feature
        #评价特征
        self.rating = rating
        #用户数量与物品数量
        self.num_user = rating['user_id'].max() + 1#用户
        self.num_item = rating['item_id'].max() + 1#物品
        #num_embeddings(python: int) – 词典的大小尺寸，比如总共出现5000个词，那就输入5000。此时index为（0 - 4999）
        #embedding_dim(python: int) – 嵌入向量的维度，即用多少维来表示一个符号。
        #padding_idx(python: int, optional) – 填充id，比如，输入长度为100，但是每次的句子长度并不一样，后面就需要用统一的数字填充，而这里就是指定这个数字，这样，网络在遇到填充id时，就不会计算其与其它符号的相关性。（初始化为0）
        # user embedding 32+4+2+8+18=64
        self.user_id_embedding = nn.Embedding(user_feature['id'].max() + 1, 32)#id，词典尺寸，嵌入维度大小
        self.user_age_embedding = nn.Embedding(user_feature['age'].max() + 1, 4)#年龄
        self.user_gender_embedding = nn.Embedding(user_feature['gender'].max() + 1, 2)#性别
        self.user_occupation_embedding = nn.Embedding(user_feature['occupation'].max() + 1, 8)#职业
        self.user_location_embedding = nn.Embedding(user_feature['location'].max() + 1, 18)#地理位置
        # item embedding 32+8+8+8+8=64
        self.item_id_embedding = nn.Embedding(item_feature['id'].max() + 1, 32)#id
        self.item_type_embedding = nn.Embedding(item_feature['type'].max() + 1, 8)#类型
        self.item_temperature_embedding = nn.Embedding(item_feature['temperature'].max() + 1, 8)#温度
        self.item_humidity_embedding = nn.Embedding(item_feature['humidity'].max() + 1, 8)#湿度
        self.item_windSpeed_embedding = nn.Embedding(item_feature['windSpeed'].max() + 1, 8)#风速
        # 自循环
        self.selfLoop = self.getSelfLoop(self.num_user + self.num_item)#I矩阵
        # 堆叠GCN层，本次实验只使用两层
        self.GCN_Layers = torch.nn.ModuleList()
        for _ in range(self.args.gcn_layers):
            self.GCN_Layers.append(GCN_Layer(self.args.embedSize, self.args.embedSize))
        self.graph = self.buildGraph()#构建邻接矩阵
        self.transForm = nn.Linear(in_features=self.args.embedSize * (self.args.gcn_layers + 1),
                                   out_features=self.args.embedSize)

    #构建单位矩阵I
    def getSelfLoop(self, num):
        i = torch.LongTensor(
            [[k for k in range(0, num)], [j for j in range(0, num)]])
        val = torch.FloatTensor([1] * num)
        return torch.sparse.FloatTensor(i, val).to(self.device)

    #构建拉普拉斯邻接矩阵L
    def buildGraph(self):
        #构建邻接矩阵L
        rating=self.rating.values
        #根据数据集构造R矩阵
        graph=sp.coo_matrix((rating[:,2],(rating[:,0],rating[:,1])),shape=(self.num_user,self.num_item)).tocsr()#给特征按列标上坐标，即构建R
        #sp.csr_matrix((n,m))构建n*m的0矩阵
        graph=sp.bmat([[sp.csr_matrix((graph.shape[0],graph.shape[0])),graph],[graph.T,sp.csr_matrix((graph.shape[1],graph.shape[1]))]])#构建A：0，R；R.T，0
        #拉普拉斯变换，一个[]里面相当于一列
        row_sum_sqrt= sp.diags(1/(np.sqrt(graph.sum(axis=1).A.ravel())+1e-8))#按列求和得行
        col_sum_sqrt = sp.diags(1 / (np.sqrt(graph.sum(axis=0).A.ravel()) + 1e-8))#按行求和得列
        # @表示常规的数学上定义的矩阵相乘；
        graph=row_sum_sqrt@graph@col_sum_sqrt
        #使用torch的稀疏张量表示
        graph=graph.tocoo()
        values=graph.data
        indices=np.vstack((graph.row,graph.col))
        graph=torch.sparse.FloatTensor(torch.LongTensor(indices),torch.FloatTensor(values),torch.Size(graph.shape))
        return graph.to(self.device)

    #获取模型特征
    def getFeature(self):
        # 根据用户特征获取对应的embedding
        user_id = self.user_id_embedding(torch.tensor(self.user_feature['id']).to(self.device))
        age = self.user_age_embedding(torch.tensor(self.user_feature['age']).to(self.device))
        gender = self.user_gender_embedding(torch.tensor(self.user_feature['gender']).to(self.device))
        occupation = self.user_occupation_embedding(torch.tensor(self.user_feature['occupation']).to(self.device))
        location = self.user_location_embedding(torch.tensor(self.user_feature['location']).to(self.device))
        user_emb = torch.cat((user_id, age, gender, occupation, location), dim=1)
        # 根据天气特征获取对应的embedding
        item_id = self.item_id_embedding(torch.tensor(self.item_feature['id']).to(self.device))
        item_type = self.item_type_embedding(torch.tensor(self.item_feature['type']).to(self.device))
        temperature = self.item_temperature_embedding(torch.tensor(self.item_feature['temperature']).to(self.device))
        humidity = self.item_humidity_embedding(torch.tensor(self.item_feature['humidity']).to(self.device))
        windSpeed = self.item_windSpeed_embedding(torch.tensor(self.item_feature['windSpeed']).to(self.device))
        item_emb = torch.cat((item_id, item_type, temperature, humidity, windSpeed), dim=1)
        # 拼接到一起
        concat_emb = torch.cat([user_emb, item_emb], dim=0)#按行合并
        return concat_emb.to(self.device)

    #模型预测
    def forward(self, users, items):
        features=self.getFeature()
        final_emb=features.clone()
        for GCN_Layer in self.GCN_Layers:
            features=GCN_Layer(self.graph,self.selfLoop,features)
            final_emb=torch.cat((final_emb,features.clone()),dim=1)#按列合并，前900行是用户特征样本，后1600行是物品特征样本，然后列是特征名称
        user_emb,item_emb=torch.split(final_emb,[self.num_user,self.num_item])#分离用户特征与物品特征
        user_emb=user_emb[users]#取批处理大小
        item_emb=item_emb[items]#取批处理大小
        user_emb=self.transForm(user_emb)#最后一层线性层转换
        item_emb=self.transForm(item_emb)#最后一层线性层转换
        prediction=torch.mul(user_emb,item_emb).sum(1)#模型预测，按列求和
        return prediction

