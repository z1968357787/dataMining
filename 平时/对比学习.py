import torch
import torch.nn as nn
import numpy as np

class ContrastiveModel(nn.Module):
    def __init__(self, embedding_dim):
        super(ContrastiveModel, self).__init__()
        self.embedding_dim = embedding_dim

        # 定义两个网络，用于提取图片区域的嵌入表示
        self.encoder1 = nn.Sequential(
            nn.Linear(784, 512),
            nn.ReLU(),
            nn.Linear(512, embedding_dim),
            nn.ReLU()
        )
        self.encoder2 = nn.Sequential(
            nn.Linear(784, 512),
            nn.ReLU(),
            nn.Linear(512, embedding_dim),
            nn.ReLU()
        )

        # 定义对比损失函数
        self.contrastive_loss = nn.CosineEmbeddingLoss()

    def forward(self, x1, x2):
        # 分别提取两个网络中的嵌入表示
        x1 = self.encoder1(x1)
        #print(x1)
        x2 = self.encoder2(x2)
        #print(x2)

        # 计算正样本对和负样本对之间的相似性
        similarity = torch.cosine_similarity(x1, x2)
        #print(similarity)
        #similarity = similarity.unsqueeze(0)
        #print(similarity)

        # 生成正样本对和负样本对的标签
        label = torch.ones(similarity.shape).cpu()
        negative_label = torch.zeros(similarity.shape).cpu()

        # 计算对比损失函数
        loss = self.contrastive_loss(x1, x2, label) + self.contrastive_loss(x1, x2, negative_label)

        return loss



# 随机生成一些正样本对和负样本对
vocab_size = 10000
init_dim=784
embedding_dim = 100
batch_size = 64

data = np.random.randn(vocab_size, init_dim)
indices = np.random.choice(vocab_size, size=batch_size, replace=False)
x1 = torch.from_numpy(data[indices]).float()
x2 = torch.from_numpy(data[indices]).float()

# 训练模型
model = ContrastiveModel(embedding_dim).cpu()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for i in range(100):
    optimizer.zero_grad()
    loss = model(x1, x2)
    loss.backward()
    optimizer.step()

    print('Epoch: {}, Loss: {}'.format(i, loss.item()))