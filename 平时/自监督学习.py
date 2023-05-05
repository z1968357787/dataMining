import torch
import torch.nn as nn
import torch.nn.functional as F


class MaskedLanguageModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(MaskedLanguageModel, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim

        # 定义词嵌入层和预测层
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.fc1 = nn.Linear(embedding_dim, vocab_size)

    def forward(self, x, mask):
        # 对输入的文本进行掩盖
        masked_x = x * mask

        # 对掩盖后的文本进行词嵌入
        embedded = self.embedding(masked_x)

        # 预测被掩盖的单词
        output = self.fc1(embedded)
        output = F.log_softmax(output, dim=2)

        return output


import numpy as np

# 随机生成一些训练数据
vocab_size = 1000
embedding_dim = 100
corpus_size = 10000
batch_size = 64
max_len = 10

data = np.random.randint(0, vocab_size, size=(corpus_size, max_len))
mask = np.zeros((corpus_size, max_len))
for i in range(corpus_size):
    mask[i, :np.random.randint(1, max_len)] = 1

x = torch.from_numpy(data).long()
mask = torch.from_numpy(mask).long()
print(mask)

# 训练模型
model = MaskedLanguageModel(vocab_size, embedding_dim).cpu()
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

for i in range(100):
    optimizer.zero_grad()
    output = model(x.cpu(), mask.cpu())
    loss = -torch.sum(output * x.cpu().unsqueeze(-1) * mask.cpu().unsqueeze(-1))
    loss.backward()
    optimizer.step()

    print('Epoch: {}, Loss: {}'.format(i, loss.item()))