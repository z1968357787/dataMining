import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import torchvision
from torchvision import transforms

# 对数据做归一化 （-1， 1）
transform = transforms.Compose([
    transforms.ToTensor(),         # 0-1; channel, high, witch,
    transforms.Normalize(0.5, 0.5)
])

train_ds = torchvision.datasets.MNIST('data',train=True,transform=transform,download=True)
dataloader = torch.utils.data.DataLoader(train_ds, batch_size=64, shuffle=True)
imgs, _ = next(iter(dataloader))

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(100, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 28*28),
            nn.Tanh()                     # -1, 1之间
        )
    def forward(self, x):              # x 表示长度为100 的noise输入
        img = self.main(x)
        img = img.view(-1, 28, 28)
        return img
## 输入为（1， 28， 28）的图片  输出为二分类的概率值，输出使用sigmoid激活 0-1
# BCEloss计算交叉熵损失

# nn.LeakyReLU   f(x) : x>0 输出 0， 如果x<0 ,输出 a*x  a表示一个很小的斜率，比如0.1
# 判别器中一般推荐使用 LeakyReLU
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.LeakyReLU(),
            nn.Linear(512, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        x = x.view(-1, 28*28)
        x = self.main(x)
        return x

device = 'cuda' if torch.cuda.is_available() else 'cpu'
gen = Generator().to(device)
dis = Discriminator().to(device)
d_optim = torch.optim.Adam(dis.parameters(), lr=0.0001)
g_optim = torch.optim.Adam(gen.parameters(), lr=0.0001)
loss_fn = torch.nn.BCELoss()

def gen_img_plot(model, test_input):
    prediction = np.squeeze(model(test_input).detach().cpu().numpy())
    fig = plt.figure(figsize=(4, 4))
    for i in range(16):
        plt.subplot(4, 4, i+1)
        plt.imshow((prediction[i] + 1)/2)
        plt.axis('off')
    plt.show()
test_input = torch.randn(16, 100, device=device)

D_loss = []
G_loss = []

# 训练循环
for epoch in range(20):
    d_epoch_loss = 0
    g_epoch_loss = 0
    count = len(dataloader)
    for step, (img, _) in enumerate(dataloader):
        img = img.to(device)
        size = img.size(0)
        random_noise = torch.randn(size, 100, device=device)

        d_optim.zero_grad()

        real_output = dis(img)  # 判别器输入真实的图片，real_output对真实图片的预测结果
        d_real_loss = loss_fn(real_output,
                              torch.ones_like(real_output))  # 得到判别器在真实图像上的损失
        d_real_loss.backward()

        gen_img = gen(random_noise)
        # 判别器输入生成的图片，fake_output对生成图片的预测，detach用于截断梯度
        # detach()
        # 返回一个新的tensor，
        # 是从当前计算图中分离下来的，
        # 但是仍指向原变量的存放位置，
        # 其grad_fn = None且requires_grad = False，
        # 得到的这个tensor永远不需要计算其梯度，
        # 不具有梯度grad，即使之后重新将它的requires_grad置为true, 它也不会具有梯度grad。

        fake_output = dis(gen_img.detach())
        d_fake_loss = loss_fn(fake_output,
                              torch.zeros_like(fake_output))  # 得到判别器在生成图像上的损失
        d_fake_loss.backward()

        d_loss = d_real_loss + d_fake_loss
        d_optim.step()

        g_optim.zero_grad()
        fake_output = dis(gen_img)
        g_loss = loss_fn(fake_output,
                         torch.ones_like(fake_output))  # 生成器的损失
        g_loss.backward()
        g_optim.step()

        with torch.no_grad():
            d_epoch_loss += d_loss
            g_epoch_loss += g_loss

    with torch.no_grad():
        d_epoch_loss /= count
        g_epoch_loss /= count
        D_loss.append(d_epoch_loss.item())
        G_loss.append(g_epoch_loss.item())
        print('Epoch:', epoch)
        gen_img_plot(gen, test_input)