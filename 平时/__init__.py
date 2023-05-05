import torch
import torch.nn as nn
stride=2
num_blocks=4
strides = [stride] + [1] * (num_blocks - 1)
print(strides)

rnn = nn.GRU(10, 20, 2)
input = torch.randn(5, 3, 10)
h0 = torch.randn(2, 3, 20)
output, hn = rnn(input, h0)
#print(input)
#print(h0)
print(output)
print(hn)
#print(224//16)
#print(torch.randn((224 // 16) **2+1,3))