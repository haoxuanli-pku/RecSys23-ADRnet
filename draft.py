import torch
import torch.nn as nn
import numpy as np
#
# m = nn.Linear(2, 3)
# input = torch.randn(5, 2)
# print(input)
# output = m(input)
# print(output.size())
#
# print(output)
#
# print('m.weight.shape:\n ', m.weight)
# print('m.bias.shape:\n', m.bias)

W = torch.nn.Embedding(5,3)
H = torch.nn.Embedding(4,3)

input = torch.tensor([1,0,1,0,3,1,4,5])
print(max(input))

print(W(input))

# print(H(input))

# input = torch.tensor(torch.Tensor(2,6))
# a=torch.tensor([[1.5,2,3],[2,3,4]])
# b=torch.tensor([[4,5,6]])

# a=torch.tensor([[1, 2, 3, 5,], [2,3,4,5]])   # 6*3
# W = torch.nn.Embedding(10, 3)
# print(W(a))

# print(a.clone().detach().numpy())

# torch.device('cpu'), torch.device('cuda'), torch.device('cuda:1') # 分别是访问CPU，访问第0个GPU，访问第1个GPU。第 𝑖 块GPU（ 𝑖 从0开始）
# print(torch.cuda.device_count()) # 查询可用gpu的数量。
# print(torch.cuda.is_available()) # 查询gpu是否可用
