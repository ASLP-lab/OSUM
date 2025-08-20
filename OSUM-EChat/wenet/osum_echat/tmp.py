import torch
import torch.nn as nn

# 初始化词向量层
A = nn.Embedding(100, 3)  # ID范围: 0-99
B = nn.Embedding(100, 3)  # ID范围: 100-199（需确保输入ID不超过199）

# 输入数据（假设ID在0-199之间）
batch = torch.randint(0, 200, (5, 4))  # 形状: (5,18)
print(batch)

# 生成掩码
mask_a = batch < 100          # 调用A的条件
mask_b = batch >= 100         # 调用B的条件
batch_a = batch[mask_a]       # 取出A的部分
print(batch_a)
embedding_a = A(batch_a)      # 调用A的embedding
print(embedding_a)
print(embedding_a.shape)
batch_b = batch[mask_b] - 100  # 取出B的部分
print(batch_b)
embedding_b = B(batch_b)      # 调用B的embedding
print(embedding_b)
print(embedding_b.shape)

output = torch.zeros(5, 4, 3)  # 输出的形状
output[mask_a] = embedding_a  # 填充A的部分
output[mask_b] = embedding_b  # 填充B的部分
print(output)
print(output.shape)