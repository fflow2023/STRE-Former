# import torch
# import torch.nn as nn
# from ViT.my_vit import vit_b_16
# # from torchvision.models.vision_transformer import vit_b_16
# class MyModel(nn.Module):
#     def __init__(self):
#         super(MyModel, self).__init__()
#         # 加载预训练的 ViT 编码器，不包含分类头，因此class number也不需要改
#         self.vit = vit_b_16(weights='IMAGENET1K_V1')
        
#         # 定义你自己的后续处理模块
#         self.custom_layer = nn.Linear(768, 512)  # 例如将 768 维的特征减少到 512 维

#     def forward(self, x):
#         # 获取 ViT 编码器输出
#         x = self.vit(x)  # 维度 (batch_size, 197, 768)
#         # print(x.shape)
        
#         # 去掉 class token，保留 196 个 patch 的特征
#         x = x[:, 1:, :]  # 维度 (batch_size, 196, 768)
        
#         # 进行你自己的处理
#         x = self.custom_layer(x)  # 维度 (batch_size, 196, 512)
        
#         return x

# # 示例：模型训练
# model = MyModel()
# optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# # 前向传播、损失计算、反向传播
# inputs = torch.randn(64, 3, 40, 300)  # 输入假定的 batch
# outputs = model(inputs)

# print("outputs shape:", outputs.shape)


"""
1D RoPE
"""

# import numpy as np

# def get_rope_matrix(seq_length, head_dim):
#     R = np.zeros((seq_length, head_dim), dtype=complex)
#     for seq_index in range(seq_length):
#         for dim_index in range(head_dim):
#             theta = 10000 ** (-1 * dim_index / head_dim)
#             R[seq_index, dim_index] = np.exp(1j * theta * seq_index)
            
#     return R
            
# seq_length = 9
# dim = 4

# q = np.ones((seq_length, dim))
# k = q

# # 产生旋转矩阵
# R = get_rope_matrix(seq_length, dim // 2)
# print(R)
# # 变换输入数据
# q1 = q[:, ::2] + q[:, 1::2] * 1j
# k1 = k[:, ::2] + k[:, 1::2] * 1j

# # 变换
# q2 = q1 * R
# k2 = k1 * R

# A = np.real(q2 @ (k2.conj()).T)
# print(A.shape)
# print(A)

# B = q @ k.T
# print(B.shape)
# print(B)



"""
2D RoPE
"""

import numpy as np

def get_rope_matrix(seq_length, head_dim, patch_number=(14, 14)):
    """_summary_: 生成2D RoPE矩阵

    Args:
        seq_length (int): 输入序列的长度
        head_dim (int): 表示每一个词向量的维度/头维度
        patch_number (tuple, optional): 一张图片横轴和纵轴对应有多少个patch. Defaults to (14, 14).
                                        patch_number[0] 对应图片的纵轴, 也就是Height
                                        patch_number[1] 对应图片的横轴, 也就是Width

    Returns:
        complex: rope_matrix
    """
    R = np.zeros((seq_length, head_dim), dtype=complex)
    for seq_index in range(seq_length):
        for dim_index in range(head_dim // 2):
            """
            px 对应图片的横轴, 也就是Width, 使用 px = dim_index % 14 来表示第n个patch的横轴坐标
            """
            px = seq_index % patch_number[1]
            theta = 100 ** (-1 * dim_index / (head_dim // 2))
            R[seq_index, 2 * dim_index] = np.exp(1j * theta * px)
        
        for dim_index in range(head_dim // 2):
            """
            py 对应图片的纵轴, 也就是Height, 使用 py = seq_index // 14 来表示第n个patch的纵轴坐标
            """
            py = seq_index // patch_number[0]
            theta = 100 ** (-1 * dim_index / (head_dim // 2))
            R[seq_index, 2 * dim_index + 1] = np.exp(1j * theta * py)
    return R


seq_length = 5
dim = 4

q = np.ones((seq_length, dim))
k = q

# 产生旋转矩阵
R = get_rope_matrix(seq_length, dim // 2)

# 变换输入数据
q1 = q[:, ::2] + q[:, 1::2] * 1j
k1 = k[:, ::2] + k[:, 1::2] * 1j

# 变换
q2 = q1 * R
k2 = k1 * R

A = np.real(q2 @ (k2.conj()).T)
print(A.shape)
# print(A)

B = q @ k.T
print(B.shape)
# print(B)