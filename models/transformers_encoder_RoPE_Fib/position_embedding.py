#models\transformers_encoder_RoPE_Fib\position_embedding.py
import math

import torch
import torch.nn as nn

# Code adapted from the fairseq repo.

def make_positions(tensor, padding_idx, left_pad):
    """Replace non-padding symbols with their position numbers.
    Position numbers begin at padding_idx+1.
    Padding symbols are ignored, but it is necessary to specify whether padding
    is added on the left side (left_pad=True) or right side (left_pad=False).
    """
    max_pos = padding_idx + 1 + tensor.size(1)
    device = tensor.get_device()
    buf_name = f'range_buf_{device}'
    if not hasattr(make_positions, buf_name):
        setattr(make_positions, buf_name, tensor.new())
    setattr(make_positions, buf_name, getattr(make_positions, buf_name).type_as(tensor))
    if getattr(make_positions, buf_name).numel() < max_pos:
        torch.arange(padding_idx + 1, max_pos, out=getattr(make_positions, buf_name))
    mask = tensor.ne(padding_idx)
    positions = getattr(make_positions, buf_name)[:tensor.size(1)].expand_as(tensor)
    if left_pad:
        positions = positions - mask.size(1) + mask.long().sum(dim=1).unsqueeze(1)
    new_tensor = tensor.clone()
    return new_tensor.masked_scatter_(mask, positions[mask]).long()


class SinusoidalPositionalEmbedding(nn.Module):
    """This module produces sinusoidal positional embeddings of any length.
    Padding symbols are ignored, but it is necessary to specify whether padding
    is added on the left side (left_pad=True) or right side (left_pad=False).
    """

    def __init__(self, embedding_dim, padding_idx=0, left_pad=0, init_size=128):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        self.left_pad = left_pad
        self.weights = dict()   # device --> actual weight; due to nn.DataParallel :-(
        self.register_buffer('_float_tensor', torch.FloatTensor(1))

    @staticmethod
    def get_embedding(num_embeddings, embedding_dim, padding_idx=None):
        """Build sinusoidal embeddings.
        This matches the implementation in tensor2tensor, but differs slightly
        from the description in Section 3.5 of "Attention Is All You Need".
        """
        half_dim = embedding_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float) * -emb)
        emb = torch.arange(num_embeddings, dtype=torch.float).unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1).view(num_embeddings, -1)
        if embedding_dim % 2 == 1:
            # zero pad
            emb = torch.cat([emb, torch.zeros(num_embeddings, 1)], dim=1)
        if padding_idx is not None:
            emb[padding_idx, :] = 0
        return emb

    def forward(self, input):
        """Input is expected to be of size [bsz x seqlen]."""
        bsz, seq_len = input.size()
        max_pos = self.padding_idx + 1 + seq_len
        device = input.get_device()
        if device not in self.weights or max_pos > self.weights[device].size(0):
            # recompute/expand embeddings if needed
            self.weights[device] = SinusoidalPositionalEmbedding.get_embedding(
                max_pos,
                self.embedding_dim,
                self.padding_idx,
            )
        self.weights[device] = self.weights[device].type_as(self._float_tensor).to(input.device)
        positions = make_positions(input, self.padding_idx, self.left_pad)
        # return self.weights[device].index_select(0, positions.view(-1)).view(bsz, seq_len, -1).detach()
        return self.weights[device].index_select(0, positions.reshape(-1)).reshape(bsz, seq_len, -1).detach()

    def max_positions(self):
        """Maximum number of supported positions."""
        return int(1e5)  # an arbitrary large number



























# customized_trial

# import numpy as np

# def construct_rope_matrix(m, d):
#     """
#     Constructs a d x d rotation matrix R(θ, m).
    
#     Args:
#         m (int): The m value that influences the rotation matrix.
#         d (int): The dimension of the rotation matrix.
    
#     Returns:
#         numpy.ndarray: A d x d rotation matrix.
#     """
#     assert d % 2 == 0, "Dimension d must be even."
    
#     R = np.eye(d)
    
#     theta_length = d // 2
#     theta = np.ones(theta_length)
#     for i in range(theta_length):
#         theta[i] = 10000 ** (-2 * i // d)
#         # print(i, ': \t', theta[i])
    
    
#     for i in range(0, d, 2):
#         theta_index = i // 2
#         cos_theta = np.cos(m * theta[theta_index])
#         sin_theta = np.sin(m * theta[theta_index])
        
#         R[i, i] = cos_theta
#         R[i + 1, i + 1] = cos_theta
#         R[i, i + 1] = -sin_theta
#         R[i + 1, i] = sin_theta
#     return R


# if __name__ == "__main__":
#     # Example usage:
#     m = 1  # Example m value
#     d = 4  # Example dimension (must be even)

#     rope_matrix = construct_rope_matrix(m, d)
#     print(rope_matrix)





PATCH_NUMBER_MAX = 100000
TARGET_NUMBER = 10






import numpy as np

# 构造对 2*pi 取余 的Fibonacci数列
# NOTE:patch的最大长度为 PATCH_NUMBER_MAX ,直接算好，避免重复计算
Fib_length = PATCH_NUMBER_MAX
Fib = [1.0, 1.0]
for i in range(Fib_length - 2):
    value = Fib[i] + Fib[i + 1]
    value -= 1.0 * value // 2 // np.pi * 2 * np.pi
    Fib.append(value)



def print_matrix(matrix, target_number):
    for i in range(len(matrix)):
        for j in range(len(matrix[i])):
            # print(matrix[i][j], end = " \t\t")
            print([target_number + 1 - k for k in matrix[i][j]], end = " ")
        print()

def get_possible_matrix(patch_number, target_number=10):    
    # 初始化参数
    MACRO = (5 ** 0.5 - 1) / 2
    height, width = patch_number
    h_max = height
    w_max = width
    h_min = 1
    w_min = 1
    index = 0
    stop = False
    
    possible = [[[] for i in range(width)] for j in range(height)]
    
    # print(MACRO)

    while stop == False:
        if index % 4 == 0:
            w_max_copy = w_max
            w_max = int(np.floor(w_min + (w_max - w_min + 1)* MACRO))
            for i in range(w_min, w_max + 1):
                for j in range(h_min, h_max + 1):
                    possible[j-1][i-1].append(index)
            w_min = w_max
            w_max = w_max_copy
            
        elif index % 4 == 1:
            h_max_copy = h_max
            h_max = int(np.floor(h_min + (h_max - h_min + 1)* MACRO))
            for i in range(w_min, w_max + 1):
                for j in range(h_min, h_max + 1):
                    possible[j-1][i-1].append(index)
            h_min = h_max
            h_max = h_max_copy
        
        elif index % 4 == 2:
            w_min_copy = w_min
            w_min = int(np.floor(w_min + (w_max - w_min + 1)* (1 - MACRO)))
            for i in range(w_min, w_max + 1):
                for j in range(h_min, h_max + 1):
                    possible[j-1][i-1].append(index)
            w_max = w_min
            w_min = w_min_copy
        
        elif index % 4 == 3:
            h_min_copy = h_min
            h_min = int(np.floor(h_min + (h_max - h_min + 1)* (1 - MACRO)))
            for i in range(w_min, w_max + 1):
                for j in range(h_min, h_max + 1):
                    possible[j-1][i-1].append(index)
            h_max = h_min
            h_min = h_min_copy
            
        for i in range(width + 1):
            for j in range(height + 1):
                if len(possible[j-1][i-1]) >= target_number:
                    stop = True
                    break
        
        index += 1
        
    # 检查 possible 矩阵的值
    # print_matrix(possible, target_number)
    
    return possible




"""
1D RoPE for Fib
"""

import numpy as np
import random

def get_rope_matrix(seq_length, head_dim, patch_number=(20, 30), target_number=TARGET_NUMBER, use_fts=True):
    """
    生成 Fibonacci 位置编码的旋转矩阵。
    
    Args:
        use_fts: 是否启用 FTS（Fibonacci Target Shutter）。
                 True: 使用黄金螺线散布的 Patch 索引（完整 FPE）
                 False: 使用顺序索引直接查 Fibonacci 数列（消融基线）
    """
    R = np.zeros((seq_length, head_dim), dtype=complex)
    
    choices = np.ones(patch_number, dtype=int)
    
    # FTS 消融开关：关闭时跳过黄金螺线矩阵计算
    if use_fts:
        possible = get_possible_matrix(patch_number, target_number)
    else:
        possible = None
    
    for seq_index in range(seq_length):
        px = seq_index % patch_number[1]
        py = seq_index // patch_number[1]
        
        if use_fts and possible is not None:
            # 完整 FPE：从 FTS 黄金螺线矩阵中随机选择索引
            Fib_index_list = possible[py][px]
            Fib_index = random.choice(Fib_index_list)
        else:
            # FTS 消融：直接使用顺序索引查 Fibonacci 数列
            Fib_index = seq_index
        
        choices[py][px] = Fib_index
        Fib_value = Fib[Fib_index]
        
        for dim_index in range(head_dim):
            theta = 10000 ** (-1 * dim_index / head_dim)
            R[seq_index, dim_index] = np.exp(1j * theta * Fib_value)
            

    # 打印当前的选择
    # print("*" * 20 , "   choices   " , "*" * 20)
    # for i in range(patch_number[0]):
    #     for j in range(patch_number[1]):
    #         print(choices[i][j], end = " ")
    #     print()
    
    
    return R
















# 下面是RoPE的编码方式，为了测试方便，也放在下面了：

# # ViT中使用RoPE论文的复现
# import numpy as np

# def get_rope_matrix(seq_length, head_dim, patch_number=(14, 14)):
#     """
#     summary: 生成2D RoPE矩阵

#     Args:
#         seq_length (int): 输入序列的长度
#         head_dim (int): 表示每一个词向量的维度/头维度
#         patch_number (tuple, optional): 一张图片横轴和纵轴对应有多少个patch. Defaults to (14, 14).
#                                         patch_number[0] 对应图片的纵轴, 也就是Height
#                                         patch_number[1] 对应图片的横轴, 也就是Width

#     Returns:
#         complex: rope_matrix
#     """
#     R = np.zeros((seq_length, head_dim), dtype=complex)
#     for seq_index in range(seq_length):
#         for dim_index in range(head_dim // 2):
#             """
#             px 对应图片的横轴, 也就是Width, 使用 px = dim_index % 14 来表示第n个patch的横轴坐标
#             """
#             px = seq_index % patch_number[1]
#             theta = 100 ** (-1 * dim_index / (head_dim // 2))
#             R[seq_index, 2 * dim_index] = np.exp(1j * theta * px)
        
#         for dim_index in range(head_dim // 2):
#             """
#             py 对应图片的纵轴, 也就是Height, 使用 py = seq_index // 14 来表示第n个patch的纵轴坐标
#             """
#             py = seq_index // patch_number[0]
#             theta = 100 ** (-1 * dim_index / (head_dim // 2))
#             R[seq_index, 2 * dim_index + 1] = np.exp(1j * theta * py)
#     return R















if __name__ == "__main__":
    get_rope_matrix(25, 5, patch_number=(5,5))