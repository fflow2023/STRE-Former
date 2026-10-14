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






















# ViT中使用RoPE论文的复现
import numpy as np

def get_rope_matrix(seq_length, head_dim, patch_number=(14, 14)):
    """
    summary: 生成2D RoPE矩阵

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