import torch
from torch import nn
from torch.nn import Parameter
import torch.nn.functional as F
import sys

import numpy as np
# from position_embedding import construct_rope_matrix
from models.transformers_encoder_RoPE_2D.position_embedding import get_rope_matrix
from config import USE_GPU, GPU_ID

# Code adapted from the fairseq repo.

class MultiheadAttention(nn.Module):
    """Multi-headed attention.
    See "Attention Is All You Need" for more details.
    """

    def __init__(self, embed_dim, num_heads, attn_dropout=0.,
                 bias=True, add_bias_kv=False, add_zero_attn=False, device="cuda:0"):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.attn_dropout = attn_dropout
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"
        self.scaling = self.head_dim ** -0.5

        self.in_proj_weight = Parameter(torch.Tensor(3 * embed_dim, embed_dim))
        self.register_parameter('in_proj_bias', None)
        if bias:
            self.in_proj_bias = Parameter(torch.Tensor(3 * embed_dim))
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        if add_bias_kv:
            self.bias_k = Parameter(torch.Tensor(1, 1, embed_dim))
            self.bias_v = Parameter(torch.Tensor(1, 1, embed_dim))
        else:
            self.bias_k = self.bias_v = None

        self.add_zero_attn = add_zero_attn
        self.device = device

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.in_proj_weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
        if self.in_proj_bias is not None:
            nn.init.constant_(self.in_proj_bias, 0.)
            nn.init.constant_(self.out_proj.bias, 0.)
        if self.bias_k is not None:
            nn.init.xavier_normal_(self.bias_k)
        if self.bias_v is not None:
            nn.init.xavier_normal_(self.bias_v)

    def forward(self, query, key, value, attn_mask=None):
        """
        input shape: [batch_size, seq_len, embed_dim]
        """

        qkv_same = query.data_ptr() == key.data_ptr() == value.data_ptr()
        kv_same = key.data_ptr() == value.data_ptr()
        # print(query.size(),"***")

        bsz, tgt_len, embed_dim = query.size()
        assert embed_dim == self.embed_dim
        assert list(query.size()) == [bsz, tgt_len, embed_dim]
        assert key.size() == value.size()

        aved_state = None

        if qkv_same:
            # self-attention
            q, k, v = self.in_proj_qkv(query)
        elif kv_same:
            # encoder-decoder attention
            q = self.in_proj_q(query)

            if key is None:
                assert value is None
                k = v = None
            else:
                k, v = self.in_proj_kv(key)
        else:
            q = self.in_proj_q(query)
            k = self.in_proj_k(key)
            v = self.in_proj_v(value)
        q *= self.scaling

        if self.bias_k is not None:
            assert self.bias_v is not None
            k = torch.cat([k, self.bias_k.repeat(1, bsz, 1)])
            v = torch.cat([v, self.bias_v.repeat(1, bsz, 1)])
            if attn_mask is not None:
                attn_mask = torch.cat([attn_mask, attn_mask.new_zeros(attn_mask.size(0), 1)], dim=1)

        # TEST: q.shape: [batch_size, seq_len, embedding_dim]
        # print(q.shape)        
        
        # 调整维度
        q = q.permute(1, 0, 2)  # [seq_len, batch_size, embedding_dim]
        if k is not None:
            k = k.permute(1, 0, 2)
        if v is not None:
            v = v.permute(1, 0, 2)
        

        q = q.contiguous().view(tgt_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        if k is not None:
            k = k.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        if v is not None:
            v = v.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)
            
        # print(q.shape)

        src_len = k.size(1)

        if self.add_zero_attn:
            src_len += 1
            k = torch.cat([k, k.new_zeros((k.size(0), 1) + k.size()[2:])], dim=1)
            v = torch.cat([v, v.new_zeros((v.size(0), 1) + v.size()[2:])], dim=1)
            if attn_mask is not None:
                attn_mask = torch.cat([attn_mask, attn_mask.new_zeros(attn_mask.size(0), 1)], dim=1)
        
        
        # 尝试直接从结果进行乘积，测试的时候使用的是矩阵乘积，数量级太大了，没有办法参与训练，
        # 目前先放弃这种方式，之后如果想再次尝试的话，可以尝试使用复数表示RoPE矩阵，加快运算速度
        
        # attn_weights = torch.ones(bsz * self.num_heads, q.shape[1], q.shape[1])
        
        # rope_matrices = [None] * q.shape[1]
        # for seq_index in range(q.shape[1]):
        #     rope_matrices[seq_index] = construct_rope_matrix(seq_index, q.shape[2])
            
        # print("finished!")
            
        # for head_index in range(q.shape[0]):
        #     for seq_index in range(q.shape[1]):
        #         for dim_index in range(q.shape[2]):
        #             # 构造rope_matrix: seq_index, embedding_dim --> (embedding_dim, embedding_dim)
        #             attn_weights[head_index, seq_index, seq_index] = \
        #                 q[head_index, seq_index, dim_index] * rope_matrices[seq_index][dim_index, dim_index] * k[head_index, seq_index, dim_index]
        
        
        # # gpu version
        # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # # 初始化 attn_weights
        # attn_weights = torch.ones(bsz * self.num_heads, q.shape[1], q.shape[1], device=device)
        # with torch.no_grad():
        # # 优化后的计算
        #     for head_index in range(q.shape[0]):
        #         for seq_index in range(q.shape[1]):
        #             # 构造 rope_matrix
        #             rope_matrix = construct_rope_matrix(seq_index, q.shape[2])
        #             # 使用矩阵运算替代嵌套循环
        #             attn_weights[head_index, seq_index, seq_index] = torch.sum(q[head_index, seq_index] * rope_matrix * k[head_index, seq_index])
        # attn_weights = attn_weights.cpu().detach()
        

        
        # attn_weights = torch.bmm(q, k.transpose(1, 2))
        
        # ViT中使用RoPE论文的复现
        device = q.device
        attn_weights = torch.ones(bsz * self.num_heads, q.shape[1], q.shape[1], device=device)
        R = torch.tensor(get_rope_matrix(seq_length=q.shape[1], head_dim=q.shape[2] // 2), device=device)
        for batch_head_index in range(q.shape[0]):
            # 变换输入数据
            q1 = (q[batch_head_index, :, ::2] + q[batch_head_index, :, 1::2] * 1j).to(device)
            k1 = (k[batch_head_index, :, ::2] + k[batch_head_index, :, 1::2] * 1j).to(device)
            
            # 变换
            q2 = q1 * R
            k2 = k1 * R
            
            attn_weights[batch_head_index] = torch.real(q2 @ (k2.conj()).T)
            
        assert list(attn_weights.size()) == [bsz * self.num_heads, tgt_len, src_len]

        if attn_mask is not None:
            try:
                attn_weights += attn_mask.unsqueeze(0)
            except:
                # print(attn_weights.shape)
                # print(attn_mask.unsqueeze(0).shape)
                assert False
                
        # 【可视化用】推断时保存 softmax 前的原始注意力分数（QK^T / √d_k）
        if not self.training:
            raw_scores = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            self.last_raw_attn_scores = (raw_scores.sum(dim=1) / self.num_heads).detach().cpu()

        attn_weights = F.softmax(attn_weights.float(), dim=-1).type_as(attn_weights)
        # attn_weights = F.relu(attn_weights)
        # attn_weights = attn_weights / torch.max(attn_weights)
        attn_weights = F.dropout(attn_weights, p=self.attn_dropout, training=self.training)

        attn = torch.bmm(attn_weights, v)
        assert list(attn.size()) == [bsz * self.num_heads, tgt_len, self.head_dim]

        attn = attn.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
        attn = self.out_proj(attn)

        # average attention weights over heads
        attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
        attn_weights = attn_weights.sum(dim=1) / self.num_heads
        
        # 让attn维度变化 (seq_len, batch_size, embed_dim) -> (batch_size, seq_len, embed_dim)
        attn = attn.permute(1, 0, 2)
        return attn, attn_weights

    def in_proj_qkv(self, query):
        return self._in_proj(query).unsafe_chunk(3, dim=-1)

    def in_proj_kv(self, key):
        return self._in_proj(key, start=self.embed_dim).chunk(2, dim=-1)

    def in_proj_q(self, query, **kwargs):
        return self._in_proj(query, end=self.embed_dim, **kwargs)

    def in_proj_k(self, key):
        return self._in_proj(key, start=self.embed_dim, end=2 * self.embed_dim)

    def in_proj_v(self, value):
        return self._in_proj(value, start=2 * self.embed_dim)

    def _in_proj(self, input, start=0, end=None, **kwargs):
        weight = kwargs.get('weight', self.in_proj_weight)
        bias = kwargs.get('bias', self.in_proj_bias)
        weight = weight[start:end, :]
        if bias is not None:
            bias = bias[start:end]
        return F.linear(input, weight, bias)
