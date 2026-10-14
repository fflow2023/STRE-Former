#models\transformers_encoder_ALiBi\multihead_attention.py
"""
ALiBi (Attention with Linear Biases) 实现
参考论文: Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation
基于 APE 的 MultiheadAttention 修改，在注意力分数上加上线性距离惩罚偏置
"""
import math
import torch
from torch import nn
from torch.nn import Parameter
import torch.nn.functional as F

class MultiheadAttention(nn.Module):
    """Multi-headed attention with ALiBi positional bias."""

    def __init__(self, embed_dim, num_heads, attn_dropout=0.,
                 bias=True, add_bias_kv=False, add_zero_attn=False):
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
        
        # 预计算 ALiBi 斜率（每个 head 不同的几何级数）
        self.slopes = self._get_alibi_slopes(num_heads)

        self.reset_parameters()

    @staticmethod
    def _get_alibi_slopes(num_heads):
        """
        按 ALiBi 论文计算每个注意力头的斜率。
        使用 2^(-8/n) 的几何级数，其中 n 为最接近 num_heads 的 2 的幂。
        """
        closest_power_of_2 = 2 ** math.floor(math.log2(num_heads))
        base = 2 ** (-(2 ** -(math.log2(closest_power_of_2) - 3)))
        slopes = [base ** (i + 1) for i in range(closest_power_of_2)]
        if closest_power_of_2 != num_heads:
            # 头数不是 2 的幂时，补充额外斜率
            extra_base = 2 ** (-(2 ** -(math.log2(2 * closest_power_of_2) - 3)))
            extra_slopes = [extra_base ** (2 * (i + 1) - 1) for i in range(num_heads - closest_power_of_2)]
            slopes = slopes + extra_slopes
        return slopes
    
    def _get_alibi_bias(self, seq_len, device):
        """
        生成 ALiBi 距离偏置矩阵。
        
        Returns:
            [num_heads, seq_len, seq_len] 的偏置矩阵，值为负数（距离越远惩罚越大）
        """
        positions = torch.arange(seq_len, device=device)
        # 计算绝对距离矩阵 [seq_len, seq_len]
        distance = positions.unsqueeze(0) - positions.unsqueeze(1)
        distance = -torch.abs(distance).float()
        # 每个 head 使用不同斜率
        slopes = torch.tensor(self.slopes, device=device, dtype=torch.float32)
        # [num_heads, seq_len, seq_len]
        alibi = slopes.unsqueeze(1).unsqueeze(1) * distance.unsqueeze(0)
        return alibi

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
        """Input shape: Time x Batch x Channel"""
        qkv_same = query.data_ptr() == key.data_ptr() == value.data_ptr()
        kv_same = key.data_ptr() == value.data_ptr()

        tgt_len, bsz, embed_dim = query.size()
        assert embed_dim == self.embed_dim
        assert list(query.size()) == [tgt_len, bsz, embed_dim]
        assert key.size() == value.size()

        aved_state = None

        if qkv_same:
            q, k, v = self.in_proj_qkv(query)
        elif kv_same:
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

        q = q.contiguous().view(tgt_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        if k is not None:
            k = k.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        if v is not None:
            v = v.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)

        src_len = k.size(1)

        if self.add_zero_attn:
            src_len += 1
            k = torch.cat([k, k.new_zeros((k.size(0), 1) + k.size()[2:])], dim=1)
            v = torch.cat([v, v.new_zeros((v.size(0), 1) + v.size()[2:])], dim=1)
            if attn_mask is not None:
                attn_mask = torch.cat([attn_mask, attn_mask.new_zeros(attn_mask.size(0), 1)], dim=1)
        
        # 计算标准注意力分数
        attn_weights = torch.bmm(q, k.transpose(1, 2))
        
        # === ALiBi: 添加线性距离偏置 ===
        alibi_bias = self._get_alibi_bias(tgt_len, attn_weights.device)
        # alibi_bias: [num_heads, tgt_len, src_len] -> [bsz*num_heads, tgt_len, src_len]
        alibi_bias = alibi_bias.repeat(bsz, 1, 1)
        attn_weights = attn_weights + alibi_bias
        
        assert list(attn_weights.size()) == [bsz * self.num_heads, tgt_len, src_len]

        if attn_mask is not None:
            try:
                attn_weights += attn_mask.unsqueeze(0)
            except:
                print(attn_weights.shape)
                print(attn_mask.unsqueeze(0).shape)
                assert False
                
        # 【可视化用】推断时保存 softmax 前的原始注意力分数（含 ALiBi 偏置）
        if not self.training:
            raw_scores = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            self.last_raw_attn_scores = (raw_scores.sum(dim=1) / self.num_heads).detach().cpu()

        attn_weights = F.softmax(attn_weights.float(), dim=-1).type_as(attn_weights)
        attn_weights = F.dropout(attn_weights, p=self.attn_dropout, training=self.training)

        attn = torch.bmm(attn_weights, v)
        assert list(attn.size()) == [bsz * self.num_heads, tgt_len, self.head_dim]

        attn = attn.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
        attn = self.out_proj(attn)

        # 对注意力头取平均
        attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
        attn_weights = attn_weights.sum(dim=1) / self.num_heads
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
