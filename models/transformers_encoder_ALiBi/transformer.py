#models\transformers_encoder_ALiBi\transformer.py
"""
ALiBi Transformer 编码器。
与 APE 的区别：不使用 SinusoidalPositionalEmbedding（ALiBi 通过注意力偏置实现位置信息）。
"""
import torch
from torch import nn
import torch.nn.functional as F
from models.transformers_encoder_ALiBi.multihead_attention import MultiheadAttention
import math

class TransformerEncoder(nn.Module):
    """
    ALiBi Transformer encoder.
    不使用加性位置编码，位置信息通过 ALiBi 注意力偏置实现。
    """

    def __init__(self, embed_dim, num_heads, layers, attn_dropout=0.0, relu_dropout=0.0, res_dropout=0.0,
                 embed_dropout=0.0, attn_mask=False):
        super().__init__()
        self.dropout = embed_dropout
        self.attn_dropout = attn_dropout
        self.embed_dim = embed_dim
        self.embed_scale = math.sqrt(embed_dim)
        # ALiBi 不需要加性位置编码
        self.embed_positions = None
        
        self.attn_mask = attn_mask

        self.layers = nn.ModuleList([])
        for layer in range(layers):
            new_layer = TransformerEncoderLayer(embed_dim,
                                                num_heads=num_heads,
                                                attn_dropout=attn_dropout,
                                                relu_dropout=relu_dropout,
                                                res_dropout=res_dropout,
                                                attn_mask=attn_mask)
            self.layers.append(new_layer)

        self.register_buffer('version', torch.Tensor([2]))
        self.normalize = True
        if self.normalize:
            self.layer_norm = LayerNorm(embed_dim)

    def forward(self, x_in, x_in_k = None, x_in_v = None):
        """
        Args:
            x_in: 输入张量，支持3D [batch, seq_len, embed_dim] 或 4D [batch, channels, seq_len, embed_dim]
        """
        # 缩放嵌入
        x = self.embed_scale * x_in
        # ALiBi: 无位置编码加法
        x = F.dropout(x, p=self.dropout, training=self.training)

        if x_in_k is not None and x_in_v is not None:
            x_k = self.embed_scale * x_in_k
            x_v = self.embed_scale * x_in_v
            x_k = F.dropout(x_k, p=self.dropout, training=self.training)
            x_v = F.dropout(x_v, p=self.dropout, training=self.training)
        
        # 处理4D输入
        if x.dim() == 4:
            batch_size, channels, seq_len, embed_dim = x.shape
            x = x.reshape(x.shape[0], -1, x.shape[3])
            
        if x_in_k is not None and x_in_v is not None:
            if x_in_k.dim() == 4:
                x_k = x_k.reshape(x_k.shape[0], -1, x_k.shape[3])
            if x_in_v.dim() == 4:
                x_v = x_v.reshape(x_v.shape[0], -1, x_v.shape[3])
        
        # 编码器层
        intermediates = [x]
        for layer in self.layers:
            if x_in_k is not None and x_in_v is not None:
                x = layer(x, x_k, x_v)
            else:
                x = layer(x)
            intermediates.append(x)

        if self.normalize:
            x = self.layer_norm(x)
        
        # 推断时收集所有层的注意力权重
        if not self.training:
            self.all_attn_weights = [layer.last_attn_weights for layer in self.layers if hasattr(layer, 'last_attn_weights')]
            self.all_raw_attn_scores = [layer.self_attn.last_raw_attn_scores for layer in self.layers if hasattr(layer.self_attn, 'last_raw_attn_scores')]
        
        x = x.transpose(1, 2)

        return x

    def max_positions(self):
        """Maximum input length supported by the encoder."""
        return int(1e5)


class TransformerEncoderLayer(nn.Module):
    """ALiBi Encoder layer block."""

    def __init__(self, embed_dim, num_heads=4, attn_dropout=0.1, relu_dropout=0.1, res_dropout=0.1,
                 attn_mask=False):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        
        self.self_attn = MultiheadAttention(
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            attn_dropout=attn_dropout
        )
        self.attn_mask = attn_mask

        self.relu_dropout = relu_dropout
        self.res_dropout = res_dropout
        self.normalize_before = True

        self.fc1 = Linear(self.embed_dim, 4*self.embed_dim)
        self.fc2 = Linear(4*self.embed_dim, self.embed_dim)
        self.layer_norms = nn.ModuleList([LayerNorm(self.embed_dim) for _ in range(2)])

    def forward(self, x, x_k=None, x_v=None):
        residual = x
        x = self.maybe_layer_norm(0, x, before=True)
        mask = buffered_future_mask(x, x_k) if self.attn_mask else None
        if x_k is None and x_v is None:
            x, attn_w = self.self_attn(query=x, key=x, value=x, attn_mask=mask)
        else:
            x_k = self.maybe_layer_norm(0, x_k, before=True)
            x_v = self.maybe_layer_norm(0, x_v, before=True) 
            x, attn_w = self.self_attn(query=x, key=x_k, value=x_v, attn_mask=mask)
        # 推断时保存注意力权重（用于可视化）
        if not self.training:
            self.last_attn_weights = attn_w.detach().cpu()
        x = F.dropout(x, p=self.res_dropout, training=self.training)
        x = residual + x
        x = self.maybe_layer_norm(0, x, after=True)

        residual = x
        x = self.maybe_layer_norm(1, x, before=True)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=self.relu_dropout, training=self.training)
        x = self.fc2(x)
        x = F.dropout(x, p=self.res_dropout, training=self.training)
        x = residual + x
        x = self.maybe_layer_norm(1, x, after=True)
        return x

    def maybe_layer_norm(self, i, x, before=False, after=False):
        assert before ^ after
        if after ^ self.normalize_before:
            return self.layer_norms[i](x)
        else:
            return x

def fill_with_neg_inf(t):
    """FP16-compatible function that fills a tensor with -inf."""
    return t.float().fill_(float('-inf')).type_as(t)


def buffered_future_mask(tensor, tensor2=None):
    dim1 = dim2 = tensor.size(0)
    if tensor2 is not None:
        dim2 = tensor2.size(0)
    future_mask = torch.triu(fill_with_neg_inf(torch.ones(dim1, dim2)), 1+abs(dim2-dim1))
    if tensor.is_cuda:
        future_mask = future_mask.to(tensor.device)
    return future_mask[:dim1, :dim2]


def Linear(in_features, out_features, bias=True):
    m = nn.Linear(in_features, out_features, bias)
    nn.init.xavier_uniform_(m.weight)
    if bias:
        nn.init.constant_(m.bias, 0.)
    return m


def LayerNorm(embedding_dim):
    m = nn.LayerNorm(embedding_dim)
    return m
