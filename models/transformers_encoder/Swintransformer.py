# models/transformers_encoder/Swintransformer.py
import torch
from torch import nn
import torch.nn.functional as F
from models.transformers_encoder.position_embedding import SinusoidalPositionalEmbedding
from models.transformers_encoder.multihead_attention import MultiheadAttention
import math

class TransformerEncoder(nn.Module):
    def __init__(self, embed_dim, num_heads=4, layers=6, attn_dropout=0.0,
                 relu_dropout=0.0, res_dropout=0.0, embed_dropout=0.0,
                 attn_mask=False, window_size=7):
        super().__init__()
        self.dropout = embed_dropout
        self.attn_dropout = attn_dropout
        self.embed_dim = embed_dim
        self.embed_scale = math.sqrt(embed_dim)
        self.embed_positions = SinusoidalPositionalEmbedding(embed_dim)
        self.attn_mask = attn_mask
        self.window_size = window_size

        self.layers = nn.ModuleList([
            SwinTransformerEncoderLayer(
                embed_dim,
                num_heads=num_heads,
                attn_dropout=attn_dropout,
                relu_dropout=relu_dropout,
                res_dropout=res_dropout,
                attn_mask=attn_mask,
                window_size=window_size
            ) for _ in range(layers)
        ])

        self.register_buffer('version', torch.Tensor([2]))
        self.normalize = True
        if self.normalize:
            self.layer_norm = LayerNorm(embed_dim)

        #  4D输入投影层
        self.input_proj = None  # 延迟初始化

    def forward(self, x_in, x_in_k=None, x_in_v=None):
        # Handle 4D input (batch, channels, seq_len, embed_dim)
        if x_in.dim() == 4:
            batch_size, channels, seq_len, embed_dim = x_in.shape
            x = x_in.permute(0, 2, 1, 3).contiguous()  # (B, L, C, E)
            x = x.view(batch_size, seq_len, channels * embed_dim)  # (B, L, C*E)

            #  延迟初始化投影层
            if self.input_proj is None:
                self.input_proj = nn.Linear(channels * embed_dim, embed_dim).to(x_in.device)
            x = self.input_proj(x)  # (B, L, E)
        else:
            batch_size, seq_len, embed_dim = x_in.shape
            x = x_in

        #  缩放 + 位置编码
        x = self.embed_scale * x
        if self.embed_positions is not None:
            pos_emb = self.embed_positions(x[:, :, 0])
            if x.size(-1) != pos_emb.size(-1):
                repeat_factor = x.size(-1) // pos_emb.size(-1)
                pos_emb = pos_emb.repeat(1, 1, repeat_factor)
            x += pos_emb
        x = F.dropout(x, p=self.dropout, training=self.training)

        #  Transformer layers
        for layer in self.layers:
            x = layer(x)

        if self.normalize:
            x = self.layer_norm(x)

        # 输出统一为 (B, E, L)
        return x.transpose(1, 2)


class SwinTransformerEncoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads=4, attn_dropout=0.1, relu_dropout=0.1,
                 res_dropout=0.1, attn_mask=False, window_size=7):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.window_size = window_size

        self.self_attn = MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            attn_dropout=attn_dropout
        )
        self.attn_mask = attn_mask

        self.relu_dropout = relu_dropout
        self.res_dropout = res_dropout
        self.normalize_before = True

        self.fc1 = Linear(embed_dim, 4 * embed_dim)
        self.fc2 = Linear(4 * embed_dim, embed_dim)
        self.layer_norms = nn.ModuleList([LayerNorm(embed_dim) for _ in range(2)])

    def forward(self, x, x_k=None, x_v=None):
        if x_k is None and x_v is None:
            x_k = x_v = x

        # Self-attention
        residual = x
        x = self.maybe_layer_norm(0, x, before=True)
        x, _ = self.self_attn(query=x, key=x_k, value=x_v, attn_mask=None)
        x = F.dropout(x, p=self.res_dropout, training=self.training)
        x = residual + x
        x = self.maybe_layer_norm(0, x, after=True)

        # FFN
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


# --- 工具函数 ---
def Linear(in_features, out_features, bias=True):
    m = nn.Linear(in_features, out_features, bias)
    nn.init.xavier_uniform_(m.weight)
    if bias:
        nn.init.constant_(m.bias, 0.)
    return m


def LayerNorm(embedding_dim):
    return nn.LayerNorm(embedding_dim)
