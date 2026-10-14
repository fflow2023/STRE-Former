import torch
from torch import nn
import torch.nn.functional as F
from models.transformers_encoder_RoPE_2D.position_embedding import SinusoidalPositionalEmbedding
from models.transformers_encoder_RoPE_2D.multihead_attention import MultiheadAttention
# from position_embedding import *
# from multihead_attention import MultiheadAttention
import math

class TransformerEncoder(nn.Module):
    """
    Transformer encoder consisting of *args.encoder_layers* layers. Each layer
    is a :class:`TransformerEncoderLayer`.
    Args:
        embed_tokens (torch.nn.Embedding): input embedding
        num_heads (int): number of heads
        layers (int): number of layers
        attn_dropout (float): dropout applied on the attention weights
        relu_dropout (float): dropout applied on the first layer of the residual block
        res_dropout (float): dropout applied on the residual block
        attn_mask (bool): whether to apply mask on the attention weights
    """

    def __init__(self, embed_dim, num_heads, layers, in_channels=3, patch_size=(10, 10) ,attn_dropout=0.0, relu_dropout=0.0, res_dropout=0.0,
                 embed_dropout=0.0, attn_mask=False):
        super().__init__()
        self.dropout = embed_dropout      # Embedding dropout
        self.attn_dropout = attn_dropout
        # print("tttembed_dim",embed_dim)
        self.embed_dim = embed_dim
        self.embed_scale = math.sqrt(embed_dim)
        self.embed_positions = SinusoidalPositionalEmbedding(embed_dim)
        self.patch_size = patch_size
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
            
        self.conv_proj = nn.Conv2d(in_channels=in_channels, out_channels=embed_dim, kernel_size=10, stride=10)

    def forward(self, x_in, x_in_k = None, x_in_v = None):
        # 处理输入(n, c, h, w) -> (n, embedding_dim, n_h, h_w)
        x_in = self.conv_proj(x_in)
        if x_in_k: x_in_k = self.conv_proj(x_in_k)
        if x_in_v: x_in_v = self.conv_proj(x_in_v)
        
        # 把输入改为 (n, embedding_dim, n_h, h_w) -> (n, n_h, h_w, embedding_dim)
        x_in = x_in.permute(0, 2, 3, 1)
        if x_in_k: x_in_k = x_in_k.permute(0, 2, 3, 1)
        if x_in_v: x_in_v = x_in_v.permute(0, 2, 3, 1)
        
        # 把输入改为 (n, n_h, h_w, embedding_dim) -> (n, n_h * h_w, embedding_dim)
        # 现在的x_in 就是(batch_size, sequence_length, embedding_dim)， 从2D转为了1D
        x_in = x_in.reshape(x_in.shape[0], -1, x_in.shape[-1])
        if x_in_k: x_in_k = x_in_k.reshape(x_in_k.shape[0], -1, x_in_k.shape[-1])
        if x_in_v: x_in_v = x_in_v.reshape(x_in_v.shape[0], -1, x_in_v.shape[-1])
        
        
        
        # embed tokens and positions
        x = self.embed_scale * x_in
        # print(self.embed_dim)
        # print(x_in.shape)
        # print(x_in.transpose(0, 1).shape)
        # print(x_in.transpose(0, 1)[:, :, 0].shape)
        # print(x_in.transpose(0, 1)[:, :, 0].transpose(0, 1).shape)
        
        if self.embed_positions is not None:
            # print("123")
            # print(self.embed_positions(x_in.transpose(0, 1)[:, :, 0].transpose(0, 1)).shape)
            x += self.embed_positions(x_in.transpose(0, 1)[:, :, 0]).transpose(0, 1)   # Add positional embedding
        x = F.dropout(x, p=self.dropout, training=self.training)

        if x_in_k is not None and x_in_v is not None:
            # embed tokens and positions    
            x_k = self.embed_scale * x_in_k
            x_v = self.embed_scale * x_in_v
            if self.embed_positions is not None:
                x_k += self.embed_positions(x_in_k.transpose(0, 1)[:, :, 0]).transpose(0, 1)   # Add positional embedding
                x_v += self.embed_positions(x_in_v.transpose(0, 1)[:, :, 0]).transpose(0, 1)   # Add positional embedding
            x_k = F.dropout(x_k, p=self.dropout, training=self.training)
            x_v = F.dropout(x_v, p=self.dropout, training=self.training)
        
        # encoder layers
        intermediates = [x]
        for layer in self.layers:
            if x_in_k is not None and x_in_v is not None:
                x = layer(x, x_k, x_v)
            else:
                # print("&&&", x.shape)
                x = layer(x)
            intermediates.append(x)

        if self.normalize:
            x = self.layer_norm(x)

        # 推断时收集所有层的注意力权重
        if not self.training:
            self.all_attn_weights = [layer.last_attn_weights for layer in self.layers if hasattr(layer, 'last_attn_weights')]
            self.all_raw_attn_scores = [layer.self_attn.last_raw_attn_scores for layer in self.layers if hasattr(layer.self_attn, 'last_raw_attn_scores')]

        return x

    def max_positions(self):
        """Maximum input length supported by the encoder."""
        if self.embed_positions is None:
            return self.max_source_positions
        return min(self.max_source_positions, self.embed_positions.max_positions())


class TransformerEncoderLayer(nn.Module):
    """Encoder layer block.
    In the original paper each operation (multi-head attention or FFN) is
    postprocessed with: `dropout -> add residual -> layernorm`. In the
    tensor2tensor code they suggest that learning is more robust when
    preprocessing each layer with layernorm and postprocessing with:
    `dropout -> add residual`. We default to the approach in the paper, but the
    tensor2tensor approach can be enabled by setting
    *args.encoder_normalize_before* to ``True``.
    Args:
        embed_dim: Embedding dimension
    """

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

        self.fc1 = Linear(self.embed_dim, 4*self.embed_dim)   # The "Add & Norm" part in the paper
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


if __name__ == '__main__':
    # encoder = TransformerEncoder(300, 4, 2)
    # x = torch.rand(20, 2, 300)
    # print(encoder(x).shape)
    # print(x)
    # print("*" * 20)
    # print(encoder(x))
    
    
    batch_size = 64
    sequence_length = 300
    embedding_dim = 200
    
    height = 200
    width = 300
    in_channels = 3
    
    # 假设已经提取了图片的特征表示 (batch_size, sequence_length, embedding_dim)
    # image_features = torch.randn(batch_size, sequence_length, embedding_dim)
    
    image_features = torch.randn(batch_size, in_channels, height, width)
    
    # 创建位置嵌入
    # positional_embedding = SinusoidalPositionalEmbedding(embedding_dim)
    print(image_features.shape)
    print("****")
    # print(positional_embedding(image_features[:,:,0]).shape)
    # positioned_features = image_features + positional_embedding(image_features[:,:,0])
    positioned_features = image_features

    # 创建Transformer编码器
    encoder = TransformerEncoder(embed_dim=embedding_dim, num_heads=4, layers=6)

    # 通过Transformer编码特征
    encoded_features = encoder(positioned_features)
    
    print(encoded_features.shape)
    # print(encoded_features)

    # encoded_features 可以用于后续的分类或其他任务
