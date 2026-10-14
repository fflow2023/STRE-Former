import torch
from torch import nn
import torch.nn.functional as F
from models.transformers_encoder.position_embedding import SinusoidalPositionalEmbedding
from models.transformers_encoder.multihead_attention import MultiheadAttention
# from position_embedding import SinusoidalPositionalEmbedding
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

    def __init__(self, embed_dim, num_heads, layers, attn_dropout=0.0, relu_dropout=0.0, res_dropout=0.0,
                 embed_dropout=0.0, attn_mask=False):
        super().__init__()
        self.dropout = embed_dropout      # Embedding dropout
        self.attn_dropout = attn_dropout
        # print("tttembed_dim",embed_dim)
        self.embed_dim = embed_dim
        self.embed_scale = math.sqrt(embed_dim)
        self.embed_positions = SinusoidalPositionalEmbedding(embed_dim)
        
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
            x_in (FloatTensor): embedded input of shape `(src_len, batch, embed_dim)`
            x_in_k (FloatTensor): embedded input of shape `(src_len, batch, embed_dim)`
            x_in_v (FloatTensor): embedded input of shape `(src_len, batch, embed_dim)`
        Returns:
            dict:
                - **encoder_out** (Tensor): the last encoder layer's output of
                  shape `(src_len, batch, embed_dim)`
                - **encoder_padding_mask** (ByteTensor): the positions of
                  padding elements of shape `(batch, src_len)`
        """
        # embed tokens and positions
        
        # torch.Size([16, 3, 256, 384])
        x = self.embed_scale * x_in

        if self.embed_positions is not None:
            # 如果输入是四维的，调整为三维以计算位置嵌入
            if x_in.dim() == 4:
                bsz, channels, seq_len, embed_dim = x_in.size()
                # 只需要提取 2D (batch_size, seq_len) 的形状，切片获取即可，不使用 squeeze 避免单维崩溃
                x_in_reshaped = x_in[:, 0, :, 0]
                pos_emb = self.embed_positions(x_in_reshaped)
                pos_emb = pos_emb.unsqueeze(1).expand(-1, channels, -1, -1)
            else:
                pos_emb = self.embed_positions(x_in[:, :, 0]) if x_in.dim() == 3 else self.embed_positions(x_in)

            x += pos_emb
        x = F.dropout(x, p=self.dropout, training=self.training)

        if x_in_k is not None and x_in_v is not None:
            # 如果输入是四维的，调整为三维以计算位置嵌入
            if x_in_k.dim() == 4:
                bsz, channels, seq_len, embed_dim = x_in_k.size()
                x_in_k_reshaped = x_in_k[:, 0, :, 0]
                pos_emb_k = self.embed_positions(x_in_k_reshaped)
                pos_emb_k = pos_emb_k.unsqueeze(1).expand(-1, channels, -1, -1)

                x_in_v_reshaped = x_in_v[:, 0, :, 0]
                pos_emb_v = self.embed_positions(x_in_v_reshaped)
                pos_emb_v = pos_emb_v.unsqueeze(1).expand(-1, channels, -1, -1)
            else:
                pos_emb_k = self.embed_positions(x_in_k[:, :, 0]) if x_in_k.dim() == 3 else self.embed_positions(x_in_k)
                pos_emb_v = self.embed_positions(x_in_v[:, :, 0]) if x_in_v.dim() == 3 else self.embed_positions(x_in_v)

            x_k = self.embed_scale * x_in_k + pos_emb_k
            x_v = self.embed_scale * x_in_v + pos_emb_v

            x_k = F.dropout(x_k, p=self.dropout, training=self.training)
            x_v = F.dropout(x_v, p=self.dropout, training=self.training)
        
        
        if x.dim() == 4:
            batch_size, channels, seq_len, embed_dim = x.shape
            x = x.reshape(x.shape[0], -1, x.shape[3])
            
        if x_in_k is not None and x_in_v is not None:
            if x_in_k.dim() == 4:
                x_k = x_k.reshape(x_k.shape[0], -1, x_k.shape[3])
            if x_in_v.dim() == 4:
                x_v = x_v.reshape(x_v.shape[0], -1, x_v.shape[3])
        
        # encoder layers
        intermediates = [x]
        
        # 修正：由于底层的 MultiheadAttention(fairseq版本) 强制把第一维度当做 tgt_len (seq_len), 
        # 为了不混淆长度为1的 batch，必须在送入Layer前将 [batch, seq_len, embed] 转为 [seq_len, batch, embed]
        x = x.transpose(0, 1)
        if x_in_k is not None and x_in_v is not None:
            x_k = x_k.transpose(0, 1)
            x_v = x_v.transpose(0, 1)
        for layer in self.layers:
            if x_in_k is not None and x_in_v is not None:
                x = layer(x, x_k, x_v)
            else:
                x = layer(x)
            intermediates.append(x)

        # 出来之后转回 [batch, seq_len, embed]
        x = x.transpose(0, 1)

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
        """
        Args:
            x (Tensor): input to the layer of shape `(seq_len, batch, embed_dim)`
            encoder_padding_mask (ByteTensor): binary ByteTensor of shape
                `(batch, src_len)` where padding elements are indicated by ``1``.
            x_k (Tensor): same as x
            x_v (Tensor): same as x
        Returns:
            encoded output of shape `(batch, src_len, embed_dim)`
        """
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
    
    # 假设已经提取了图片的特征表示 (batch_size, sequence_length, embedding_dim)
    image_features = torch.randn(batch_size, sequence_length, embedding_dim)
    
    # 创建位置嵌入
    positional_embedding = SinusoidalPositionalEmbedding(embedding_dim)
    print(image_features.shape)
    print("****")
    print(positional_embedding(image_features[:,:,0]).shape)
    positioned_features = image_features + positional_embedding(image_features[:,:,0])

    # 创建Transformer编码器
    encoder = TransformerEncoder(embed_dim=embedding_dim, num_heads=4, layers=6)

    # 通过Transformer编码特征
    encoded_features = encoder(positioned_features)
    
    print(encoded_features.shape)
    # print(encoded_features)

    # encoded_features 可以用于后续的分类或其他任务
