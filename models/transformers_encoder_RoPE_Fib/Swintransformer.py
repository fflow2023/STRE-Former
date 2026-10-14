#\transformers_encoder_RoPE_Fib\swintransformer.py
import torch
from torch import nn
import torch.nn.functional as F
from models.transformers_encoder_RoPE_Fib.position_embedding import *
from models.transformers_encoder_RoPE_Fib.multihead_attention import MultiheadAttention
import math

class WindowPartition(nn.Module):
    def __init__(self, window_size):
        super().__init__()
        self.window_size = window_size

    def forward(self, x):
        B, H, W, C = x.shape
        # 确保尺寸能被窗口大小整除
        assert H % self.window_size == 0, f"H ({H}) must be divisible by window_size ({self.window_size})"
        assert W % self.window_size == 0, f"W ({W}) must be divisible by window_size ({self.window_size})"
        
        # 计算窗口数量
        num_h_windows = H // self.window_size
        num_w_windows = W // self.window_size
        
        # 划分窗口 - 使用reshape代替view确保连续内存
        x = x.reshape(B, num_h_windows, self.window_size, num_w_windows, self.window_size, C)
        windows = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        windows = windows.reshape(-1, self.window_size, self.window_size, C)
        return windows

class WindowReverse(nn.Module):
    def __init__(self, window_size):
        super().__init__()
        self.window_size = window_size

    def forward(self, windows, H, W):
        # 计算窗口数量
        num_h_windows = H // self.window_size
        num_w_windows = W // self.window_size
        num_windows = num_h_windows * num_w_windows
        
        # 计算batch size
        B = windows.shape[0] // num_windows
        
        # 还原窗口 - 使用reshape代替view
        x = windows.reshape(B, num_h_windows, num_w_windows, 
                            self.window_size, self.window_size, -1)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        x = x.reshape(B, H, W, -1)
        return x

class TransformerEncoder(nn.Module):
    def __init__(self, num_heads, layers, in_channels=3, pic_size=(200, 300), patch_size=(10, 10), 
                 embed_dim=None, attn_dropout=0.0, relu_dropout=0.0, res_dropout=0.0,
                 embed_dropout=0.0, attn_mask=False, device="cuda:0", target_number=10, 
                 ratio=None, window_size=7, shift_size=None):
        super().__init__()
        self.embed_dim = embed_dim if embed_dim else patch_size[0] * patch_size[1]
        self.embed_scale = math.sqrt(self.embed_dim)
        self.patch_size = patch_size
        self.pic_size = pic_size
        self.window_size = window_size
        self.shift_size = shift_size if shift_size else window_size // 2
        
        self.conv_proj = nn.Conv2d(in_channels, self.embed_dim, kernel_size=patch_size, stride=patch_size)
        
        self.window_partition = WindowPartition(window_size)
        self.window_reverse = WindowReverse(window_size)
        
        self.layers = nn.ModuleList([
            SwinTransformerEncoderLayer(
                embed_dim=self.embed_dim,
                pic_size=pic_size,
                patch_size=patch_size,
                num_heads=num_heads,
                attn_dropout=attn_dropout,
                relu_dropout=relu_dropout,
                res_dropout=res_dropout,
                attn_mask=attn_mask,
                device=device,
                target_number=target_number,
                window_size=window_size,
                shift_size=0 if (i % 2 == 0) else self.shift_size
            ) for i in range(layers)
        ])
        
        self.normalize = True
        if self.normalize:
            self.layer_norm = LayerNorm(self.embed_dim)

    def forward(self, x_in):
        # Patch embedding
        x = self.conv_proj(x_in)
        x = x.permute(0, 2, 3, 1)  # (B, H, W, C)
        B, H, W, C = x.shape
        
        # 计算填充量以确保能被窗口大小整除
        pad_h = (self.window_size - H % self.window_size) % self.window_size
        pad_w = (self.window_size - W % self.window_size) % self.window_size
        
        # 对特征图进行填充
        if pad_h > 0 or pad_w > 0:
            # 使用正确的填充顺序：左、右、上、下
            x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))  # 通道维度不填充
            H_padded = H + pad_h
            W_padded = W + pad_w
        else:
            H_padded, W_padded = H, W
        
        # 确保填充后尺寸可被整除
        assert H_padded % self.window_size == 0, f"H_padded {H_padded} must be divisible by window_size {self.window_size}"
        assert W_padded % self.window_size == 0, f"W_padded {W_padded} must be divisible by window_size {self.window_size}"
        
        # 应用窗口划分
        x = self.window_partition(x)  # (B*num_windows, ws, ws, C)
        x = x.reshape(-1, self.window_size*self.window_size, C)  # 使用reshape代替view
        
        # 处理Swin层
        for layer in self.layers:
            x = layer(x, H_padded, W_padded)
        
        # 从窗口重建
        x = x.reshape(-1, self.window_size, self.window_size, C)
        x = self.window_reverse(x, H_padded, W_padded)
        
        # 移除填充
        if pad_h > 0 or pad_w > 0:
            x = x[:, :H, :W, :]
        
        # 使用reshape代替view确保连续内存
        x = x.reshape(B, H*W, C)  # (B, H*W, C)
        
        if self.normalize:
            x = self.layer_norm(x)
            
        return x

class SwinTransformerEncoderLayer(nn.Module):
    def __init__(self, embed_dim, pic_size, patch_size, num_heads=4, attn_dropout=0.1, 
                 relu_dropout=0.1, res_dropout=0.1, attn_mask=False, device='cuda:0', 
                 target_number=10, window_size=7, shift_size=0):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        
        self.self_attn = MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            pic_size=pic_size,
            patch_size=patch_size,
            attn_dropout=attn_dropout,
            device=device,
            target_number=target_number
        )
        
        self.fc1 = Linear(embed_dim, 4*embed_dim)
        self.fc2 = Linear(4*embed_dim, embed_dim)
        self.layer_norms = nn.ModuleList([LayerNorm(embed_dim) for _ in range(2)])
        self.dropout = nn.Dropout(res_dropout)
        
    def forward(self, x, H, W):
        # x shape: (B*num_windows, ws*ws, C)
        residual = x
        x = self.maybe_layer_norm(0, x, before=True)
        
        # 如果需要，应用循环移位
        if self.shift_size > 0:
            x_4d = x.reshape(-1, self.window_size, self.window_size, self.embed_dim)
            shifted_x = torch.roll(
                x_4d,
                shifts=(-self.shift_size, -self.shift_size),
                dims=(1, 2)
            )
            shifted_x = shifted_x.reshape(-1, self.window_size*self.window_size, self.embed_dim)
        else:
            shifted_x = x
            
        # 窗口注意力
        attn_mask = None
        if self.shift_size > 0:
            # 计算窗口数量
            num_h_windows = H // self.window_size
            num_w_windows = W // self.window_size
            num_windows = num_h_windows * num_w_windows
            
            # 创建注意力掩码
            img_mask = torch.zeros((1, H, W, 1), device=x.device)
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1
            
            # 将掩码划分为窗口
            mask_windows = img_mask.reshape(
                1, 
                num_h_windows, self.window_size, 
                num_w_windows, self.window_size, 1
            ).permute(0, 1, 3, 2, 4, 5).contiguous()
            mask_windows = mask_windows.reshape(-1, self.window_size*self.window_size)
            
            # 创建注意力掩码
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))

            # 获取批次大小
            B = x.shape[0] // (H // self.window_size * W // self.window_size)  # 计算真实批次大小

            # 确保 attn_mask 的形状与 attn_weights 对齐
            attn_mask = attn_mask.unsqueeze(0).expand(B, -1, -1, -1)  # [B, num_windows, seq_len, seq_len]
            attn_mask = attn_mask.flatten(0, 1)  # [B * num_windows, seq_len, seq_len]
            attn_mask = attn_mask.unsqueeze(1).expand(-1, self.num_heads, -1, -1)  # [B * num_windows, num_heads, seq_len, seq_len]
            attn_mask = attn_mask.flatten(0, 1)  # [B * num_heads * num_windows, seq_len, seq_len]

        
        # 应用注意力
        x_attn, _ = self.self_attn(shifted_x, shifted_x, shifted_x, attn_mask=attn_mask)
        
        # 如果需要，反转移位
        if self.shift_size > 0:
            x_attn_4d = x_attn.reshape(-1, self.window_size, self.window_size, self.embed_dim)
            reversed_x = torch.roll(
                x_attn_4d,
                shifts=(self.shift_size, self.shift_size),
                dims=(1, 2)
            )
            x_attn = reversed_x.reshape(-1, self.window_size*self.window_size, self.embed_dim)
        
        # 残差连接
        x = residual + self.dropout(x_attn)
        x = self.maybe_layer_norm(0, x, after=True)
        
        # FFN
        residual = x
        x = self.maybe_layer_norm(1, x, before=True)
        x = F.gelu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        x = residual + self.dropout(x)
        x = self.maybe_layer_norm(1, x, after=True)
        
        return x

    def maybe_layer_norm(self, i, x, before=False, after=False):
        if after:
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
    batch_size = 64
    height = 300
    width = 40
    in_channels = 3
    
    image_features = torch.randn(batch_size, in_channels, height, width).to(device=torch.device('cuda:0'))
    
    print("input shape: ", image_features.shape)
    
    positioned_features = image_features

    # 创建Transformer编码器
    encoder = TransformerEncoder(num_heads=4, layers=6, pic_size=(height, width), patch_size=(20, 10)).to(device=torch.device('cuda:0'))

    # 通过Transformer编码特征
    encoded_features = encoder(positioned_features)
    
    print(encoded_features.shape)
