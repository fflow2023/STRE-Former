import torch
import torch.nn.functional as F
from models import Generator, MultiPeriodDiscriminator, MultiScaleDiscriminator, feature_loss
from meldataset import mel_spectrogram
from env import AttrDict
import json
import os
import numpy as np
from scipy.io.wavfile import read

# 加载配置文件和检查点
def load_model(config_path, checkpoint_path):
    # 加载配置
    with open(config_path) as f:
        data = f.read()
    json_config = json.loads(data)
    h = AttrDict(json_config)
    
    # 确定设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 初始化模型
    generator = Generator(h).to(device)
    mpd = MultiPeriodDiscriminator().to(device)
    msd = MultiScaleDiscriminator().to(device)
    
    # 加载检查点
    checkpoint_dict = torch.load(checkpoint_path, map_location=device)
    generator.load_state_dict(checkpoint_dict['generator'])
    
    # 如果checkpoint包含判别器权重，也加载它们
    if 'mpd' in checkpoint_dict and 'msd' in checkpoint_dict:
        mpd.load_state_dict(checkpoint_dict['mpd'])
        msd.load_state_dict(checkpoint_dict['msd'])
    
    generator.eval()
    mpd.eval()
    msd.eval()
    
    return generator, mpd, msd, h, device

# 计算损失函数
def compute_loss(original_audio, features, generator, mpd, msd, h, device):
    # 确保输入是tensor并在正确的设备上
    if not isinstance(original_audio, torch.Tensor):
        original_audio = torch.FloatTensor(original_audio).to(device)
    if not isinstance(features, torch.Tensor):
        features = torch.FloatTensor(features).to(device)
    
    # 确保形状正确
    if original_audio.dim() == 1:
        original_audio = original_audio.unsqueeze(0)  # [batch_size, audio_len]
    if features.dim() == 2:
        features = features.unsqueeze(0)  # [batch_size, n_mels, time]
    
    # 通过generator生成音频
    with torch.no_grad():
        generated_audio = generator(features)  # [batch_size, 1, audio_len]
    
    # 将生成的音频转为梅尔谱图以便比较
    generated_mel = mel_spectrogram(generated_audio.squeeze(1), h.n_fft, h.num_mels, 
                                    h.sampling_rate, h.hop_size, h.win_size, 
                                    h.fmin, h.fmax_for_loss)
    
    original_mel = mel_spectrogram(original_audio, h.n_fft, h.num_mels, 
                                 h.sampling_rate, h.hop_size, h.win_size, 
                                 h.fmin, h.fmax_for_loss)
    
    # 1. 梅尔谱图L1损失
    mel_loss = F.l1_loss(original_mel, generated_mel)
    
    # 2. 特征匹配损失和对抗损失
    # 准备输入格式
    original_audio = original_audio.unsqueeze(1)  # [batch_size, 1, audio_len]
    
    # 多周期判别器损失
    _, _, fmap_mpd_real, fmap_mpd_gen = mpd(original_audio, generated_audio)
    # 多尺度判别器损失
    _, _, fmap_msd_real, fmap_msd_gen = msd(original_audio, generated_audio)
    
    # 特征匹配损失
    fm_loss_mpd = feature_loss(fmap_mpd_real, fmap_mpd_gen)
    fm_loss_msd = feature_loss(fmap_msd_real, fmap_msd_gen)
    
    # 总损失
    total_loss = mel_loss + fm_loss_mpd + fm_loss_msd
    
    return {
        'total_loss': total_loss.item(),
        'mel_loss': mel_loss.item(),
        'feature_loss_mpd': fm_loss_mpd.item(),
        'feature_loss_msd': fm_loss_msd.item(),
    }

# 使用示例
def evaluate_feature(original_audio_path, feature_path, config_path, checkpoint_path):
    # 加载原始音频
    audio, sr = read(original_audio_path)
    audio = audio / 32768.0  # 归一化到 [-1, 1]
    
    # 加载特征（假设是梅尔谱图）
    feature = np.load(feature_path)
    
    # 加载模型
    generator, mpd, msd, h, device = load_model(config_path, checkpoint_path)
    
    # 计算损失
    losses = compute_loss(audio, feature, generator, mpd, msd, h, device)
    
    return losses






# # 假设您有一个模型，输出音频特征
# class YourModel(torch.nn.Module):
#     def __init__(self):
#         super(YourModel, self).__init__()
#         # 您的模型定义
        
#     def forward(self, x):
#         # 您的模型前向传播，返回特征
#         return features

# # 在训练循环中
# def train_with_hifigan_loss():
#     # 加载HiFi-GAN模型
#     generator, mpd, msd, h, device = load_model("/home/chason/Project/FibFormer-2025-05-27/models/hifi-gan/pretrained_models/UNIVERSAL_V1/config.json",
#                                                 "/home/chason/Project/FibFormer-2025-05-27/models/hifi-gan/pretrained_models/UNIVERSAL_V1/g_02500000")
    
#     # 冻结HiFi-GAN参数，因为我们只用它来计算损失
#     for param in generator.parameters():
#         param.requires_grad = False
#     for param in mpd.parameters():
#         param.requires_grad = False
#     for param in msd.parameters():
#         param.requires_grad = False
    
#     # 您的模型
#     your_model = YourModel().to(device)
#     optimizer = torch.optim.Adam(your_model.parameters(), lr=0.0001)
    
#     # 训练循环
#     for epoch in range(1):
#         for batch in dataloader:
#             # 获取输入数据和原始音频
#             input_data, original_audio = batch
#             input_data = input_data.to(device)
#             original_audio = original_audio.to(device)
            
#             # 前向传播，获取您模型的特征输出
#             features = your_model(input_data)
            
#             # 使用HiFi-GAN生成音频
#             generated_audio = generator(features)
            
#             # 计算各种损失
#             # 1. 您模型的主要损失
#             primary_loss = your_primary_loss_function(features)
            
#             # 2. HiFi-GAN损失
#             generated_mel = mel_spectrogram(generated_audio.squeeze(1), h.n_fft, h.num_mels, 
#                                           h.sampling_rate, h.hop_size, h.win_size, 
#                                           h.fmin, h.fmax_for_loss)
            
#             original_mel = mel_spectrogram(original_audio, h.n_fft, h.num_mels, 
#                                          h.sampling_rate, h.hop_size, h.win_size, 
#                                          h.fmin, h.fmax_for_loss)
            
#             mel_loss = F.l1_loss(original_mel, generated_mel)
            
#             # 3. 特征匹配损失
#             original_audio = original_audio.unsqueeze(1)
#             _, _, fmap_mpd_real, fmap_mpd_gen = mpd(original_audio, generated_audio)
#             _, _, fmap_msd_real, fmap_msd_gen = msd(original_audio, generated_audio)
            
#             fm_loss_mpd = feature_loss(fmap_mpd_real, fmap_mpd_gen)
#             fm_loss_msd = feature_loss(fmap_msd_real, fmap_msd_gen)
            
#             # 总损失
#             total_loss = primary_loss + mel_loss + fm_loss_mpd + fm_loss_msd
            
#             # 反向传播和优化
#             optimizer.zero_grad()
#             total_loss.backward()
#             optimizer.step()




generator, mpd, msd, h, device = load_model("/home/chason/Project/FibFormer-2025-05-27/models/hifigan/pretrained_models/UNIVERSAL_V1/config.json",
                                            "/home/chason/Project/FibFormer-2025-05-27/models/hifigan/pretrained_models/UNIVERSAL_V1/g_02500000")



# 冻结HiFi-GAN参数，因为我们只用它来计算损失
for param in generator.parameters():
    param.requires_grad = False
for param in mpd.parameters():
    param.requires_grad = False
for param in msd.parameters():
    param.requires_grad = False




generator.to(device)
mpd.to(device)
msd.to(device)


# 创建梅尔谱图特征: [batch_size, n_mels, time]
batch_size = 32  # 设置batch_size
n_mels = 80
time_steps = 3

# 创建特征数据
features = torch.ones(batch_size, n_mels, time_steps).float().to(device)





print(f"特征形状: {features.shape}")  # 输出: [1, 80, 800]


# h.num_mels = 128
# h.hop_size = 10
# h.n_fft = 800
# h.win_size = 40

# h.sampling_rate = 16000




# 使用HiFi-GAN生成音频
with torch.no_grad():
    generated_audio = generator(features)  # [batch_size, 1, audio_len]
    
print(f"生成音频形状: {generated_audio.shape}")  # 例如: [1, 1, 204800]

# 创建假的原始音频，与生成音频长度匹配
audio_len = generated_audio.shape[2]

original_audio = torch.ones(batch_size, audio_len).float().to(device)  # [batch_size, audio_len]
print(f"原始音频形状: {original_audio.shape}")  # 例如: [1, 204800]

# 计算梅尔谱图
generated_mel = mel_spectrogram(
    generated_audio.squeeze(1),  # [batch_size, audio_len]
    n_fft=800, num_mels=128, sampling_rate=16000, hop_size=10, win_size=40, fmin=0, fmax=8000
)
print(f"生成梅尔谱图形状: {generated_mel.shape}")  # 例如: [1, 80, ~800]

original_mel = mel_spectrogram(
    original_audio,  # [batch_size, audio_len] 已经是正确形状
    n_fft=800, num_mels=128, sampling_rate=16000, hop_size=10, win_size=40, fmin=0, fmax=8000
)
print(f"原始梅尔谱图形状: {original_mel.shape}")  # 例如: [1, 80, ~800]

# 计算梅尔谱图L1损失
mel_loss = F.l1_loss(original_mel, generated_mel)
print(f"梅尔谱图L1损失: {mel_loss.item()}")

# 计算特征匹配损失 - 需要将原始音频转为3D
original_audio_3d = original_audio.unsqueeze(1)  # [batch_size, 1, audio_len]
print(f"原始音频3D形状: {original_audio_3d.shape}")  # 例如: [1, 1, 204800]

_, _, fmap_mpd_real, fmap_mpd_gen = mpd(original_audio_3d, generated_audio)
_, _, fmap_msd_real, fmap_msd_gen = msd(original_audio_3d, generated_audio)

fm_loss_mpd = feature_loss(fmap_mpd_real, fmap_mpd_gen)
fm_loss_msd = feature_loss(fmap_msd_real, fmap_msd_gen)

print(f"MPD特征匹配损失: {fm_loss_mpd.item()}")
print(f"MSD特征匹配损失: {fm_loss_msd.item()}")

# 计算总损失
total_loss = mel_loss + fm_loss_mpd + fm_loss_msd
print(f"总损失: {total_loss.item()}")