import torch
import torch.nn as nn
import torch.nn.functional as F
import json
from config import *
from models.hifigan.models import Generator, MultiPeriodDiscriminator, MultiScaleDiscriminator, feature_loss
from models.hifigan.meldataset import mel_spectrogram
from models.hifigan.env import AttrDict


class FocalLoss(nn.Module):
    def __init__(self, gamma=2.5, alpha=1, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.size_average = size_average
        self.elipson = 0.000001

    def forward(self, logits, labels):
        """
        cal culates loss
        logits: batch_size * labels_length * seq_length
        labels: batch_size * seq_length
        """
        if labels.dim() > 2:
            labels = labels.contiguous().view(labels.size(0), labels.size(1), -1)
            labels = labels.transpose(1, 2)
            labels = labels.contiguous().view(-1, labels.size(2)).squeeze()
        if logits.dim() > 3:
            logits = logits.contiguous().view(logits.size(0), logits.size(1), logits.size(2), -1)
            logits = logits.transpose(2, 3)
            logits = logits.contiguous().view(-1, logits.size(1), logits.size(3)).squeeze()
        labels_length = logits.size(1)
        seq_length = logits.size(0)

        new_label = labels.unsqueeze(1)
        label_onehot = torch.zeros([seq_length, labels_length]).cuda().scatter_(1, new_label, 1)

        log_p = F.log_softmax(logits, dim=-1)
        pt = label_onehot * log_p
        sub_pt = 1 - pt
        fl = -self.alpha * (sub_pt) ** self.gamma * log_p
        if self.size_average:
            return fl.mean()
        else:
            return fl.sum()


class HiFiGANLoss(nn.Module):
    """HiFi-GAN损失函数类
    计算mel谱图损失和特征匹配损失
    """
    def __init__(self, **kwargs):
        super(HiFiGANLoss, self).__init__()
        # 加载预训练的HiFi-GAN模型
        self.generator, self.mpd, self.msd, self.h, self.device = self.load_model(
            HIFI_GAN_CONFIG_PATH,
            HIFI_GAN_MODEL_PATH
        )
        
        # 冻结HiFi-GAN参数
        for param in self.generator.parameters():
            param.requires_grad = False
        for param in self.mpd.parameters():
            param.requires_grad = False
        for param in self.msd.parameters():
            param.requires_grad = False
            
        # 将模型移至正确设备
        self.generator = self.generator.to(self.device)
        self.mpd = self.mpd.to(self.device)
        self.msd = self.msd.to(self.device)
        
        # 特征投影层 - 只投影到频率维度，不固定时间维度
        self.feature_project = nn.Linear(128, 80)
        
        
        
    def load_model(self, config_path, checkpoint_path):
        # 加载配置
        with open(config_path) as f:
            data = f.read()
        json_config = json.loads(data)
        h = AttrDict(json_config)
        
        # 确定设备
        device = torch.device(f"cuda:{GPU_ID}" if USE_GPU else "cpu")
        # device = torch.device("cpu")
        
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

    
    def forward(self, feature, audio):
        """前向传播计算损失
        Args:
            feature: [batch_size, 128] 的特征向量
            audio: [batch_size, 48000] 的原始音频
        Returns:
            total_loss: 总损失 = mel_loss + fm_loss_mpd + fm_loss_msd
        """
        # 确保feature和feature_project在同一设备上
        batch_size = feature.shape[0]
        self.feature_project = self.feature_project.to(self.device)
        feature = feature.to(self.device)
        
        # 计算原始音频的梅尔谱图以获取时间维度
        audio = audio.to(self.device)
        original_mel = mel_spectrogram(
            audio,
            self.h.n_fft, self.h.num_mels, self.h.sampling_rate,
            self.h.hop_size, self.h.win_size, self.h.fmin, self.h.fmax
        )
        
        # 获取正确的时间维度
        _, _, time_dim = original_mel.shape
        
        # 投影特征并调整为正确的形状
        projected_features = self.feature_project(feature)  # [batch_size, 80]
        mel_feature = projected_features.unsqueeze(-1).expand(-1, -1, time_dim)  # [batch_size, 80, time_dim]
        
        # 生成音频
        with torch.no_grad():
            generated_audio = self.generator(mel_feature)  # [batch_size, 1, audio_len]
        
        # 截断音频长度以确保一致 - 使用较短的长度
        min_audio_len = min(audio.size(1), generated_audio.size(2))
        audio = audio[:, :min_audio_len]
        generated_audio = generated_audio[:, :, :min_audio_len]
        
        # 计算生成音频的梅尔谱图
        generated_mel = mel_spectrogram(
            generated_audio.squeeze(1),
            self.h.n_fft, self.h.num_mels, self.h.sampling_rate, 
            self.h.hop_size, self.h.win_size, self.h.fmin, self.h.fmax
        )
        
        # 重新计算原始音频的梅尔谱图（使用截断后的音频）
        original_mel = mel_spectrogram(
            audio,
            self.h.n_fft, self.h.num_mels, self.h.sampling_rate,
            self.h.hop_size, self.h.win_size, self.h.fmin, self.h.fmax
        )
        
        # 计算梅尔谱图L1损失
        mel_loss = F.l1_loss(original_mel, generated_mel)
        
        # 计算特征匹配损失
        audio_3d = audio.unsqueeze(1)  # [batch_size, 1, audio_len]
        
        # 获取判别器特征
        _, _, fmap_mpd_real, fmap_mpd_gen = self.mpd(audio_3d, generated_audio)
        _, _, fmap_msd_real, fmap_msd_gen = self.msd(audio_3d, generated_audio)
        
        # 计算特征匹配损失
        fm_loss_mpd = 0.0
        for i in range(len(fmap_mpd_real)):
            for j in range(len(fmap_mpd_real[i])):
                feat_r = fmap_mpd_real[i][j]
                feat_g = fmap_mpd_gen[i][j]
                fm_loss_mpd += torch.mean(torch.abs(feat_r - feat_g))
        
        fm_loss_msd = 0.0
        for i in range(len(fmap_msd_real)):
            for j in range(len(fmap_msd_real[i])):
                feat_r = fmap_msd_real[i][j]
                feat_g = fmap_msd_gen[i][j]
                fm_loss_msd += torch.mean(torch.abs(feat_r - feat_g))
        
        # 计算总损失
        total_loss = mel_loss + fm_loss_mpd + fm_loss_msd
        
        return total_loss


def get_focal(**kwargs):
    return FocalLoss(**kwargs)

def get_cross_entropy(**kwargs):
    return nn.CrossEntropyLoss(**kwargs)

def get_multi_margin(**kwargs):
    return nn.MultiMarginLoss(**kwargs)

def get_weighted_cross_entropy(**kwargs):
    """
    加权损失函数
    输入需要有classes_count和device
    """
    # 获取device 和 classes_count
    device = kwargs.get('device')
    classes_count = kwargs.get('classes_count')
    
    # 将classes_count放到对应device上
    class_counts = torch.tensor(classes_count).float().to(device)

    # 计算每个类别的权重（频率的倒数）
    class_weights = 1.0 / class_counts

    # 归一化权重到一个合理范围（例如 [0.1, 1] 或其他合适范围）
    class_weights = class_weights / class_weights.sum() * len(class_counts)
    
    return nn.CrossEntropyLoss(weight=class_weights)

def get_hifigan(**kwargs):
    """获取HiFiGAN损失函数实例"""
    return HiFiGANLoss(**kwargs)

# 获取损失函数的统一接口
def get_loss_function(loss_name: str, **kwargs):
    # 定义损失函数字典
    loss_functions = {
        "cross_entropy": get_cross_entropy,
        "focal": get_focal,
        "multi_margin": get_multi_margin,
        "weighted_cross_entropy": get_weighted_cross_entropy,
        "hifigan": get_hifigan
    }
    
    # 从字典中获取对应的损失函数
    loss_fn = loss_functions.get(loss_name)
    if loss_fn is None:
        raise ValueError(f"不支持的损失函数: {loss_name}")
        
    return loss_fn(**kwargs)


