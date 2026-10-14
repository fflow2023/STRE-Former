import torch.nn as nn
import torch
import torch.nn.functional as F

class CustomLoss(nn.Module):
    def __init__(self, class_counts):
        super(CustomLoss, self).__init__()
        self.class_counts = class_counts
        self.emotion_weight = None
        self.neu_weight = None
        self.neu_penalty = None
        self.update_weights()

    def update_weights(self):
        total_count = self.class_counts.sum()
        # 计算每个类的权重，反比例
        weights = total_count / self.class_counts

        # 归一化权重
        weights /= weights.sum()

        # 设定各类的权重
        self.emotion_weight = weights[1:].mean()  # 忽略 neu 的权重
        self.neu_weight = weights[0] * 2  # 增加 neu 的权重以鼓励学习
        self.neu_penalty = weights[0] * 3  # 更高的惩罚权重

    def forward(self, outputs, targets):
        emotion_mask = targets != 0  # 假设 0 为 neu 标签
        emotion_outputs = outputs[emotion_mask]
        emotion_targets = targets[emotion_mask] - 1

        # 情感标签的损失
        emotion_loss = F.cross_entropy(emotion_outputs, emotion_targets)

        # neu 标签的损失
        neu_mask = targets == 0
        neu_outputs = outputs[neu_mask]

        if len(neu_outputs) > 0:
            # 确保 neu_outputs 是浮点类型
            neu_loss = -torch.mean(torch.log(1 - F.softmax(neu_outputs, dim=1)[:, 0] + 1e-10))
            
            # 创建目标张量，确保其为 LongTensor
            neu_target = torch.full((neu_outputs.shape[0],), 0, dtype=torch.long, device=neu_outputs.device)
            neu_penalty_loss = F.cross_entropy(neu_outputs, neu_target)
        else:
            neu_loss = 0.0
            neu_penalty_loss = 0.0

        total_loss = (self.emotion_weight * emotion_loss +
                      self.neu_weight * neu_loss +
                      self.neu_penalty * neu_penalty_loss)

        return total_loss