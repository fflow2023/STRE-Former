import json
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.font_manager as fm

# 读取JSON数据
# with open('training_logs/8_training_data.json', 'r') as f:
with open('training_logs/IEMOCAP_4classes_training_data.json', 'r') as f:
    data = json.load(f)

# 设置字体 - 使用系统默认字体而不指定中文字体
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 提取数据
epochs = data['epochs']
train_loss = data['train']['loss']
train_wa = data['train']['wa']
train_ua = data['train']['ua']
val_loss = data['val']['loss']
val_wa = data['val']['wa']
val_ua = data['val']['ua']
val_f1 = data['val']['f1']

best_epoch = data['best']['epoch']
best_metrics = data['best']
test_metrics = data['test']

# 创建图表
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
fig.suptitle('Training Data Visualization', fontsize=16)

# 1. 损失曲线
axes[0, 0].plot(epochs, train_loss, 'b-', label='Train Loss')
axes[0, 0].plot(epochs, val_loss, 'r-', label='Val Loss')
axes[0, 0].axvline(x=best_epoch, color='g', linestyle='--', label=f'Best Model (epoch {best_epoch})')
axes[0, 0].set_xlabel('Epoch')
axes[0, 0].set_ylabel('Loss')
axes[0, 0].set_title('Train and Validation Loss')
axes[0, 0].legend()
axes[0, 0].grid(True)

# 2. WA (weighted accuracy) 曲线
axes[0, 1].plot(epochs, train_wa, 'b-', label='Train WA')
axes[0, 1].plot(epochs, val_wa, 'r-', label='Val WA')
axes[0, 1].axvline(x=best_epoch, color='g', linestyle='--', label=f'Best Model (epoch {best_epoch})')
axes[0, 1].axhline(y=test_metrics['wa'], color='m', linestyle='-.', label=f'Test WA: {test_metrics["wa"]:.2f}')
axes[0, 1].set_xlabel('Epoch')
axes[0, 1].set_ylabel('WA (%)')
axes[0, 1].set_title('Weighted Accuracy (WA)')
axes[0, 1].legend()
axes[0, 1].grid(True)

# 3. UA (unweighted accuracy) 曲线
axes[1, 0].plot(epochs, train_ua, 'b-', label='Train UA')
axes[1, 0].plot(epochs, val_ua, 'r-', label='Val UA')
axes[1, 0].axvline(x=best_epoch, color='g', linestyle='--', label=f'Best Model (epoch {best_epoch})')
axes[1, 0].axhline(y=test_metrics['ua'], color='m', linestyle='-.', label=f'Test UA: {test_metrics["ua"]:.2f}')
axes[1, 0].set_xlabel('Epoch')
axes[1, 0].set_ylabel('UA (%)')
axes[1, 0].set_title('Unweighted Accuracy (UA)')
axes[1, 0].legend()
axes[1, 0].grid(True)

# 4. F1分数曲线
axes[1, 1].plot(epochs, val_f1, 'r-', label='Val F1')
axes[1, 1].axvline(x=best_epoch, color='g', linestyle='--', label=f'Best Model (epoch {best_epoch})')
axes[1, 1].axhline(y=test_metrics['f1'], color='m', linestyle='-.', label=f'Test F1: {test_metrics["f1"]:.2f}')
axes[1, 1].set_xlabel('Epoch')
axes[1, 1].set_ylabel('F1 Score')
axes[1, 1].set_title('F1 Score')
axes[1, 1].legend()
axes[1, 1].grid(True)

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig('training_visualization.png', dpi=300)
print("图表已保存为 training_visualization.png")

# 额外创建一个表格，总结最佳模型和测试集性能
plt.figure(figsize=(8, 4))
plt.axis('off')
plt.title('Performance Metrics Summary', fontsize=14)

col_labels = ['Dataset', 'Loss', 'WA (%)', 'UA (%)', 'F1 Score']
table_data = [
    ['Best Val', f"{best_metrics['loss']:.4f}", f"{best_metrics['wa']:.2f}", f"{best_metrics['ua']:.2f}", f"{best_metrics['f1']:.2f}"],
    ['Test', f"{test_metrics['loss']:.4f}", f"{test_metrics['wa']:.2f}", f"{test_metrics['ua']:.2f}", f"{test_metrics['f1']:.2f}"]
]

table = plt.table(cellText=table_data, colLabels=col_labels, loc='center', cellLoc='center')
table.auto_set_font_size(False)
table.set_fontsize(12)
table.scale(1, 1.5)

plt.tight_layout()
plt.savefig('performance_summary.png', dpi=300)
print("性能指标总结已保存为 performance_summary.png")

plt.show() 