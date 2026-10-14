# tools.py
import torch
import argparse
from dataclasses import dataclass
import os
import random
import numpy as np
import warnings
from config import *
from rich.console import Console, Group
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, TextColumn, BarColumn, TaskProgressColumn
from rich.live import Live
from datetime import datetime
import json

# 过滤掉警告信息
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

# 创建控制台对象
console = Console()
# 创建文件日志控制台
log_path = os.path.join("logs", f"training_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
os.makedirs(os.path.dirname(log_path), exist_ok=True)
file_console = Console(file=open(log_path, "w", encoding="utf-8"))

# 日志记录器
class Logger:
    @staticmethod
    def log(message, style=None):
        """同时向控制台和文件写入日志"""
        console.print(message, style=style)
        file_console.print(message)
    
    @staticmethod
    def log_table(table):
        """同时向控制台和文件输出表格"""
        console.print(table)
        file_console.print(table)
        
    @staticmethod
    def log_panel(panel):
        """同时向控制台和文件输出面板"""
        console.print(panel)
        file_console.print(panel)

    @staticmethod
    def close():
        """关闭文件日志"""
        if hasattr(file_console, '_file') and file_console._file is not None:
            file_console._file.close()
            print(f"日志已保存至: {log_path}")

# 训练状态类
class TrainingStats:
    def __init__(
        self, epoch: int, best_epoch: int,
        train_loss_history: list[str], train_wa_history: list[str], train_ua_history: list[str],
        val_loss_history: list[str], val_wa_history: list[str], val_ua_history: list[str], val_f1_history: list[str],
        train_loss: str, train_wa: str, train_ua: str,
        test_loss: str, test_wa: str, test_ua: str, test_f1: str,
        confusion_matrix: np.ndarray,
        best_val_wa: float, best_val_ua: float, best_val_f1: float,
        best_val_acc: float, best_val_loss: float,
        loss_format: str, acc_format: str, acc_format2: str,
        current_lr: float
    ):
        self.epoch = epoch
        self.best_epoch = best_epoch
        self.train_loss_history = train_loss_history
        self.train_wa_history = train_wa_history
        self.train_ua_history = train_ua_history
        self.val_loss_history = val_loss_history
        self.val_wa_history = val_wa_history
        self.val_ua_history = val_ua_history
        self.val_f1_history = val_f1_history
        self.test_loss = test_loss
        self.test_wa = test_wa
        self.test_ua = test_ua
        self.test_f1 = test_f1
        self.confusion_matrix = confusion_matrix
        self.best_val_wa = best_val_wa
        self.best_val_ua = best_val_ua
        self.best_val_f1 = best_val_f1
        self.best_val_acc = best_val_acc
        self.best_val_loss = best_val_loss
        self.loss_format = loss_format
        self.acc_format = acc_format
        self.acc_format2 = acc_format2
        self.current_lr = current_lr
        self.train_loss = train_loss
        self.train_wa = train_wa
        self.train_ua = train_ua
        
        # 为记录训练过程创建JSON数据
        self.training_data = {
            "epochs": [],
            "train": {"loss": [], "wa": [], "ua": []},
            "val": {"loss": [], "wa": [], "ua": [], "f1": []},
            "best": {"epoch": best_epoch, "wa": best_val_wa, "ua": best_val_ua, "f1": best_val_f1, "loss": best_val_loss}
        }
    
    def update_train_history(self, train_loss, train_wa, train_ua):
        self.train_loss = self.loss_format.format(train_loss)
        self.train_wa = self.acc_format2.format(train_wa)
        self.train_ua = self.acc_format2.format(train_ua)
        
        self.train_loss_history.append(self.loss_format.format(train_loss))
        self.train_wa_history.append(self.acc_format2.format(train_wa))
        self.train_ua_history.append(self.acc_format2.format(train_ua))
        
        # 更新JSON数据
        self.training_data["train"]["loss"].append(float(train_loss))
        self.training_data["train"]["wa"].append(float(train_wa))
        self.training_data["train"]["ua"].append(float(train_ua))
    
    def update_val_history(self, val_loss, val_wa, val_ua, val_f1):
        self.val_loss_history.append(self.loss_format.format(val_loss))
        self.val_wa_history.append(self.acc_format2.format(val_wa))
        self.val_ua_history.append(self.acc_format2.format(val_ua))
        self.val_f1_history.append(self.acc_format2.format(val_f1))
        
        # 更新JSON数据
        self.training_data["val"]["loss"].append(float(val_loss))
        self.training_data["val"]["wa"].append(float(val_wa))
        self.training_data["val"]["ua"].append(float(val_ua))
        self.training_data["val"]["f1"].append(float(val_f1))
        self.training_data["epochs"].append(self.epoch + 1)
    
    def update_test_history(self, test_loss, test_wa, test_ua, test_f1,test_uf1, confusion_matrix):
        self.test_loss = self.loss_format.format(test_loss)
        self.test_wa = self.acc_format2.format(test_wa)
        self.test_f1 = self.acc_format2.format(test_f1)
        self.test_ua = self.acc_format2.format(test_ua)
        self.test_uf1 = self.acc_format2.format(test_uf1)
        self.confusion_matrix = confusion_matrix
        
        # 更新JSON数据
        self.training_data["test"] = {
            "loss": float(test_loss),
            "wa": float(test_wa),
            "f1": float(test_f1),
            "ua": float(test_ua),
            "uf1": float(test_uf1),
        }
    
    def display_train_process(self, train_loss, val_loss, val_wa, val_ua):
        table = Table(title=f"本轮训练后的结果 - 第 {self.epoch+1} 轮", show_header=True, header_style="bold magenta")
        table.add_column("指标", style="cyan")
        table.add_column("train", style="green")
        table.add_column("val", style="yellow")
        table.add_column("val_best", style="red")
        
        table.add_row(
            "loss", 
            self.loss_format.format(train_loss), 
            self.loss_format.format(val_loss),
            self.loss_format.format(self.best_val_loss)
        )
        table.add_row(
            "WA", 
            self.train_wa, 
            self.acc_format.format(val_wa), 
            self.acc_format.format(self.best_val_wa)
        )
        table.add_row(
            "UA", 
            self.train_ua, 
            self.acc_format.format(val_ua), 
            self.acc_format.format(self.best_val_ua)
        )
        
        # 添加学习率信息
        lr_info = f"当前学习率: {self.current_lr:.8f}"
        
        # 创建面板包装表格
        panel = Panel(table, border_style="blue", title=lr_info, expand=False)
        
        # 使用Logger输出并添加换行
        Logger.log_panel(panel)
        Logger.log("")
    
    def display_val_process(self, val_loss, val_wa, val_ua, val_f1):
        table = Table(title=f"VAL结果 - 第 {self.epoch+1} 轮", show_header=True, header_style="bold magenta")
        table.add_column("指标", style="cyan")
        table.add_column("当前值", style="yellow")
        table.add_column("最佳值", style="red")
        
        table.add_row("val loss", self.loss_format.format(val_loss), self.loss_format.format(self.best_val_loss))
        table.add_row("WA", self.acc_format.format(val_wa), self.acc_format.format(self.best_val_wa))
        table.add_row("UA", self.acc_format.format(val_ua), self.acc_format.format(self.best_val_ua))
        table.add_row("F1分数", self.acc_format.format(val_f1), self.acc_format.format(self.best_val_f1))
        
        # 使用Logger输出
        Logger.log_panel(Panel(table, border_style="yellow", title=f"学习率: {self.current_lr:.8f}", subtitle=f"最佳epoch: {self.best_epoch + 1}", expand=False))
        Logger.log("")
    
    def display_test_process(self, test_loss, test_wa, test_ua, test_f1,test_uf1, confusion_matrix):
        table = Table(title=f"TEST结果 - 第 {self.epoch+1} 轮", show_header=True, header_style="bold magenta")
        table.add_column("指标", style="cyan")
        table.add_column("值", style="green")
        
        table.add_row("test loss", self.loss_format.format(test_loss))
        table.add_row("WA", self.acc_format.format(test_wa))
        table.add_row("F1分数", self.acc_format.format(test_f1))
        table.add_row("UA", self.acc_format.format(test_ua))
        table.add_row("uw_F1", self.acc_format.format(test_uf1))
        
        
        # 创建混淆矩阵表格
        if confusion_matrix is not None and hasattr(confusion_matrix, 'shape'):
            cm_table = Table(title="TEST结果的混淆矩阵", show_header=True)
            # 添加列标题
            cm_table.add_column("真实\\预测", style="bold")
            for i in range(confusion_matrix.shape[1]):
                cm_table.add_column(f"{list(EMOTION_MAPS[DATABASE_KIND].keys())[i]}", style="cyan")
            
            # 添加行数据
            for i in range(confusion_matrix.shape[0]):
                row_data = [f"{list(EMOTION_MAPS[DATABASE_KIND].keys())[i]}"]
                for j in range(confusion_matrix.shape[1]):
                    cell_value = confusion_matrix[i, j]
                    # 高亮对角线元素
                    if i == j:
                        row_data.append(f"[bold green]{cell_value:.0f}[/bold green]")
                    else:
                        row_data.append(f"{cell_value:.0f}")
                cm_table.add_row(*row_data)
            
            # 创建Group组合两个表格
            group = Group(
                table,
                "\n",  # 添加一个空行作为分隔
                cm_table
            )
            
            # 使用Logger输出包含两个表格的Panel
            Logger.log_panel(Panel(group, border_style="green", title="TEST评估", expand=False))
            Logger.log("")
        else:
            # 如果没有混淆矩阵,只输出性能表格
            Logger.log_panel(Panel(table, border_style="green", title="TEST评估", expand=False))
            Logger.log("")
    
    def update_best_info(self, best_val_wa, best_val_ua, best_val_f1, best_val_loss, best_val_acc, best_epoch):
        self.best_val_wa = best_val_wa
        self.best_val_ua = best_val_ua
        self.best_val_f1 = best_val_f1
        self.best_val_loss = best_val_loss
        self.best_val_acc = best_val_acc
        self.best_epoch = best_epoch
        
        # 更新JSON数据
        self.training_data["best"] = {
            "epoch": best_epoch, 
            "wa": best_val_wa, 
            "ua": best_val_ua, 
            "f1": best_val_f1, 
            "loss": best_val_loss
        }
    
    def save_training_data(self, model_save_label):
        """保存训练数据到JSON文件"""
        os.makedirs("training_logs", exist_ok=True)
        json_path = os.path.join("training_logs", f"{model_save_label}_training_data.json")
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(self.training_data, f, indent=4)
        Logger.log(f"训练数据已保存到: [bold blue]{json_path}[/bold blue]")

@dataclass
class Args1:
    ratio: float = RATIO
    # coding_way: str = CODING_WAY
    coding_way_spec: str = CODING_WAY_SPEC
    coding_way_mfcc: str = CODING_WAY_MFCC
    coding_way_wav2vec: str = CODING_WAY_WAV2VEC
    input_feature: str = INPUT_FEATURE
    features_file_path: str = FEATURES_FILE_PATH
    ser_task: str = SER_TASK
    model_save_prefix: str = MODEL_SAVE_PREFIX
    model_overwrite_mode: bool = MODEL_OVERWRITE_MODE

def parse_args_1():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ratio', type=float, default=Args1.ratio)
    # parser.add_argument('--coding_way', type=str, default=Args1.coding_way, help="Fib(default), APE, RoPE")
    parser.add_argument('--coding_way_spec', type=str, default=Args1.coding_way_spec, help="Fib(default), APE, RoPE")
    parser.add_argument('--coding_way_mfcc', type=str, default=Args1.coding_way_mfcc, help="Fib(default), APE, RoPE")
    parser.add_argument('--coding_way_wav2vec', type=str, default=Args1.coding_way_wav2vec, help="Fib(default), APE, RoPE")
    parser.add_argument('--input_feature', type=str, default=Args1.input_feature)
    parser.add_argument('--features_file_path', type=str, default=Args1.features_file_path)
    parser.add_argument('--ser_task', type=str, default=Args1.ser_task)
    parser.add_argument('--model_save_prefix', type=str, default=Args1.model_save_prefix)
    parser.add_argument('--model_overwrite_mode', type=bool, default=Args1.model_overwrite_mode)
    return parser.parse_args()


@dataclass
class Args2:
    repeat_idx: str = None
    val_id: str = None
    test_id: str = None
    num_epochs: int = None
    early_stop: int = None
    batch_size: int = None
    lr: float = None
    seed: int = None
    use_gpu: bool = None
    gpu_id: int = None
    model_save_label: str = None
    oversampling: bool = None
    ratio: float = None
    use_Fib: bool = None
    model_overwrite_mode: bool = None
    pretrained: bool = None
    # coding_way: str = None
    coding_way_spec: str = None
    coding_way_mfcc: str = None
    coding_way_wav2vec: str = None
    input_feature: str = None
    features_file_path: str = None
    ser_task: str = None
    model_save_prefix: str = None
    num_classes: int = None
    backbone = BACKBONE
    layers = LAYERS
    use_fts: bool = USE_FTS

def parse_args_2(argv: Args2):
    parser = argparse.ArgumentParser(
        formatter_class = argparse.ArgumentDefaultsHelpFormatter,
        description = "Train a SER  model in an iterative-based manner with pyTorch and IEMOCAP dataset."
    )
    
    #Training
    parser.add_argument('--repeat_idx', type=str, default=Args2.repeat_idx, help='ID of repeat_idx')
    parser.add_argument('--val_id', type=str, default=Args2.val_id, help='ID of speaker to be used as validation')
    parser.add_argument('--test_id', type=str, default=Args2.test_id, help='ID of speaker to be used as test')
    parser.add_argument('--num_epochs', type=int, default=Args2.num_epochs, help='Number of training epochs.') 
    parser.add_argument('--early_stop', type=int, default=Args2.early_stop, help='Number of early stopping epochs.') 
    parser.add_argument('--batch_size', type=int, default=Args2.batch_size, help='Mini batch size.')
    parser.add_argument('--lr', type=float, default=Args2.lr, help='Learning rate.')
    parser.add_argument('--seed', type=int, default=Args2.seed, help='Random seed for reproducibility.')
    parser.add_argument('--use_gpu', type=bool, default=Args2.use_gpu, help='If True, use GPU')
    parser.add_argument('--gpu_id', type=int, default=Args2.gpu_id, help='If 0, use GPU 0')        
    
    #Best Model
    parser.add_argument('--model_save_label', type=str, default=Args2.model_save_label, help='Label for the current run, used to save the best model ')

    #Parameters for model tuning
    parser.add_argument('--oversampling', type=bool, default=Args2.oversampling, help='By default, no oversampling is applied to training dataset. Set this to true to apply random oversampling to balance training dataset')
    
    parser.add_argument('--ratio', type=float, default=Args2.ratio)
    parser.add_argument('--use_Fib', action='store_true')
    parser.add_argument('--model_overwrite_mode', type=bool, default=Args2.model_overwrite_mode)
    
    parser.add_argument('--pretrained', action='store_true', help='By default, SER_AlexNet or SER_AlexNet_GAP model weights are initialized randomly. Set this flag to initalize with ImageNet pre-trained weights.')
    # parser.add_argument('--coding_way', type=str, default=Args2.coding_way, help="Fib(default), APE, RoPE")
    parser.add_argument('--coding_way_spec', type=str, default=Args2.coding_way_spec, help="Fib(default), APE, RoPE")
    parser.add_argument('--coding_way_mfcc', type=str, default=Args2.coding_way_mfcc, help="Fib(default), APE, RoPE")
    parser.add_argument('--coding_way_wav2vec', type=str, default=Args2.coding_way_wav2vec, help="Fib(default), APE, RoPE")
    parser.add_argument('--input_feature', type=str, default=Args2.input_feature)
    parser.add_argument('--features_file_path', type=str, default=Args2.features_file_path)
    parser.add_argument('--ser_task', type=str, default=Args2.ser_task)
    parser.add_argument('--model_save_prefix', type=str, default=Args2.model_save_prefix)
    parser.add_argument('--num_classes', type=int, default=Args2.num_classes)
    parser.add_argument('--backbone', type=str, default=Args2.backbone, choices=['transformer', 'swintransformer'], help="backbone type")
    parser.add_argument('--layers', type=int, default=Args2.layers, help="number of transformer layers")
    parser.add_argument('--use_fts', action='store_true', default=Args2.use_fts, help="启用FTS（Fibonacci Target Shutter）")
    parser.add_argument('--no_fts', action='store_true', help="禁用FTS（消融实验）")
    args = parser.parse_args(argv)
    # 处理 --no_fts 互斥逻辑
    if args.no_fts:
        args.use_fts = False
    delattr(args, 'no_fts')
    return args


# 设置随机种子
def seed_everything(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def confirm_args(args2: Args2) -> None:
    # 将args转换为字典
    params = vars(args2)
    
    # 创建一个表格展示参数
    table = Table(title="训练参数配置", show_header=True, header_style="bold magenta")
    table.add_column("参数", style="cyan")
    table.add_column("值", style="green")
    
    for key, value in params.items():
        table.add_row(key, str(value))
    
    # 使用Logger输出参数表格
    Logger.log(Panel(table, border_style="blue", expand=False))
    
    model_save_path = os.path.join(args2.model_save_prefix, f'{args2.model_save_label}.pth')
    if not args2.model_overwrite_mode and os.path.exists(model_save_path):
        Logger.log(f"[bold yellow]警告:[/bold yellow] 模型文件 {model_save_path} 已存在，是否继续训练（断点继续）？", style="yellow")
        answer = input('请输入 y 或 n: ')
        if answer.lower() == 'n':
            Logger.log("训练已取消", style="red")
            exit()

def display_cuda_info():
    # 创建表格展示CUDA信息
    table = Table(title="CUDA信息", show_header=True, header_style="bold magenta")
    table.add_column("信息", style="cyan")
    table.add_column("值", style="green")
    
    table.add_row("PyTorch版本", torch.__version__)
    table.add_row("CUDA版本", str(torch.cuda.get_device_capability()))
    table.add_row("cuDNN版本", str(torch.backends.cudnn.version()))
    table.add_row("GPU名称", torch.cuda.get_device_name())
    table.add_row("GPU索引", str(torch.cuda.current_device()))
    
    # 使用Logger输出表格
    Logger.log_table(table)

def display_trainable_parameters_count(model):
    # 计算模型中可训练参数的数量
    trainable_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # 使用Logger输出参数数量
    Logger.log(Panel(f"可训练参数数量: [bold green]{trainable_parameters:,}[/bold green]", border_style="green", expand=False))
    
    return trainable_parameters


# 动态调整学习率
def adjust_learning_rate(lr_0, optimizer, epoch):
    lr = lr_0 * (0.8 ** (epoch // 20))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr