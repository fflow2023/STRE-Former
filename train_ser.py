
import warnings
# 忽略 Hugging Face Transformers 的特定警告
warnings.filterwarnings("ignore", message="Some weights of the model checkpoint")
warnings.filterwarnings("ignore", message="You should probably TRAIN this model")

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as f
from torch.utils.data import DataLoader
import sys
import argparse
import pickle
from data_utils import SERDataset, TestDataset
import numpy as np
from models.ser_model import Ser_Model
import os
import random
from tqdm import tqdm
from rich.panel import Panel
from rich.table import Table
from collections import Counter
from torch.backends import cudnn
from sklearn.manifold import TSNE
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time
from concurrent.futures import ThreadPoolExecutor
from models.losses import get_loss_function
from tools import *
from config import *


def main(args2: Args2) -> TrainingStats:
    # 展示并确认训练参数
    confirm_args(args2)
    
    # 设置随机种子
    seed_everything(args2.seed)
    
    # 开始训练！
    train_stat = train(args2)
    
    # # 检查train_stat是否为None
    # if train_stat is None:
    #     Logger.log("[bold red]错误：训练状态为None！[/bold red]")
    #     # 创建空的训练状态对象
    #     train_stat = TrainingStats()
    
    # 保存训练数据
    train_stat.save_training_data(args2.model_save_label)
    
    # 关闭日志文件
    Logger.close()

    return train_stat



def test(mode, params, model: Ser_Model, criterion_ce: nn.Module, criterion_mml: nn.Module, test_dataset: TestDataset, batch_size, device,
         return_matrix=False):

    """Test an SER model.

    Parameters
    ----------
    model
        PyTorch model
    criterion
        loss_function
    test_dataset
        The test dataset
    batch_size : int
    device
    return_matrix : bool
        Whether to return the confusion matrix.

    Returns
    -------
    loss, weighted accuracy (WA), unweighted accuracy (UA), confusion matrix 
       

    """
    with torch.no_grad():
        total_loss = 0
        test_preds_segs = []
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        # we'll store the features as NumPy array of size num_images x feature_size and the labels
        sne_features = [None, None, None, None]
        out_features = None
        sne_labels = []
            
        model.eval()
        
        for batch_idx, test_batch in enumerate(tqdm(test_loader, desc=f"{'TEST' if mode == 'TEST' else 'VAL'}", leave=False)):
            # Send data to correct device
            test_data_spec_batch = test_batch['seg_spec'].to(device)
            test_data_mfcc_batch = test_batch['seg_mfcc'].to(device)
            test_data_audio_batch = test_batch['seg_audio'].to(device)
            test_labels_batch =  test_batch['seg_label'].to(device,dtype=torch.long)
        
            labels = test_batch['seg_label'].cpu().detach().numpy()
            
            # sne_labels: [3739] = 3712 + 27 = 64 * 58 + 27
            sne_labels += list(labels)
            
            # Forward
            test_outputs = model(test_data_spec_batch, test_data_mfcc_batch, test_data_audio_batch)
            test_preds_segs.append(f.log_softmax(test_outputs['M'], dim=1).cpu())

            #test loss
            test_loss_ce = criterion_ce(test_outputs['M'], test_labels_batch)
            # test_loss_mml = criterion_mml(test_outputs['M'], test_labels_batch)
            test_loss = test_loss_ce#  + test_loss_mml
        
            total_loss += test_loss.item()
     
        # 显示模式信息    
        if mode == 'TEST':
            Logger.log(Panel(f"[bold green]{'>>>>>>>>>>>'} 正在进行test测试 {'<<<<<<<<<<<'}", border_style="green", expand=False))
        elif mode == 'VAL':
            Logger.log(Panel(f"[bold yellow]{'>>>>>>>>>>>'} 正在进行val测试 {'<<<<<<<<<<<'}", border_style="yellow", expand=False))
        
        # Average loss
        test_loss = total_loss / len(test_loader)

        # Accumulate results for val data
        test_preds_segs = np.vstack(test_preds_segs)
        test_preds = test_dataset.get_preds(test_preds_segs)
        
        # Make sure everything works properly
        assert len(test_preds) == test_dataset.n_actual_samples
        test_wa = test_dataset.weighted_accuracy(test_preds)
        test_ua = test_dataset.unweighted_accuracy(test_preds)
        
        # Calculate weighted F1 score
        test_f1 = f1_score(test_dataset.actual_target, test_preds, average='weighted') * 100
        test_uf1 = f1_score(test_dataset.actual_target, test_preds, average='macro') * 100
        
        results = (test_loss, test_wa*100, test_ua*100, test_f1)
        
        if return_matrix:
            # 创建混淆矩阵
            if DATABASE_KIND == "IEMOCAP":
                test_conf = test_dataset.confusion_matrix_iemocap(test_preds)
            elif DATABASE_KIND == "MELD":
                test_conf = test_dataset.confusion_matrix_meld(test_preds)
            # 确保返回的是混淆矩阵numpy数组，而不是元组
            return results, test_conf, test_uf1
        else:
            return results,test_uf1

# scale and move the coordinates so they fit [0; 1] range
def scale_to_01_range(x):
    # compute the distribution range
    value_range = (np.max(x) - np.min(x))

    # move the distribution so that it starts from zero
    # by extracting the minimal value from all its values
    starts_from_zero = x - np.min(x)

    # make the distribution fit [0; 1] by dividing by its range
    return starts_from_zero / value_range
    
def visualize_tsne_points_2(name, tx, ty, labels, params):
    # initialize matplotlib plot
    fig = plt.figure()
    ax = fig.add_subplot(111)

    # for every class, we'll add a scatter plot separately
    for label in COLORS_PER_CLASS:
        # find the samples of the current class in the data
        indices = [i for i, l in enumerate(labels) if l == label]

        # extract the coordinates of the points of this class only
        current_tx = np.take(tx, indices)
        current_ty = np.take(ty, indices)

        # convert the class color to matplotlib format:
        # BGR -> RGB, divide by 255, convert to np.array
        color = np.array([COLORS_PER_CLASS[label][::-1]], dtype=np.float) / 255

        # add a scatter plot with the correponding color and label
        ax.scatter(current_tx, current_ty, s=1, c=color, label=label)

    # build a legend using the labels we set previously
    ax.legend(loc='best')
    plt.show()
    
    t = round(time.time()*1000)
    t_str = time.strftime('%H_%M_%S',time.localtime(t/1000))

    img_path = './results/t-SNE/' + t_str + '_' + name + '_' + params['repeat_idx'] + '_' + params['test_id'] + '.png'
    # finally, show the plot
    fig.savefig(img_path, dpi=fig.dpi)
    
def visualize_tsne_points_3(name, tx, ty, tz, labels, params):
    # initialize matplotlib plot
    fig = plt.figure()
    ax = Axes3D(fig)
    # ax = fig.add_subplot(111, projection='3d')

    # for every class, we'll add a scatter plot separately
    for label in COLORS_PER_CLASS:
        # find the samples of the current class in the data
        indices = [i for i, l in enumerate(labels) if l == label]

        # extract the coordinates of the points of this class only
        current_tx = np.take(tx, indices)
        current_ty = np.take(ty, indices)
        current_tz = np.take(tz, indices)

        # convert the class color to matplotlib format:
        # BGR -> RGB, divide by 255, convert to np.array
        color = np.array([COLORS_PER_CLASS[label][::-1]], dtype=np.float) / 255

        # add a scatter plot with the correponding color and label
        ax.scatter(current_tx, current_ty, current_tz, s=4, c=color, label=label)

    # build a legend using the labels we set previously
    ax.legend(loc='best')
    
    t = round(time.time()*1000)
    t_str = time.strftime('%H_%M_%S',time.localtime(t/1000))

    img_path = './results/t-SNE/' + t_str + '_' + name + '_' + params['repeat_idx'] + '_' + params['test_id'] + '.png'
    Logger.log(img_path)
    # finally, show the plot
    fig.savefig(img_path, dpi=fig.dpi)
        
def visualize_tsne_2(name, tsne, labels, params):

    # extract x and y coordinates representing the positions of the images on T-SNE plot
    tx = tsne[:, 0]
    ty = tsne[:, 1]

    # scale and move the coordinates so they fit [0; 1] range
    tx = scale_to_01_range(tx)
    ty = scale_to_01_range(ty)

    # visualize the plot: samples as colored points
    visualize_tsne_points_2(name, tx, ty, labels, params)

def visualize_tsne_3(name, tsne, labels, params):

    # extract x and y coordinates representing the positions of the images on T-SNE plot
    tx = tsne[:, 0]
    ty = tsne[:, 1]
    tz = tsne[:, 2]

    # scale and move the coordinates so they fit [0; 1] range
    tx = scale_to_01_range(tx)
    ty = scale_to_01_range(ty)
    tz = scale_to_01_range(tz)

    # visualize the plot: samples as colored points
    visualize_tsne_points_3(name, tx, ty, tz, labels, params)
    
def save_best_model(model: Ser_Model, train_stat: TrainingStats, val_wa, val_ua, val_f1, val_loss, save_path, epoch):
    """
    如果当前模型在验证集上的性能优于之前的最佳模型，则更新最佳模型信息并保存模型
    
    Parameters
    ----------
    model : Ser_Model
        当前模型
    train_stat : TrainingStats
        训练状态对象
    val_wa : float
        验证集加权准确率
    val_ua : float
        验证集非加权准确率
    val_f1 : float
        验证集F1分数
    val_loss : float
        验证集损失
    save_path : str
        模型保存路径
    epoch : int
        当前训练轮次
        
    Returns
    -------
    bool
        如果模型被更新则返回True，否则返回False
    """
    if val_wa + val_ua > train_stat.best_val_acc:
        train_stat.update_best_info(val_wa, val_ua, val_f1, val_loss, val_wa + val_ua, epoch)
        Logger.log(Panel("[bold red]发现更好的模型! 正在保存...[/bold red]", border_style="red", expand=False))
        
        if save_path is not None:
            # === 内存优化保存方案 ===
            try:
                # 1. 释放模型计算图中的中间变量
                for param in model.parameters():
                    param.grad = None  # 清除梯度缓存
                torch.cuda.empty_cache()  # 清空显存缓存
                
                # 2. 使用低内存模式保存
                with torch.no_grad():
                    # 在CPU上创建状态字典副本
                    cpu_state_dict = {k: v.cpu().detach().clone() for k, v in model.state_dict().items()}
                
                # 3. 保存到临时文件
                temp_path = save_path + f".tmp.{os.getpid()}"
                torch.save(cpu_state_dict, temp_path)
                
                # 4. 原子操作替换文件
                if os.path.exists(save_path):
                    os.remove(save_path)
                os.rename(temp_path, save_path)
                
                # 5. 释放内存
                del cpu_state_dict
                import gc
                gc.collect()
                
            except Exception as e:
                Logger.log(f"[bold red]保存模型失败: {str(e)}[/bold red]")
                # 回退到简单保存
                torch.save(model.state_dict(), save_path)
            # === 内存优化结束 ===
        return True
    return False

# 创建新的训练状态对象的辅助函数
def create_new_training_stats(args2) -> TrainingStats:
    return TrainingStats(
        epoch=0,
        best_epoch=0,
        train_loss_history=[],
        train_wa_history=[],
        train_ua_history=[],
        val_loss_history=[],
        val_wa_history=[],
        val_ua_history=[],
        val_f1_history=[],
        test_loss=0,
        test_wa=0,
        test_ua=0,
        test_f1=0,
        confusion_matrix=None,
        best_val_wa=0,
        best_val_ua=0,
        best_val_f1=0,
        best_val_acc=-1e8,
        best_val_loss=1e8,
        loss_format="{:.04f}",
        acc_format="{:.02f}%",
        acc_format2="{:.02f}",
        current_lr=args2.lr,
        train_loss=0,
        train_wa=0,
        train_ua=0
    )

def train(args2: Args2):
    from tools import TrainingStats
    
    # 首先定义 device
    device = torch.device(f"cuda:{args2.gpu_id}" if args2.use_gpu else "cpu")
    torch.cuda.set_device(device)
    
    # 显示cuda信息
    display_cuda_info()
    
    # 创建模型和优化器（必须在检查点恢复前定义）
    model = Ser_Model(args2=args2).to(device) 
    optimizer = optim.AdamW(model.parameters(), lr=args2.lr, foreach=False)
    
    # 设置训练状态对象
    train_stat = None
    
    # === 检查点恢复机制 ===
    checkpoint_path = os.path.join(args2.model_save_prefix, f"{args2.model_save_label}_checkpoint.pth")
    
    if os.path.exists(checkpoint_path):
        Logger.log(f"[bold yellow]发现检查点文件: {checkpoint_path}[/bold yellow]")
        Logger.log("[bold yellow]尝试恢复训练状态...[/bold yellow]")
        
        try:
            # 尝试安全加载（PyTorch 2.6+）
            from torch.serialization import safe_globals
            # 创建新的安全全局集合
            custom_safe_globals = set(safe_globals)
            # 添加自定义类到安全列表
            custom_safe_globals.add(TrainingStats)
            
            # 使用自定义安全列表加载检查点
            with torch.serialization._use_safe_globals(custom_safe_globals):
                checkpoint = torch.load(checkpoint_path, map_location=device)
            
            # 恢复模型和优化器
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            # 创建训练状态对象
            train_stat = create_new_training_stats(args2)
            if 'train_stat' in checkpoint:
                train_stat.__dict__.update(checkpoint['train_stat'])
                Logger.log(f"成功加载训练状态 (epoch {checkpoint['epoch']})")
            
            start_epoch = checkpoint['epoch'] + 1  # 从下一个epoch开始
            Logger.log(f"将从第 {start_epoch} 轮继续训练")
            
        except Exception as e:
            Logger.log(f"[bold red]安全加载失败: {str(e)}[/bold red]")
            Logger.log("[bold yellow]尝试非安全加载（weights_only=False）...[/bold yellow]")
            try:
                # 回退到非安全加载
                checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
                
                # 恢复模型和优化器
                model.load_state_dict(checkpoint['model_state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                
                # 创建训练状态对象
                train_stat = create_new_training_stats(args2)
                if 'train_stat' in checkpoint:
                    train_stat.__dict__.update(checkpoint['train_stat'])
                    Logger.log(f"成功加载训练状态 (epoch {checkpoint['epoch']})")
                
                start_epoch = checkpoint['epoch'] + 1
                Logger.log(f"将从第 {start_epoch} 轮继续训练")
                
            except Exception as e2:
                Logger.log(f"[bold red]非安全加载失败: {str(e2)}[/bold red]")
                Logger.log("[bold yellow]重置训练状态，从头开始训练[/bold yellow]")
                train_stat = create_new_training_stats(args2)
                start_epoch = 0
    else:
        # 创建新的训练状态对象
        train_stat = create_new_training_stats(args2)
        start_epoch = 0
        Logger.log("[bold green]未发现检查点，开始新训练[/bold green]")
    
    # 加载数据集
    with open(args2.features_file_path, "rb") as fin:
        features_data = pickle.load(fin)
    
    # 创建SERDataset对象
    dataset = SERDataset(
        features_data,
        val_speaker_id=args2.val_id,
        test_speaker_id=args2.test_id,
        oversample=args2.oversampling,
        num_classes=NUM_CLASSES,
    )
    
    # 立即释放features_data内存
    del features_data
    import gc
    gc.collect()  # 强制进行垃圾回收
    
    # 打印模型参数
    display_trainable_parameters_count(model)

    # 设置损失函数
    criterion_mml: nn.Module = get_loss_function("multi_margin") 
    criterion_focal: nn.Module = get_loss_function("focal")
    
    # 获取数据集
    train_dataset = dataset.get_train_dataset()                            
    val_dataset = dataset.get_val_dataset()
    test_dataset = dataset.get_test_dataset()
    
    # 销毁dataset对象,释放内存
    del dataset
    gc.collect()  # 再次强制垃圾回收

    # 创建数据加载器
    train_loader: DataLoader[dict[str, torch.Tensor]] = DataLoader(train_dataset, batch_size=args2.batch_size, shuffle=True)  
    
    # 模型保存初始化
    os.makedirs(args2.model_save_prefix, exist_ok=True)
    save_path = os.path.join(args2.model_save_prefix, args2.model_save_label + '.pth')

    # 创建数据集信息表格
    table = Table(title="数据集信息", show_header=True, header_style="bold magenta")
    table.add_column("数据集", style="cyan")
    table.add_column("样本数", style="green")
    table.add_column("批次数", style="yellow")
    
    table.add_row("train集", str(len(train_dataset)), str(len(train_loader)))
    table.add_row("val集", str(len(val_dataset)), "-")
    table.add_row("test集", str(len(test_dataset)), "-")
    
    # 使用Panel包装表格并显示
    panel = Panel(
        table,
        title=f"[bold cyan]开始训练 - 共{args2.num_epochs}个epoch[/bold cyan]",
        border_style="blue",
        expand=False
    )
    
    # 使用Logger输出面板
    Logger.log_panel(panel)
       
    # 开始训练
    for epoch in range(start_epoch, args2.num_epochs):
        # Logger.log("[bold yellow]触发早停: 模型性能没有改善![/bold yellow]", style="yellow")
        # break
        # 说明：把上面两行的注释取消，就能跳过训练直接进入最终测试，但注意会把training_logs里的训练数据覆盖，最好提前备份
        
        # 每10轮执行完整清理
        if epoch % 10 == 0:
            Logger.log("[yellow]执行深度内存清理...[/yellow]")
            model.to('cpu')
            torch.cuda.empty_cache()
            import gc
            gc.collect()
            time.sleep(1)
            model.to(device)
        
        # 动态调整学习率
        # adjust_learning_rate(args2.lr, optimizer, epoch)
        
        # 获取当前学习率
        for param_group in optimizer.param_groups:
            train_stat.current_lr = param_group['lr']
        
        # 计算实际显示的轮次（全局编号）
        global_epoch = epoch + 1
        
        Logger.log("\n\n")
        # 显示当前轮次信息
        Logger.log(f"[bold green]================= 第 {global_epoch}/{args2.num_epochs} 轮 (学习率: {train_stat.current_lr:.8f}) =================")
        
        # 显示当前时间
        current_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
        Logger.log(f"[bold blue]当前时间: {current_time}[/bold blue]")
            
        # 临时变量
        total_loss = 0
        train_preds = []
        train_labels = []
        
        # 启动训练模式
        model.train()
        
        # 批次训练 - 使用tqdm进度条
        pbar = tqdm(train_loader, desc=f"epoch {epoch+1}/{args2.num_epochs}", leave=False)
        for batch_idx, train_batch in enumerate(pbar):
            # train_batch类型注解
            train_batch: dict[str, torch.Tensor]
            
            # 清空梯度
            optimizer.zero_grad()
            
            # 将数据发送到正确的设备
            train_data_spec_batch = train_batch['seg_spec'].to(device)
            train_data_mfcc_batch = train_batch['seg_mfcc'].to(device)
            train_data_audio_batch = train_batch['seg_audio'].to(device)
            train_labels_batch = train_batch['seg_label'].to(device, dtype=torch.long)
            
            # 前向传播
            outputs = model(train_data_spec_batch, train_data_mfcc_batch, train_data_audio_batch)
            
            # 获取预测结果，并确保展平为一维数组
            train_preds.append(f.log_softmax(outputs['M'], dim=1).cpu().detach().numpy())
            
            # 将标签转为一维数组
            train_labels.append(train_labels_batch.cpu().detach().numpy().flatten())
            
            # 计算损失,梯度,更新参数
            train_loss_focal: torch.Tensor = criterion_focal(outputs['M'], train_labels_batch)
            
            #! 计算hifi-gan损失
            # train_loss_hifigan: torch.Tensor = criterion_hifigan(feature=outputs['F2'], audio=train_data_audio_batch)
            
            train_loss: torch.Tensor = train_loss_focal
            # train_loss: torch.Tensor = train_loss_focal + train_loss_hifigan
            train_loss.backward()
            total_loss += train_loss.item()
            
            try:
                optimizer.step()
            except RuntimeError as e:
                if "state_steps" in str(e):
                    # 手动更新步骤计数器
                    for param_group in optimizer.param_groups:
                        for p in param_group['params']:
                            if p.grad is not None:
                                state = optimizer.state[p]
                                if 'step' not in state:
                                    state['step'] = torch.tensor(0, device=p.device)
                                else:
                                    state['step'] += 1
                    # 重试步骤
                    optimizer.step()
                else:
                    raise
            
            # === 添加内存清理 ===
            # 删除中间变量以释放内存
            del train_data_spec_batch, train_data_mfcc_batch, train_data_audio_batch, outputs
            del train_loss, train_loss_focal
            
            # 如果使用GPU，清空缓存
            if args2.use_gpu:
                torch.cuda.empty_cache()
            # === 内存清理结束 ===
            
            # 更新进度条信息
            avg_loss = total_loss / (batch_idx + 1)
            pbar.set_postfix({'loss': f'{avg_loss:.4f}'})
        
        # 评估训练数据
        train_loss = total_loss / len(train_loader)
        
        # 累积训练数据的结果
        train_preds = np.vstack(train_preds)
        train_preds = train_dataset.get_preds(train_preds)
        
        # 累加训练标签并确保是一维数组
        train_labels = np.concatenate(train_labels)
        
        # 计算训练数据指标
        train_wa = train_dataset.weighted_accuracy(train_preds, train_labels) * 100
        train_ua = train_dataset.unweighted_accuracy(train_preds, train_labels) * 100
        
        # 更新训练状态对象的训练指标
        train_stat.update_train_history(train_loss, train_wa, train_ua)
        
        #! VAL
        (val_loss, val_wa, val_ua, val_f1), confusion_matrix,test_uf1 = test(
            'VAL', 
            args2, model, criterion_focal, criterion_mml, val_dataset, 
            batch_size=64, 
            device=device, return_matrix=True
        )
        
        # 更新训练状态对象的验证指标
        train_stat.update_val_history(val_loss, val_wa, val_ua, val_f1)

        # 保存最佳模型
        is_best = save_best_model(model, train_stat, val_wa, val_ua, val_f1, val_loss, save_path, epoch)
        
        # 显示验证结果
        train_stat.display_val_process(val_loss, val_wa, val_ua, val_f1)

        #! TEST
        (test_loss, test_wa, test_ua, test_f1), confusion_matrix,test_uf1 = test(
            'TEST', 
            args2, model, criterion_focal, criterion_mml, test_dataset, 
            batch_size=64, 
            device=device, return_matrix=True
        )
        
        # 更新训练状态对象的当前epoch
        train_stat.epoch = epoch
        
        # 显示测试结果
        train_stat.display_test_process(test_loss, test_wa, test_ua, test_f1,test_uf1, confusion_matrix)
        
        # 显示训练过程
        train_stat.display_train_process(train_loss, val_loss, val_wa, val_ua)
        
        # 显示最佳模型信息
        best_info = f"VAL最佳epoch: {train_stat.best_epoch+1} (WA={train_stat.best_val_wa:.2f}%, UA={train_stat.best_val_ua:.2f}%)"
        if is_best:
            best_info = f"[bold green]{best_info} (新)[/bold green]"
        
        # 显示epoch摘要
        summary_table = Table(show_header=True, header_style="bold magenta")
        summary_table.add_column("指标", style="cyan")
        summary_table.add_column("train", style="green") 
        summary_table.add_column("val", style="yellow")
        summary_table.add_column("test", style="red")
        
        # 设置当前轮次
        train_stat.epoch = epoch  # 使用0起始索引

        # 添加损失值
        summary_table.add_row(
            "Loss",
            f"{train_loss:.4f}",
            f"{val_loss:.4f}", 
            f"{test_loss:.4f}"
        )

        # 添加准确率指标
        summary_table.add_row(
            "WA",
            f"{train_wa:.2f}%",
            f"{val_wa:.2f}%",
            f"{test_wa:.2f}%"
        )
        summary_table.add_row(
            "UA", 
            f"{train_ua:.2f}%",
            f"{val_ua:.2f}%",
            f"{test_ua:.2f}%"
        )
        summary_table.add_row(
            "F1分数",
            "-", 
            f"{val_f1:.2f}%",
            f"{test_f1:.2f}%"
        )

        # 添加最佳模型信息
        summary_table.add_row("VAL最佳", best_info, "", "")

        Logger.log(Panel(
            summary_table,
            title=f"[bold blue]第 {global_epoch}/{args2.num_epochs} 轮训练总结[/bold blue]",
            border_style="blue",
            expand=False
        ))

        # 保存检查点（包含当前训练状态）
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            
            # 替换为可序列化的数据字典
            'train_stat': train_stat.__dict__,
            'args2': vars(args2)  # 保存参数配置
        }
        torch.save(checkpoint, checkpoint_path)
        Logger.log(f"[bold blue]检查点已保存 (epoch {epoch})[/bold blue]")

        # 设置早停机制
        if (train_stat.epoch - train_stat.best_epoch >= args2.early_stop) and (train_stat.epoch > 5):
            Logger.log("[bold yellow]触发早停: 模型性能没有改善![/bold yellow]", style="yellow")
            break
            
    # 最终测试最好的模型
    Logger.log(Panel(f"[bold green]{'#' * 10} 测试最好的模型在test上的性能 {'#' * 10}", border_style="green", expand=False))
    model.load_state_dict(torch.load(save_path))
    
    # 进行最终测试
    (test_loss, test_wa, test_ua, test_f1), confusion_matrix,test_uf1 = test(
        'TEST', 
        args2, model, criterion_focal, criterion_mml, test_dataset, 
        batch_size=64,
        device=device, return_matrix=True
    )
    
    # 显示最终测试结果
    train_stat.display_test_process(test_loss, test_wa, test_ua, test_f1,test_uf1, confusion_matrix)
    
    # 更新训练状态对象的最终测试指标
    train_stat.update_test_history(test_loss, test_wa, test_ua, test_f1,test_uf1, confusion_matrix)
    
    # 显示训练完成
    Logger.log("[bold green]训练完成![/bold green]")
    
    # === 训练完成后删除检查点 ===
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)
        Logger.log("[bold green]检查点已删除![/bold green]")

    return train_stat


if __name__ == '__main__':
    main(parse_args_2(sys.argv[1:]))

