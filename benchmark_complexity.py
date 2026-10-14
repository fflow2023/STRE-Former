# benchmark_complexity.py
"""
================================================================================
【模型复杂度与推理时间对比分析】

优化策略：
  1. 推理时间：使用 batch 推理 (batch_size=16)
  2. 显存测量：先清空缓存再测，确保测的是净增量
  3. 输出格式：生成 LaTeX 表格代码 + 图表
  4. warm-up：默认 50 次，正式 200 次，取中位数（去除离群值影响）

使用方式：
  python benchmark_complexity.py \
      --features_file ./features_extraction/IEMOCAP_features_4class_fu_new.pkl \
      --checkpoints "Ours=./saved_models/model_FPE.pth" \
                    "APE=./saved_models/model_APE.pth" \
                    "ALiBi=./saved_models/model_ALiBi.pth" \
      --batch_size 16 \
      --num_warmup 50 \
      --num_runs 200
================================================================================
"""

import argparse
import pickle
import numpy as np
import torch
import time
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def load_model_for_benchmark(checkpoint_path, device):
    """复用 plot_attention_decay.py 中的解析逻辑加载模型"""
    from plot_attention_decay import parse_config_from_filename, load_model
    model, name = load_model(checkpoint_path, device)
    return model, name


def count_parameters(model):
    """统计模型总参数量和可训练参数量"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


def count_pe_parameters(model):
    """只统计位置编码相关模块的参数量（去除共享 backbone 的影响）"""
    pe_params = 0
    pe_keywords = ['embed_positions', 'position', 'pos_emb', 'alibi', 'fib', 'rope',
                   'transformer_encoder_spec', 'transformer_encoder_mfcc', 'transformer_encoder_wav2vec']
    shared_keywords = ['wav2vec2_model', 'alexnet', 'post_spec', 'post_mfcc', 'post_wav',
                       'post_att', 'embedding_layer']
    for name, param in model.named_parameters():
        # 属于共享模块则跳过
        is_shared = any(kw in name for kw in shared_keywords)
        if not is_shared:
            # 属于编码器模块则计入
            is_pe = any(kw in name for kw in pe_keywords)
            if is_pe:
                pe_params += param.numel()
    return pe_params


def estimate_flops(model, batch_input, device):
    """估算 FLOPs（优先 thop，回退到 fvcore）"""
    spec, mfcc, wav = batch_input
    spec, mfcc, wav = spec.to(device), mfcc.to(device), wav.to(device)

    # 用 batch_size=1 估算单样本 FLOPs
    s1 = spec[:1]
    m1 = mfcc[:1]
    w1 = wav[:1]

    try:
        from thop import profile
        flops, _ = profile(model, inputs=(s1, m1, w1), verbose=False)
        return flops
    except ImportError:
        pass
    except Exception as e:
        print(f"thop 估算失败: {e}")

    try:
        from fvcore.nn import FlopCountAnalysis
        flop_counter = FlopCountAnalysis(model, (s1, m1, w1))
        return flop_counter.total()
    except ImportError:
        pass
    except Exception as e:
        print(f"fvcore 估算失败: {e}")

    # 手动粗估
    print("未安装 thop/fvcore，使用粗略估算")
    total_flops = 0
    for module in model.modules():
        if isinstance(module, torch.nn.Linear):
            total_flops += 2 * module.in_features * module.out_features
        elif isinstance(module, torch.nn.Conv2d):
            total_flops += 2 * module.in_channels * module.out_channels * \
                          module.kernel_size[0] * module.kernel_size[1]
    return total_flops


def measure_inference_time(model, batch_input, device, num_warmup=50, num_runs=200):
    """
    使用 batch 推理测量推理时间，取中位数（更稳定，不受离群值影响）。
    返回：每样本推理时间（毫秒）的中位数和 IQR。
    """
    spec, mfcc, wav = batch_input
    spec, mfcc, wav = spec.to(device), mfcc.to(device), wav.to(device)
    batch_size = spec.shape[0]

    model.eval()
    with torch.no_grad():
        # Warm-up（充分预热 CUDA kernel 和 JIT 缓存）
        for _ in range(num_warmup):
            _ = model(spec, mfcc, wav)
            if device.type == 'cuda':
                torch.cuda.synchronize()

        # 正式计时
        times = []
        for _ in range(num_runs):
            if device.type == 'cuda':
                torch.cuda.synchronize()
            start = time.perf_counter()
            _ = model(spec, mfcc, wav)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            end = time.perf_counter()
            # 转换为每样本毫秒
            times.append((end - start) * 1000 / batch_size)

    times = np.array(times)
    # 去掉前 5% 和后 5% 的离群值
    p5, p95 = np.percentile(times, [5, 95])
    trimmed = times[(times >= p5) & (times <= p95)]

    return np.median(trimmed), np.std(trimmed)


def measure_gpu_memory(model, batch_input, device):
    """测量 GPU 峰值显存占用 (MB)，先彻底清缓存再测"""
    if device.type != 'cuda':
        return 0.0

    spec, mfcc, wav = batch_input
    spec, mfcc, wav = spec.to(device), mfcc.to(device), wav.to(device)

    # 先彻底清理
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    model.eval()
    with torch.no_grad():
        _ = model(spec, mfcc, wav)

    peak_mem = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
    return peak_mem


def make_batch_input(sample_input, batch_size):
    """将单个样本扩展为 batch"""
    spec, mfcc, wav = sample_input
    return (spec.unsqueeze(0).repeat(batch_size, *([1]*spec.dim())),
            mfcc.unsqueeze(0).repeat(batch_size, *([1]*mfcc.dim())),
            wav.unsqueeze(0).repeat(batch_size, *([1]*wav.dim())))


def main():
    print("=" * 60)
    print(">>  模型复杂度与推理时间对比分析 <<")
    print("=" * 60)

    parser = argparse.ArgumentParser(description='模型复杂度对比分析')
    parser.add_argument('--features_file', type=str, required=True, help='特征文件 (.pkl)')
    parser.add_argument('--checkpoints', nargs='+', required=True, help='名称=路径.pth')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch 推理 (默认16)')
    parser.add_argument('--num_warmup', type=int, default=50, help='预热次数 (默认50)')
    parser.add_argument('--num_runs', type=int, default=200, help='正式运行次数 (默认200)')
    args = parser.parse_args()

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"[环境] 运算设备: {device}, Batch Size: {args.batch_size}")

    # 加载样本数据
    print(f"[数据] 正在加载: {args.features_file}")
    with open(args.features_file, 'rb') as f:
        data = pickle.load(f)

    if isinstance(data, dict):
        key = list(data.keys())[0]
        item = data[key]
        spec = torch.tensor(item.get('spec', np.zeros((3, 256, 384))), dtype=torch.float32)
        mfcc = torch.tensor(item.get('mfcc', np.zeros((300, 40))), dtype=torch.float32)
        wav = torch.tensor(item.get('wav', item.get('audio', np.zeros(48000))), dtype=torch.float32)
    elif isinstance(data, list):
        item = data[0]
        spec = torch.tensor(item[0], dtype=torch.float32)
        mfcc = torch.tensor(item[1], dtype=torch.float32)
        wav = torch.tensor(item[2], dtype=torch.float32)
    else:
        raise ValueError("无法解析特征文件格式")

    sample_input = (spec, mfcc, wav)
    batch_input = make_batch_input(sample_input, args.batch_size)
    print(f"[数据] Batch 输入形状: spec={batch_input[0].shape}")
    print("-" * 60)

    # 收集结果
    results = []
    for item_str in args.checkpoints:
        if '=' in item_str:
            model_name, path = item_str.split('=', 1)
        else:
            path = item_str
            model_name = os.path.basename(path).split('.pth')[0]

        print(f"\n【分析目标】{model_name}")
        if not os.path.exists(path):
            print(f"模型文件不存在: {path}")
            continue

        try:
            model, _ = load_model_for_benchmark(path, device)

            # 1. 参数量
            total_params, trainable_params = count_parameters(model)
            pe_params = count_pe_parameters(model)
            print(f"总参数量: {total_params/1e6:.2f}M | 编码器模块: {pe_params/1e6:.2f}M")

            # 2. FLOPs
            flops = estimate_flops(model, batch_input, device)
            print(f"FLOPs (单样本): {flops/1e9:.2f}G")

            # 3. 推理时间 (batch 推理，取中位数)
            median_time, std_time = measure_inference_time(
                model, batch_input, device,
                num_warmup=args.num_warmup, num_runs=args.num_runs)
            print(f" (per sample, median): {median_time:.2f}ms")

            # 4. GPU 显存
            gpu_mem = measure_gpu_memory(model, batch_input, device)
            if gpu_mem > 0:
                print(f"GPU 峰值显存: {gpu_mem:.1f}MB")

            results.append({
                'name': model_name,
                'params_m': total_params / 1e6,
                'pe_params_m': pe_params / 1e6,
                'flops_g': flops / 1e9,
                'time_ms': median_time,
                'time_std': std_time,
                'gpu_mb': gpu_mem
            })

            del model
            if device.type == 'cuda':
                torch.cuda.empty_cache()

        except Exception as e:
            print(f"异常: {e}")
            import traceback
            traceback.print_exc()

    if not results:
        print("没有可用的结果")
        return

    # ==================== 输出结果 ====================
    output_dir = os.path.join(os.getcwd(), 'attention_plots')
    os.makedirs(output_dir, exist_ok=True)

    # 找到每列的最优值（最小值）
    best_params = min(r['params_m'] for r in results)
    best_flops = min(r['flops_g'] for r in results)
    best_time = min(r['time_ms'] for r in results)
    best_gpu = min(r['gpu_mb'] for r in results if r['gpu_mb'] > 0) if any(r['gpu_mb'] > 0 for r in results) else 0

    print("\n" + "=" * 70)
    print("结果")
    print("=" * 70)

    # ===== 1. Markdown 表格（加粗最优值）=====
    md_lines = ["# Model Complexity Comparison\n"]
    md_lines.append(f"> Batch Size = {args.batch_size}, Warm-up = {args.num_warmup}, Runs = {args.num_runs}\n")
    md_lines.append("| Method | Params (M) ↓ | FLOPs (G) ↓ | Latency (ms) ↓ | GPU Mem (MB) ↓ |")
    md_lines.append("|--------|:-----------:|:----------:|:-------------:|:-------------:|")

    for r in results:
        p = f"**{r['params_m']:.2f}**" if abs(r['params_m'] - best_params) < 0.01 else f"{r['params_m']:.2f}"
        f = f"**{r['flops_g']:.2f}**" if abs(r['flops_g'] - best_flops) < 0.01 else f"{r['flops_g']:.2f}"
        t = f"**{r['time_ms']:.2f}**" if abs(r['time_ms'] - best_time) < 0.5 else f"{r['time_ms']:.2f}"
        g = f"**{r['gpu_mb']:.1f}**" if (r['gpu_mb'] > 0 and abs(r['gpu_mb'] - best_gpu) < 1) else f"{r['gpu_mb']:.1f}"
        md_lines.append(f"| {r['name']} | {p} | {f} | {t} | {g} |")

    md_text = "\n".join(md_lines)
    print(md_text)

    # 保存 Markdown
    md_path = os.path.join(output_dir, 'complexity_comparison.md')
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write(md_text)
    print(f"\nMarkdown 表格: {md_path}")

    # ===== 2. LaTeX 表格 =====
    latex_lines = []
    latex_lines.append("% 可直接粘贴到 LaTeX 论文中")
    latex_lines.append("\\begin{table}[htbp]")
    latex_lines.append("  \\centering")
    latex_lines.append("  \\caption{Model Complexity Comparison}")
    latex_lines.append("  \\label{tab:complexity}")
    latex_lines.append("  \\begin{tabular}{lcccc}")
    latex_lines.append("    \\toprule")
    latex_lines.append("    Method & Params (M) $\\downarrow$ & FLOPs (G) $\\downarrow$ & Latency (ms) $\\downarrow$ & GPU Mem (MB) $\\downarrow$ \\\\")
    latex_lines.append("    \\midrule")
    for r in results:
        p = f"\\textbf{{{r['params_m']:.2f}}}" if abs(r['params_m'] - best_params) < 0.01 else f"{r['params_m']:.2f}"
        fl = f"\\textbf{{{r['flops_g']:.2f}}}" if abs(r['flops_g'] - best_flops) < 0.01 else f"{r['flops_g']:.2f}"
        t = f"\\textbf{{{r['time_ms']:.2f}}}" if abs(r['time_ms'] - best_time) < 0.5 else f"{r['time_ms']:.2f}"
        g = f"\\textbf{{{r['gpu_mb']:.1f}}}" if (r['gpu_mb'] > 0 and abs(r['gpu_mb'] - best_gpu) < 1) else f"{r['gpu_mb']:.1f}"
        name_escaped = r['name'].replace('_', '\\_')
        latex_lines.append(f"    {name_escaped} & {p} & {fl} & {t} & {g} \\\\")
    latex_lines.append("    \\bottomrule")
    latex_lines.append("  \\end{tabular}")
    latex_lines.append("\\end{table}")

    latex_text = "\n".join(latex_lines)

    latex_path = os.path.join(output_dir, 'complexity_comparison.tex')
    with open(latex_path, 'w', encoding='utf-8') as f:
        f.write(latex_text)
    print(f"LaTeX 表格: {latex_path}")

    # ===== 3. 论文级柱状图 =====
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        names = [r['name'] for r in results]
        n_models = len(results)

        # 颜色方案：我们的方法用突出色，其他用灰/淡色
        colors = []
        for r in results:
            if 'Ours' in r['name'] or 'F-FR' in r['name'] or 'Fib' in r['name']:
                colors.append('#E74C3C')  # 红色突出
            else:
                colors.append('#95A5A6')  # 灰色

        fig, axes = plt.subplots(1, 4, figsize=(18, 4.5))

        metrics = [
            ([r['params_m'] for r in results], 'Params (M)', 'Parameter Count'),
            ([r['flops_g'] for r in results], 'FLOPs (G)', 'FLOPs'),
            ([r['time_ms'] for r in results], 'Latency (ms)', 'Inference Latency'),
            ([r['gpu_mb'] for r in results], 'GPU Mem (MB)', 'GPU Memory'),
        ]

        for ax, (vals, ylabel, title) in zip(axes, metrics):
            bars = ax.bar(names, vals, color=colors, alpha=0.88, edgecolor='white', linewidth=1.5)
            ax.set_ylabel(ylabel, fontsize=11, fontweight='bold')
            ax.set_title(title, fontsize=12, fontweight='bold')
            ax.tick_params(axis='x', rotation=25, labelsize=9)
            ax.set_ylim(0, max(vals) * 1.2)
            ax.grid(axis='y', alpha=0.3)

            # 在柱子上方添加数值
            for bar, val in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(vals)*0.02,
                        f'{val:.1f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

            # 标注最优值（加星号）
            best_val = min(vals)
            for bar, val in zip(bars, vals):
                if abs(val - best_val) < best_val * 0.01:
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(vals)*0.08,
                            '★', ha='center', va='bottom', fontsize=12, color='#E74C3C')

        plt.suptitle('Model Complexity Comparison', fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()

        chart_path = os.path.join(output_dir, 'complexity_comparison.png')
        plt.savefig(chart_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✔ 柱状对比图: {chart_path}")
    except Exception as e:
        print(f"图表生成失败: {e}")
        import traceback
        traceback.print_exc()

    print("\n[完毕] 所有输出已保存到 attention_plots/ 文件夹。")


if __name__ == '__main__':
    main()
