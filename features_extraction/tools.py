import argparse
import os
import numpy as np
import random
import warnings
from dataclasses import dataclass
import sys
from database import SER_Database_Base
import pandas as pd
from collections import Counter


@dataclass
class Args:
    dataset: str = "IEMOCAP"
    features: str = "logspec"
    window: str = "hamming"
    win_length: float = 40
    hop_length: float = 10
    ndft: int = 800
    nfreq: int = 200
    nmel: int = 128
    segment_size: int = 300
    mixnoise: bool = False
    save_dir: str = "./"
    save_label: str = "1"

def parse_arguments(argv):
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    #DATASET
    parser.add_argument('--dataset', type=str, default=Args.dataset,
        help='Dataset to extract features. Options:'
             '  - IEMOCAP (default)'
             '  - MELD')
    
    #FEATURES
    parser.add_argument('--features', type=str, default=Args.features,
        help='Feature to be extracted. Options:'
             '  - logspec (default) : (1 ch.)log spectrogram'
             '  - logmelspec        : (1 ch.)log mel spectrogram')
    
    parser.add_argument('--window', type=str, default=Args.window,
        help='Window type. Default: hamming')

    parser.add_argument('--win_length', type=float, default=Args.win_length,
        help='Window size (msec). Default: 40')

    parser.add_argument('--hop_length', type=float, default=Args.hop_length,
        help='Window hop size (msec). Default: 10')
    
    parser.add_argument('--ndft', type=int, default=Args.ndft,
        help='DFT size. Default: 800')

    parser.add_argument('--nfreq', type=int, default=Args.nfreq,
        help='Number of lowest DFT points to be used as features. Default: 200'
             '  Only effective for <logspec, lognrevspec> features')
    
    parser.add_argument('--nmel', type=int, default=Args.nmel,
        help='Number of mel frequency bands used as features. Default: 128'
             '  Only effectice for <logmel, logmeldeltaspec> features')
    
    parser.add_argument('--segment_size', type=int, default=Args.segment_size,
        help='Size of each features segment')

    parser.add_argument('--mixnoise', action='store_true',
        help='Set this flag to mix with noise.')
    

    #FEATURES FILE
    parser.add_argument('--save_dir', type=str, default=Args.save_dir,
        help='Path to directory to save the extracted features.')
    
    parser.add_argument('--save_label', type=str, default=Args.save_label,
        help='Label to save the feature')

    args = parser.parse_args(argv)
    # 转换为带类型的对象
    return Args(
        dataset=args.dataset,
        features=args.features,
        window=args.window,
        win_length=args.win_length,
        hop_length=args.hop_length,
        ndft=args.ndft,
        nfreq=args.nfreq,
        nmel=args.nmel,
        segment_size=args.segment_size,
        mixnoise=args.mixnoise,
        save_dir=args.save_dir,
        save_label=args.save_label
    )


def seed_everything(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    #torch.manual_seed(seed)
    #torch.cuda.manual_seed(seed)
    #torch.backends.cudnn.deterministic = True


def ignore_warnings():
    """
    忽略相关的警告信息
    """
    # 忽略 UserWarning: PySoundFile failed 警告
    warnings.filterwarnings("ignore", message="PySoundFile failed. Trying audioread instead.")
    
    # 忽略 FutureWarning: librosa.core.audio.__audioread_load 警告
    warnings.filterwarnings("ignore", category=FutureWarning, message="librosa.core.audio.__audioread_load.*")


def get_output_filepath(args: Args, check_exist: bool = True):
    """
    获取输出文件路径，以及是否存在这个文件
    输出文件命名格式：{save_dir} / {dataset}_features_{save_label}.pkl
    """
    filepath = os.path.join(args.save_dir, args.dataset + "_features_" + args.save_label + ".pkl")
    
    if check_exist:
        if os.path.exists(filepath):
            print(f"文件 {filepath} 已存在。")
            response = input("是否要覆盖该文件? [y/N] ")
            if response.lower() != 'y':
                print("退出程序...")
            sys.exit(0)
    
    return filepath, os.path.exists(filepath)


def get_feature_extraction_params(args: Args, out_filename: str, display_params: bool = True):
    #Get spectrogram parameters
    params={
        'window'        : args.window,
        'win_length'    : args.win_length,
        'hop_length'    : args.hop_length,
        'ndft'          : args.ndft,
        'nfreq'         : args.nfreq,
        'nmel'          : args.nmel,
        'segment_size'  : args.segment_size,
        'mixnoise'      : args.mixnoise
    }
    
    if display_params:
        print('\n')
        print('*'*50)
        print('\nFEATURES EXTRACTION')
        print(f'\t{"Dataset":>20}: {args.dataset}')
        print(f'\t{"Features":>20}: {args.features}')
        print(f'\t{"Features file":>20}: {out_filename}')
        print(f'\t{"Add noise version":>20}: {args.mixnoise}')
        print(f"\nPARAMETERS:")
    
        for key in params:
            print(f'\t{key:>20}: {params[key]}')
        print('\n')

    return params


def display_statistics(database: SER_Database_Base, features_data: dict, args: Args):
    """
    显示数据集特征的统计信息，包括每个说话者的情感类别分布
    
    参数:
        database: 数据集对象，提供情感类别映射
        features_data: 提取的特征数据字典
        args: 参数对象
    """
    print(f'\nSEGMENT CLASS DISTRIBUTION PER SPEAKER:\n')
    
    # 获取情感类别映射
    classes = database.get_classes()
    
    # 统计变量初始化
    n_speaker = len(features_data)
    n_class = len(classes)
    class_dist = np.zeros((n_speaker, n_class), dtype=np.int32)
    speakers = []
    data_shape = []
    
    # 遍历每个说话者的数据
    for i, speaker in enumerate(features_data.keys()):
        # 获取该说话者的情感标签计数
        label_counts = sorted(Counter(features_data[speaker]["seg_label"]).items())
        
        # 填充到统计矩阵中
        for class_id, count in label_counts:
            class_dist[i][class_id] = count
        
        # 保存说话者ID
        speakers.append(speaker)
        
        # 获取数据形状
        if args.mixnoise:
            # 针对混合噪声模式的数据形状
            data_shape.append(str(features_data[speaker]["seg_spec"][0].shape))
        else:
            # 正常模式的数据形状
            data_shape.append(str(features_data[speaker]["seg_spec"].shape))
    
    # 准备数据框架
    df_data = {
        "speakerID": speakers,
        "shape (N,C,F,T)": data_shape
    }
    
    # 为每个情感类别添加计数列
    for class_id in range(n_class):
        df_data[classes[class_id]] = class_dist[:, class_id]
    
    # 创建并显示数据框架
    class_distribution = pd.DataFrame(df_data)
    print(class_distribution.to_string(index=False))
    
    # 打印分隔符
    print('\n')
    print('*' * 50)
    print('\n')
