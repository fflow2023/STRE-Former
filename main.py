import warnings
# 忽略 Hugging Face Transformers 的特定警告
warnings.filterwarnings("ignore", message="Some weights of the model checkpoint")
warnings.filterwarnings("ignore", message="You should probably TRAIN this model")

import train_ser
import sys
import pickle
import os
import time
from datetime import datetime
from config import *
from tools import *
from transformers import logging as transformers_logging

# 忽略 Hugging Face Transformers 的特定警告
transformers_logging.set_verbosity_error()

def run(args1: Args1):
    all_stat = []
    tmp_seed = RANDOM_SEED
    for repeat in range(REPEAT_TIMES):
        # 每次重复，都设置不同的随机数
        tmp_seed += (repeat*100)
        # 交叉验证「有可能就只有一次验证，并默认2M是test，2F是val」
        for v_id, t_id in list(zip(VAL_ID, TEST_ID)):
            # 配置训练参数
            train_ser.sys.argv = [
                'train_ser.py',
                '--repeat_idx', str(repeat),
                '--val_id',v_id, 
                '--test_id', t_id,
                '--use_gpu', str(USE_GPU),
                '--gpu_id', str(GPU_ID),
                '--num_epochs', str(NUM_EPOCHS),
                '--early_stop', str(EARLY_STOP),
                '--batch_size', str(BATCH_SIZE),
                '--lr', str(LR),
                '--seed', str(tmp_seed),
                '--model_save_label', str(MODEL_SAVE_LABEL),
                '--ratio', str(args1.ratio),
                '--pretrained',
                # '--coding_way', args1.coding_way,
                '--coding_way_spec', args1.coding_way_spec,
                '--coding_way_mfcc', args1.coding_way_mfcc,
                '--coding_way_wav2vec', args1.coding_way_wav2vec,
                '--input_feature', args1.input_feature,
                '--model_overwrite_mode', args1.model_overwrite_mode,
                '--features_file_path', args1.features_file_path,
                '--ser_task', args1.ser_task,
                '--model_save_prefix', args1.model_save_prefix,
                '--num_classes', str(NUM_CLASSES)
            ]
            
            # 运行训练
            stat = train_ser.main(parse_args_2(sys.argv[1:]))
            
            # 保存结果并删除模型文件
            all_stat.append(stat)
            
    return all_stat

def display_stat(all_stat: list[train_ser.TrainingStats]):
    n_total = REPEAT_TIMES * len(VAL_ID)
    total_best_epoch, total_epoch, total_loss, total_wa, total_ua = 0, 0, 0, 0, 0

    for i in range(n_total):
        # 使用TrainingStats类的属性访问，而不是元组索引
        print(i, ": ", all_stat[i].epoch, all_stat[i].best_epoch, all_stat[i].test_loss, all_stat[i].test_wa, all_stat[i].test_ua) 
        total_best_epoch += all_stat[i].epoch
        total_epoch += all_stat[i].best_epoch
        total_loss += float(all_stat[i].test_loss)
        total_wa += float(all_stat[i].test_wa)
        total_ua += float(all_stat[i].test_ua)

    print("AVERAGE:", total_best_epoch/n_total, total_epoch/n_total, total_loss/n_total, total_wa/n_total, total_ua/n_total)

    print(all_stat)

if __name__ == "__main__":
    # 解析参数
    arg1 = parse_args_1()
    
    # 重复运行训练函数
    all_stat = run(arg1)
    
    # 显示结果
    display_stat(all_stat)
    


