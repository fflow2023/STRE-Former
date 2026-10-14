import sys
import numpy as np
import pickle
from features_util import extract_features
from collections import Counter
import pandas as pd
from database import SER_DATABASES, SER_Database_Base
from tools import parse_arguments, seed_everything, ignore_warnings, Args, get_output_filepath, get_feature_extraction_params, display_statistics

def main(args: Args):
    # 获取输出文件路径
    output_filepath, _ = get_output_filepath(args, check_exist=False)
    # 获取特征提取参数
    feature_extraction_params = get_feature_extraction_params(args, output_filepath)
    # 获取数据集实现的类
    database: SER_Database_Base = SER_DATABASES[args.dataset]()
    # 获取文件路径和标签字典
    id_path_label = database.get_files()

    # 提取特征
    features_data = extract_features(id_path_label, args.features, feature_extraction_params)
    
    # 保存特征数据
    with open(output_filepath, "wb") as fout:
        pickle.dump(features_data, fout)
    
    # 显示统计信息
    display_statistics(database, features_data, args)
    


if __name__ == '__main__':
    # 忽略警告
    ignore_warnings()
    # 设置随机种子
    seed_everything(111)
    # 解析命令行参数
    args = parse_arguments(sys.argv[1:])
    
    
    # 执行主函数
    main(args)
    
    




