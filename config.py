#####################################
#############  IEMOCAP ##############
#####################################

# 训练超参数
REPEAT_TIMES  = 1                 # use different seed for each repeat
NUM_EPOCHS    = 500
EARLY_STOP    = 100
BATCH_SIZE    = 16
LR            = 1e-5  
RANDOM_SEED   = 67
USE_GPU       = True
GPU_ID        = 0
SER_TASK      = "SLM"
# 新增骨干网络配置
BACKBONE = 'transformer'
LAYERS = 1

# 模型保存参数
MODEL_SAVE_PREFIX      = "./saved_models/"
MODEL_SAVE_LABEL = "IEMOCAP_6class_fu_newv2_F-FR-FR_1e-5"

# # 数据集参数：4classes # lr:1e-4
# DATABASE_KIND          = "IEMOCAP"
# VAL_ID  = ['5F']
# TEST_ID = ['5M']
# NUM_CLASSES            = 4
# FEATURES_FILE_PATH     = "./features_extraction/IEMOCAP_features_4class_fu_new.pkl"


# 数据集参数：6classes # lr:1e-5
DATABASE_KIND          = "IEMOCAP"
VAL_ID  = ['5F']
TEST_ID = ['5M']
NUM_CLASSES            = 6
FEATURES_FILE_PATH     = "./features_extraction/IEMOCAP_features_6class_fu_newv2.pkl"

# # 数据集参数：MELD # lr:1e-5
# DATABASE_KIND          = "MELD"
# VAL_ID  = ['2F']
# TEST_ID = ['2M']
# NUM_CLASSES            = 7
# FEATURES_FILE_PATH     = "./features_extraction/MELD_features_fu_new.pkl"

# 消融实验：FTS 开关
USE_FTS = True

# 位置编码方式 APE RoPE Fib Fib+RoPE ALiBi

RATIO                  = 0.8
CODING_WAY_SPEC        = "Fib"
CODING_WAY_MFCC        = "Fib+RoPE"
CODING_WAY_WAV2VEC     = "Fib+RoPE"

INPUT_FEATURE          = "SMW"
MODEL_OVERWRITE_MODE   = False
WAV2VEC2_MODEL_PATH    = "./features_extraction/pretrained_model/wav2vec2-base-960h/"

SPEC_PIC_SIZE          = (256, 384)
SPEC_PATCH_SIZE        = (16, 16)
MFCC_PIC_SIZE          = (300, 40)
MFCC_PATCH_SIZE        = (20, 10)
WAV2VEC_PIC_SIZE       = (149, 768)
WAV2VEC_PATCH_SIZE     = (16, 16)

LSTM_MFCC_HIDDEN_SIZE  = 256

EMOTION_MAPS = {
    "IEMOCAP": {
        'ang': 0, 'sad': 1, 'hap': 2, 'neu': 3, 'exc': 4, 'fru': 5
        # 'ang': 0, 'sad': 1, 'h+e': 2, 'neu': 3
    }, 
    "MELD": {
        'anger': 0, 'sadness':1, 'joy':2, 'neutral':3, 'surprise':4, 'fear':5, 'disgust':6
    }
}


# 可视化配置
COLORS_PER_CLASS = {
    0 : [0, 0, 0],
    1 : [255, 107, 107],
    2 : [100, 100, 255],
    3 : [16, 172, 132],
}

# HIFI-GAN参数
HIFI_GAN_MODEL_PATH = "./models/hifigan/pretrained_models/LJ_FT_T2_V3/generator_v3"
HIFI_GAN_CONFIG_PATH = "./models/hifigan/pretrained_models/LJ_FT_T2_V3/config.json"

### Fib+RoPE参数
# Fib中算出来的矩阵和RoPE中算出来的矩阵的比值
Fib_RoPE_RATIO = 1

# 是否使用上下随机反转
USE_UP_DOWN_FLIP = True
# 是否使用左右随机反转
USE_LEFT_RIGHT_FLIP = True
