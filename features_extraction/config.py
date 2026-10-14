WAV2VEC2_MODEL_PATH = "/home/chason/Project/FibFormer-2025-05-27/features_extraction/pretrained_model/wav2vec2-base-960h"

MELD_DATA_PATH = "/home/chason/database/MELD/MELD.Raw"

IEMOCAP_DATA_PATH = "/home/chason/database/IEMOCAP/IEMOCAP_full_release/IEMOCAP_full_release"

FEATURE_EXTRACTION_MAX_WORKERS = 8

EMOTION_MAPS = {
    "IEMOCAP": {
        # 'ang': 0, 'sad': 1, 'hap': 2, 'neu': 3, 'exc': 4, 'fru': 5
        'ang': 0, 'sad': 1, 'hap': 2, 'neu': 3, 'exc': 2
    }, 
    "MELD": {
        'anger': 0, 'sadness':1, 'joy':2, 'neutral':3, 'surprise':4, 'fear':5, 'disgust':6
    }
}