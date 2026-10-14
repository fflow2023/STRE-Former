import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import math
import os
from collections import defaultdict
from tqdm import tqdm
# from pysndfx import  AudioEffectsChain
import random
from transformers import BertTokenizer, BertModel, Wav2Vec2ForCTC, Wav2Vec2CTCTokenizer, Wav2Vec2Processor, AutoTokenizer
import gc
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, Future
from config import WAV2VEC2_MODEL_PATH, FEATURE_EXTRACTION_MAX_WORKERS


def extract_features(id_path_label: dict, features_kind: str, params: dict):
    speaker_features = defaultdict()
    for speaker_id in tqdm(id_path_label.keys()):
        data_spec, labels_tot, labels_segs_tot, segs, data_mfcc, data_audio = list(), list(), list(), list(), list(), list()
        
        def remove_silence(x, top_db=20):
            """
            移除音频中的静默部分
            """
            intervals = librosa.effects.split(x, top_db=top_db)
            non_silent_parts = [x[start:end] for start, end in intervals]
            return np.concatenate(non_silent_parts, axis=0)

        def process_file(wav_path, label, params):
            # 读取音频数据
            wave_data, sample_rate = librosa.load(wav_path, sr=16000)
            
            # 移除静默部分
            wave_data = remove_silence(wave_data)
            
            # 使用预加重滤波器，让高频分量更突出，初始状态zi默认设置为0
            wave_data = librosa.effects.preemphasis(wave_data, zi=[0.0])

            # 确保所有音频具有相同长度
            # 如果长度不足，进行填充；如果长度过长，进行截断
            target_length = 48000  # 选择一个合适的固定长度
            if len(wave_data) < target_length:
                # 填充
                padding = np.zeros(target_length - len(wave_data))
                wave_data = np.concatenate([wave_data, padding])
            elif len(wave_data) > target_length:
                # 截断
                wave_data = wave_data[:target_length]

            # 提取spec特征 (channel, frequency, time_sequence_length)
            spec_features_data = GET_FEATURES[features_kind](wave_data, sample_rate, params)
            
            # hop_length需要设置的比seq_len小
            hop_length = 160 
            
            # 提取mfcc特征 (seq_len, 20)
            mfcc = librosa.feature.mfcc(
                y=wave_data, 
                sr=sample_rate, 
                n_mfcc=40, 
                hop_length=hop_length, 
                htk=True
            ).T
            
            # 将单个视频根据segment_size分割成多个sample
            features_segmented = segment_nd_features(wave_data, mfcc, spec_features_data, label, params['segment_size'])

            return features_segmented

        with ThreadPoolExecutor(max_workers=FEATURE_EXTRACTION_MAX_WORKERS) as executor:
            futures: list[Future] = []
            for wav_path, label in tqdm(id_path_label[speaker_id], desc=f"Appending {speaker_id} files tasks"):
                futures.append(executor.submit(process_file, wav_path, label, params))
            
            for future in tqdm(futures, desc=f"Collecting {speaker_id} results"):
                try:
                    features_segmented = future.result()
                    data_spec.append(features_segmented[1])
                    data_mfcc.append(features_segmented[4]) 
                    data_audio.append(features_segmented[5])
                    labels_segs_tot.extend(features_segmented[2])
                    labels_tot.append(features_segmented[3])
                    segs.append(features_segmented[0])
                    gc.collect()
                except Exception as e:
                    print(f"处理{speaker_id}的特征时发生错误: {str(e)}")
                    continue

        # 叠加所有特征
        data_spec = np.vstack(data_spec).astype(np.float32)
        data_mfcc = np.vstack(data_mfcc).astype(np.float32)
        data_audio = np.vstack(data_audio).astype(np.float32)
        labels_tot = np.asarray(labels_tot, dtype=np.int8)
        labels_segs_tot = np.asarray(labels_segs_tot, dtype=np.int8)
        segs = np.asarray(segs, dtype=np.int8)
        
        # 确保所有特征都提取正确
        assert len(labels_tot) == len(segs) #+ == data_mfcc.shape[0]
        assert data_spec.shape[0] == labels_segs_tot.shape[0] == sum(segs)

        # 打印各个特征的形状以确保维度正确
        print(f"{'spec形状':>10}", data_spec.shape)         # 所有分段的语谱图特征
        print(f"{'audio形状':>10}", data_audio.shape)          # 所有分段的音频波形
        print(f"{'mfcc形状':>10}", data_mfcc.shape)          # 所有分段的mfcc特征
        print(f"{'标签形状':>10}", labels_tot.shape)          # 每个完整音频的情感标签
        print(f"{'分段标签形状':>10}", labels_segs_tot.shape)     # 每个分段的情感标签
        print(f"{'segs数量形状':>10}", segs.shape)                # 每个音频被分成的段数
        
        audio_features = defaultdict()
        audio_features["seg_spec"] = data_spec
        audio_features["seg_mfcc"] = data_mfcc
        audio_features["seg_audio"] = data_audio
        audio_features["utter_label"] = labels_tot
        audio_features["seg_label"] = labels_segs_tot
        audio_features["seg_num"] = segs
        
        speaker_features[speaker_id] = audio_features

    # 确保speaker_features的数量与id_path_label的数量相同，数量就是ID的种类数
    assert len(speaker_features) == len(id_path_label)

    return speaker_features


def padding(feature, MAX_LEN):
    """
    mode: 
        zero: padding with 0
        normal: padding with normal distribution
    location: front / back
    """
    padding_mode  = 'zeros'
    padding_location = 'back'

    length = feature.shape[0]
    if length >= MAX_LEN:
        return feature[:MAX_LEN, :]
        
    if padding_mode == "zeros":
        pad = np.zeros([MAX_LEN - length, feature.shape[-1]])
    elif padding_mode == "normal":
        mean, std = feature.mean(), feature.std()
        pad = np.random.normal(mean, std, (MAX_LEN-length, feature.shape[1]))

    feature = np.concatenate([pad, feature], axis=0) if(padding_location == "front") else \
              np.concatenate((feature, pad), axis=0)
    return feature

                 
def paddingSequence(sequences):
    if len(sequences) == 0:
        return sequences
    feature_dim = sequences[0].shape[-1]
    lens = [s.shape[0] for s in sequences]
    # confirm length using (mean + std)
    final_length = int(np.mean(lens) + 3 * np.std(lens))
    # padding sequences to final_length
    final_sequence = np.zeros([len(sequences), final_length, feature_dim])
    for i, s in enumerate(sequences):
        final_sequence[i] = padding(s, final_length)

    return final_sequence

       
def extract_logspec(wav_data, sample_rate, params):
    #unpack params
    window        = params['window']
    win_length    = int((params['win_length']/1000) * sample_rate)
    hop_length    = int((params['hop_length']/1000) * sample_rate)
    ndft          = params['ndft']
    nfreq         = params['nfreq']

    # print("ndft:\t",ndft)
    # print("win_length:\t",win_length)
    
    #calculate stft
    spec = np.abs(
        librosa.stft(
            wav_data, 
            n_fft=ndft, 
            hop_length=hop_length,
            win_length=win_length,
            window=window
            )
        )
    
    spec =  librosa.amplitude_to_db(spec, ref=np.max)
    
    #extract the required frequency bins
    spec = spec[:nfreq]
    
    #Shape into (C, F, T), C = 1
    spec = np.expand_dims(spec, 0)

    return spec


def extract_logmelspec(wav_data, sample_rate, params):
    #unpack params
    window        = params['window']
    win_length    = int((params['win_length']/1000) * sample_rate)
    hop_length    = int((params['hop_length']/1000) * sample_rate)
    ndft          = params['ndft']
    n_mels        = params['nmel']
    

    #calculate stft
    melspec = librosa.feature.melspectrogram(
        y=wav_data, 
        sr=sample_rate, 
        n_mels=n_mels,
        n_fft=ndft,
        hop_length=hop_length,
        win_length=win_length,
        window=window
    )
    
    logmelspec =  librosa.power_to_db(melspec, ref=np.max)

    # Expand to (C, F, T), C = 3
    logmelspec =  np.expand_dims(logmelspec, 0)
    
    return logmelspec


def extract_logdeltaspec(wav_data, sample_rate, params):
      
    #unpack params
    window        = params['window']
    win_length    = int((params['win_length']/1000) * sample_rate)
    hop_length    = int((params['hop_length']/1000) * sample_rate)
    ndft          = params['ndft']
    n_freq        = params['nfreq']
    
    #calculate stft
    logspec = extract_logspec(wav_data, sample_rate, params) # (C, F, T)

    logdeltaspec = librosa.feature.delta(logspec.squeeze(0))
    logdelta2spec = librosa.feature.delta(logspec.squeeze(0), order=2)
    
    #Arrange into (C, F, T), C = 3
    logdeltaspec = np.expand_dims(logdeltaspec, axis=0)
    logdelta2spec = np.expand_dims(logdelta2spec, axis=0)
    logspec = np.concatenate((logspec, logdeltaspec, logdelta2spec), axis=0)
    
    return logspec


def segment_nd_features(wave_data: np.ndarray, mfcc: np.ndarray, spec_features_data: np.ndarray, label: str, segment_size: int):
    '''
    将特征分割成 segment_size 帧。
    如果数据帧小于 segment_size ，则用0填充。

    Input:
    ------
        - wave_data: shape is (Time)
        - mfcc: shape is (Sequence_length, 20)
        - spec_features_data: shape is (Channels, Frequency index, Time)
        - label: emotion label for the current utterance data
        - segment_size: length of each segment
    
    Return:
    -------
    Tuples of (number of segments, spec_total, segment labels, utterance label, mfcc_total, audio_total)
        - spec_total: ndarray of shape (N, C, F, T)
            - N: number of segments
            - C: number of channels
            - F: frequency index
            - T: time index
        - segment labels: list of labels for each segments
            - len(segment labels) == number of segments
    '''
    segment_size_wav = segment_size * 160
    
    # 将特征转换为 (channel, time, frequency)
    # 从 (C, F, T) 转换为 (C, T, F)
    spec_features_data = spec_features_data.transpose(0,2,1)
    
    # 获取语音的时长，单位为毫秒
    time = spec_features_data.shape[1]
    
    # 获取语音的时长，单位为HZ
    time_wav = wave_data.shape[0]
    
    start, end = 0, segment_size
    start_wav, end_wav = 0, segment_size_wav
    
    # 计算每个语音被分割的段数
    num_segs = math.ceil(time / segment_size)
    
    mfcc_tot = []
    audio_tot = []
    spec_tot = []
    
    processor = Wav2Vec2Processor.from_pretrained(WAV2VEC2_MODEL_PATH)
    
    for _ in range(num_segs):
        # 如果最后的部分超过了最长的部分，则截断
        if end > time:
            end, end_wav = time, time_wav
            start, start_wav = max(0, end - segment_size), max(0, end_wav - segment_size_wav)
        
        # 设置填充，如果当前段的长度小于segment_size，则用0填充
        mfcc_pad = np.pad(
            mfcc[start:end], 
            (
                (0, segment_size - (end - start)), 
                (0, 0)
            ), 
            mode="constant"
        )
        audio_pad = np.pad(
            wave_data[start_wav:end_wav], 
            (
                (segment_size_wav - (end_wav - start_wav)), 
                (0)
            ), 
            mode="constant"
        )
        spec_pad = np.pad(
            spec_features_data[:, start:end], 
            (
                (0, 0),  # 通道维度不填充
                (0, segment_size - (end - start)), 
                (0, 0)
            ), 
            mode="constant"
        )
        
        # 叠加mfcc和spec特征
        mfcc_tot.append(mfcc_pad)
        spec_tot.append(spec_pad)
        
        # 叠加音频特征
        audio_pad_np = np.array(audio_pad)
        audio_pad_pt = processor(audio_pad_np, sampling_rate=16000, return_tensors="pt").input_values
        audio_pad_pt = audio_pad_pt.view(-1)
        audio_pad_pt_np = audio_pad_pt.cpu().detach().numpy()
        audio_tot.append(audio_pad_pt_np)
        
        # 更新segment的开始和结束
        start, start_wav = end, end_wav
        end, end_wav = min(time, end + segment_size), min(time_wav, end_wav + segment_size_wav)      
    
    mfcc_tot = np.stack(mfcc_tot)
    spec_tot = np.stack(spec_tot)
    audio_tot = np.stack(audio_tot)
    
    # 设置标签
    utt_label = label
    segment_labels = [label] * num_segs
    
    # 将结果转置输出为 N,C,F,T
    spec_tot = spec_tot.transpose(0,1,3,2)
    
    return (num_segs, spec_tot, segment_labels, utt_label, mfcc_tot, audio_tot)

# 特征提取函数映射
GET_FEATURES = {
    'logspec': extract_logspec,
    'logmelspec': extract_logmelspec,
    'logdeltaspec': extract_logdeltaspec
}


if __name__ == '__main__':
    #test
    sig,sr = librosa.load('noise_wav/presto.wav', sr=None)

    params = {
        'window': 'hamming',
        'win_length': 40,
        'hop_length': 10,
        'ndft':800,
        'nfreq':200
    }
    logdeltaspec = extract_logdeltaspec(sig[5000:37000], sr, params)
    data = segment_nd_features(logdeltaspec, None, 300)
    segment = data[1]
    segment = np.squeeze(segment)
    segment = np.squeeze(segment)
    print(segment.shape)
    plt.figure()
    plt.subplot(3,1,1)
    librosa.display.specshow(segment[0])
    plt.subplot(3,1,2)
    librosa.display.specshow(segment[1])
    plt.subplot(3,1,3)
    librosa.display.specshow(segment[2])
    plt.show()

