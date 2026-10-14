import os
from collections import defaultdict, OrderedDict
import pandas as pd
import librosa
import audioread
import gc
from natsort import natsorted
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from config import IEMOCAP_DATA_PATH, MELD_DATA_PATH, FEATURE_EXTRACTION_MAX_WORKERS, EMOTION_MAPS

class SER_Database_Base:
    """
    情感语音识别数据库的基类
    
    提供共同的接口和方法，用于处理不同的情感语音数据库
    """
    def __init__(self):
        # 情感标签映射到数字索引
        self.emot_map = {}
        # 数据集的基础路径
        self.base_path = ""
    
    def get_classes(self) -> dict:
        """
        获取情感类别映射字典
        
        将情感标签数字索引映射到对应的情感名称。如果不同的情感设置为相同的数字索引，
        则输出的类别名称中将它们用'+'连接起来。
        
        返回:
            dict: 数字索引到情感名称的映射字典，如 {0: 'anger', 1: 'sadness+fear', ...}
        """
        classes = {}
        for key, value in self.emot_map.items():
            # 如果不同的情感设置为相同的数字，则输出的类别的字符串中将他们用+连接
            if value in classes.keys():
                classes[value] += '+'+key
            else:
                classes[value] = key
        
        return classes
    
    def get_files(self) -> dict:
        """
        获取文件路径和标签的抽象方法，子类需要实现此方法
        
        返回格式：
        {
            'speaker_id': [(file_path, label), ...]
        }
        """
        raise NotImplementedError("子类必须实现此方法")



class MELD_database(SER_Database_Base):
    def __init__(self): 
        super().__init__()
        self.emot_map = EMOTION_MAPS['MELD']
        
        self.csv_name = {'train': 'train_sent_emo.csv', 'test': 'test_sent_emo.csv', 'val': 'dev_sent_emo.csv'}
        
        self.kind_name = {'train': 'train_splits', 'test': 'output_repeated_splits_test', 'val': 'dev_splits_complete'}
        
        self.kind = ['train', 'test', 'val']
        
        self.virtual_id = {'train': '1M', 'test': '2M', 'val': '2F'}
        
        self.base_path = MELD_DATA_PATH
    
    def get_files(self):
        """
        获取每个virtual id的所有所需的.wav文件路径，并组织成字典：
        keys   -> virtual id
        values -> 对应virtual id的(.wav文件路径, 标签)元组列表，此时的标签为数字
        
        最终的结果如：
        {
            '1M': [
                ('/test1/1.wav', 0), ('/test1/2.wav', 1), ('/test1/3.wav', 2)
            ],
            '2M': [
                ('/test2/4.wav', 3), ('/test2/5.wav', 4), ('/test2/6.wav', 5)
            ],
            '2F': [
                ('/test3/7.wav', 6), ('/test3/8.wav', 7), ('/test3/9.wav', 8)
            ]
        }
        """
        # 存储每个virtual id的文件路径和标签
        all_virtual_id_files = defaultdict()
        
        # 枚举train，test，val
        for kind in self.kind:
            # 获取标签csv的文件路径
            csv_path = os.path.join(self.base_path, self.csv_name[kind])
            # 获取wav文件的路径前缀
            wav_path_prefix = os.path.join(self.base_path, self.kind_name[kind])
            
            # 读取csv文件
            label_csv_df = pd.read_csv(csv_path)
            
            # 处理文件
            def process_file(name: str, label_csv_df: pd.DataFrame, wav_path_prefix: str):
                """
                处理单个音频文件，最终返回改文件的文件路径和数字标签，如 (/test/1.wav, 0)
                
                name: 文件名
                label_csv_df: 标签csv文件
                wav_path_prefix: wav文件的路径前缀
                """
                # 排除异常文件
                if name.startswith(('.', 'final')) or name.endswith(".wav"):
                    return None
                try:
                    # 获取wav文件的路径
                    wav_path = os.path.join(wav_path_prefix, name)
                    # 获取对话id和话语id
                    dia_id = int(name.split('_')[0][3:])
                    utt_id = int(name.split('_')[1][3:-4])
                    
                    # 检查文件是否存在
                    if not os.path.exists(wav_path):
                        print(f"文件不存在: {wav_path}")
                        return None
                    
                    # 检查文件是否能正常打开并获取时长
                    try:
                        with audioread.audio_open(wav_path) as audio_file:
                            # 检查音频时长是否足够
                            duration = audio_file.duration
                            if duration < 0.1:  # 小于100ms的音频可能会导致问题
                                print(f"音频时长过短: {wav_path}，时长: {duration}秒")
                                return None
                    except Exception as e:
                        print(f"无法打开音频文件: {wav_path}, 错误: {e}")
                        return None
                    
                    # 获取标签
                    try:
                        row = label_csv_df[(label_csv_df['Dialogue_ID'] == dia_id) & (label_csv_df['Utterance_ID'] == utt_id)]
                        if row.empty:
                            print(f"找不到对应标签: {name}, 对话ID: {dia_id}, 话语ID: {utt_id}")
                            return None
                        
                        emotion = row['Emotion'].values[0]
                        if emotion not in self.emot_map:
                            print(f"未知情感类别: {emotion}, 文件: {name}")
                            return None
                            
                        label = self.emot_map[emotion]
                    except Exception as e:
                        print(f"处理标签时出错: {name}, 错误: {e}")
                        return None
                    
                    # 返回文件路径和标签（此时的标签为数字）
                    return (wav_path, label)
                except Exception as e:
                    print(f"处理文件时出错: {name}, 错误: {e}")
                    return None
            
            # 多线程处理文件
            def get_files_multithreaded(wav_path_prefix: str, label_csv_df: pd.DataFrame):
                # 存储每个virtual id的文件路径和数字标签
                total = []
                
                # 创建线程池
                futures = []
                with ThreadPoolExecutor(max_workers=FEATURE_EXTRACTION_MAX_WORKERS) as executor:  # 根据需要调整线程数
                    # 枚举每个文件
                    for i, name in enumerate(tqdm(natsorted(os.listdir(wav_path_prefix)), desc=f"Appending {kind} files tasks")):
                        # 提交任务
                        futures.append(executor.submit(process_file, name, label_csv_df, wav_path_prefix))
                        
                        # 定期调用垃圾回收器释放内存
                        if i % 100 == 0:
                            gc.collect()
                    
                    # 获取结果
                    for future in tqdm(as_completed(futures), desc=f"Processing {kind} files"):
                        result = future.result()
                        if result:
                            total.append(result)
                
                # 最终垃圾回收
                gc.collect()
                return total
            
            # 获得每个virtual id的文件路径和标签
            total = get_files_multithreaded(wav_path_prefix, label_csv_df)
            
            # 将每个virtual id的元组列表存储到all_virtual_id_files中
            all_virtual_id_files[self.virtual_id[kind]] = total
        
            print(f"\n**** All (path, label) loaded for {self.virtual_id[kind]} ****\n")
        return all_virtual_id_files


class IEMOCAP_database(SER_Database_Base):
    """
    IEMOCAP database contains data from 10 actors, 5 male and 5 female,
    during their affective dyadic interaction. The database consists of
    5 sessions, containing both improvised and scripted sessions. Each session
    consists of 2 unique speakers: 1 male and 1 female.

    For each session, the utterances are organized into conversation folders
        eg. Ses01F_impro01/                     -> improvised conversation 01 of Session 01
                |-- Ses01F_impro01_F000.wav     -> speaker F, utterance 000
                |-- Ses01F_impro01_M000.wav     -> speaker M, utterance 000
                |-- ...

    This function extract utterance filenames and labels for improvised sessions,
    organized into dictionary of {'speakerID':[(conversation_wavs,lab),(wavs,lab),...,(wavs,lab)]}

        > speakerID eg. 1M: Session 1, Male speaker
    
    Database Reference:
        (2008). IEMOCAP: Interactive emotional dyadic motion capture database. 
        Language Resources and Evaluation.
    
    Authors:
        Busso, Carlos
        Bulut, Murtaza
        Lee, Chi-Chun
        Kazemzadeh, Abe
        Mower, Emily
        Kim, Samuel
        Chang, Jeannette
        Lee, Sungbok
        Narayanan, Shrikanth
    
    Download request link:
        https://sail.usc.edu/iemocap/iemocap_release.htm
    """

    def __init__(self, include_scripted=False): 
        super().__init__()
        IEMOCAP_EMO_CODES = {
            'neu': ['neu', 'neutral'],
            'hap': ['hap', 'happy', 'happiness'],
            'sad': ['sad', 'sadness'],
            'ang': ['ang', 'angry', 'anger'],
            'sur': ['sur', 'surprise', 'surprised'],
            'fea': ['fea', 'fear'],
            'dis': ['dis', 'disgust', 'disgusted'],
            'fru': ['fru', 'frustrated', 'frustration'],
            'exc': ['exc', 'excited', 'excitement'],
            'oth': ['oth', 'other', 'others']
        }
        
        #Path
        self.base_path = IEMOCAP_DATA_PATH

        #Emotion to label mapping for features
        self.emot_map = EMOTION_MAPS['IEMOCAP']

        #IEMOCAP Session name
        self.sessions = ['Session1','Session2','Session3','Session4','Session5']
        # self.sessions = ['Session3','Session4','Session5']

        #IEMOCAP available emotion classes 
        self.all_emo_classes = IEMOCAP_EMO_CODES.keys()

        #to include scripted session
        self.include_scripted = include_scripted

    def get_speaker_id(self, session, gender):

        return session[-1]+gender
    
    def get_files(self):
        """
        Get all the required .wav file paths for each speaker and organized into
            dictionary:
                keys   -> speaker ID
                values -> list of (.wav filepath, label) tuples for corresponding speaker
        """
        emotions = self.emot_map.keys()
        dataset_dir = self.base_path
        all_speaker_files = defaultdict()
        total_num_files = 0
        for session_name in os.listdir(dataset_dir):
           
            if session_name not in self.sessions:
                continue
            wav_dir = os.path.join(dataset_dir, session_name, "sentences/wav")
            lab_dir = os.path.join(dataset_dir, session_name, "dialog/EmoEvaluation")

            M_wav, F_wav = list(), list()
            for conversation_folder in os.listdir(wav_dir):
                #omit hidden folders
                if conversation_folder.startswith('.'):
                    continue

                if self.include_scripted == False:
                    # Only use improvised data, for example ".../wav/Ses01F_impro01"
                    if conversation_folder[7:12] != "impro":
                        continue
                
                # Path to the directory containing all the *.wav files of the
                # current conversation
                conversation_dir = os.path.join(wav_dir, conversation_folder)
                
                # Get labels of all utterance in the current conversation
                label_path = os.path.join(lab_dir, conversation_folder + ".txt")
                labels = dict() 
                with open(label_path, "r") as fin:
                    #print(label_path)
                    for line in fin:
                        # If this line is sth like
                        # [6.2901 - 8.2357]	Ses01F_impro01_F000	neu	[2.5000, 2.5000, 2.5000]
                        if line[0] == "[":
                            t = line.split()
                            # For e.g., {"Ses01F_impro01_F000": "neu", ...}
                            labels[t[3]] = t[4]
                
                # Get a list of paths to all *.wav files
                wav_files = []
                for wav_name in os.listdir(conversation_dir):
                    #omit hidden folders
                    if wav_name.startswith('.'):
                        continue
                    #omit non .wav files
                    name, ext = os.path.splitext(wav_name)
                    if ext != ".wav":
                        continue
                    #emotion label
                    emotion = labels[name]
                    if emotion not in emotions:
                        continue
                    label = self.emot_map[emotion]
                    
                    wav_files.append((os.path.join(conversation_dir, wav_name), label))
                
                #separate into individual speakers
                F_wav.extend([emowav for emowav in wav_files if emowav[0][-8] == "F"])
                #F_lab.append(label_path)
                M_wav.extend([emowav for emowav in wav_files if emowav[0][-8] == "M"])
                #M_lab.append(label_path)
                
            #Put speaker utterance and label paths into dictionary
            all_speaker_files[self.get_speaker_id(session_name,'M')] = M_wav
            all_speaker_files[self.get_speaker_id(session_name,'F')] = F_wav

            total_num_files += len(M_wav) + len(F_wav)
        print(f'\nNUMBER OF FILES: {total_num_files}\n')
        return all_speaker_files



SER_DATABASES = {
    'IEMOCAP': IEMOCAP_database, 
    'MELD': MELD_database
}


    