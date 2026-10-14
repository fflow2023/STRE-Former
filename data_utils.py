import os
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import get_cmap
from concurrent.futures import ThreadPoolExecutor, as_completed
import gc
from tqdm import tqdm
import pickle
import numpy as np
import torch
from torchvision import transforms
from sklearn.metrics import confusion_matrix
import pandas as pd
from imblearn.over_sampling import RandomOverSampler
from collections import Counter
from sklearn import preprocessing
import random
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
from collections import defaultdict
from io import BytesIO
from rich.table import Table
from rich.panel import Panel
from tools import Logger


SCALER_TYPE = {
    'standard': 'preprocessing.StandardScaler()',
    'minmax'  : 'preprocessing.MinMaxScaler(feature_range=(0,1))'
}


class TrainDataset(torch.utils.data.Dataset):
    """
    data : ndarray
        Input data of shape `N x C x H x W`, where `N` is the number of examples
        (segments), C is number of input channels (3 in the case of image), `H` is image height,
        `W` is image width
    target : ndarray
        Labels for segments (note that one utterance might contain more than
        one segments) of shape `(N,)`.
    num_classes :
        Number of classes.    
    """
    def __init__(self, data, num_classes=7):
        super(TrainDataset).__init__()
        self.data_spec = data['seg_spec']
        self.data_mfcc = data['seg_mfcc']
        self.data_audio = data['seg_audio']
        self.seg_label = data['seg_label']
        # self.target = target
        self.n_samples = len(self.seg_label)
        self.num_classes = num_classes

    def __len__(self):
        return self.n_samples

    def __getitem__(self, index):
        sample = {
            'seg_spec': self.data_spec[index], 
            'seg_mfcc': self.data_mfcc[index],
            'seg_audio': self.data_audio[index],
            'seg_label': self.seg_label[index]
        }
        return sample
        
    def get_preds(self, preds):
        """
        Get predictions for all utterances from their segments' prediction.
        This function will accumulate the predictions for each utterance by
        taking the maximum probability along the dimension 0 of all segments
        belonging to that particular utterance.
        """     
        preds = np.argmax(preds, axis=1)
        return preds 

    def weighted_accuracy(self, predictions, labels):
        """
        Calculate accuracy score given the predictions.

        Parameters
        ----------
        predictions : ndarray
            Model's predictions.

        Returns
        -------
        float
            Accuracy score.

        """
        acc = (labels == predictions).sum() / len(labels)
        return acc

    def unweighted_accuracy(self, predictions, labels):
        """
        Calculate unweighted accuracy score given the predictions.

        Parameters
        ----------
        utt_preds : ndarray
            Processed predictions.

        Returns
        -------
        float
            Unweighted Accuracy (UA) score.

        """


        class_acc = 0
        n_classes = 0
        for c in range(self.num_classes):
            class_pred = np.multiply(( labels == predictions),
                                     ( labels == c)).sum()
            
            if (labels == c).sum() > 0:
                 class_pred /= ( labels == c).sum()
                 n_classes += 1

                 class_acc += class_pred
            
        return class_acc / n_classes


class TestDataset(torch.utils.data.Dataset):
    """
    Holds data for a validation/test set.

    Parameters
    ----------
    data : ndarray
        Input data of shape `N x C x H x W`, where `N` is the number of examples
        (segments), C is number of input channels (3 in the case of image), `H` is image height, 
        `W` is image width
    actual_target : ndarray
        Actual target labels (labels for utterances) of shape `(U,)`, where
        `U` is the number of utterances.
    seg_target : ndarray
        Labels for segments (note that one utterance might contain more than
        one segments) of shape `(N,)`.
    num_segs : ndarray
        Array of shape `(U,)` indicating how many segments each utterance
        contains.
    num_classes :
        Number of classes.
    """
        
    def __init__(self, data, num_classes=7):
        super(TestDataset).__init__()
        # self.data = data
        self.data_spec = data['seg_spec']
        self.data_mfcc = data['seg_mfcc']
        self.data_audio = data['seg_audio']
        # self.utter_label = data['utter_label']
        # self.seg_label = data['seg_label']
        # self.seg_num = data['seg_num']
        
        self.target = data['seg_label']
        self.n_samples = len(self.target)
        self.actual_target = data['utter_label']
        self.n_actual_samples = len(self.actual_target)
        self.num_segs = data['seg_num']
        self.num_classes = num_classes


    def __len__(self):
        return self.n_samples

    def __getitem__(self, index):
        sample = {
            'seg_spec': self.data_spec[index], 
            'seg_mfcc': self.data_mfcc[index],
            'seg_audio': self.data_audio[index],
            'seg_label': self.target[index]#,
            #'utter_label': self.actual_target[index],
            #'seg_num': self.num_segs[index]
            } 
        return sample
        # return self.data[index], self.target[index]

    def get_preds(self, seg_preds):
        """
        Get predictions for all utterances from their segments' prediction.
        This function will accumulate the predictions for each utterance by
        taking the maximum probability along the dimension 0 of all segments
        belonging to that particular utterance.
        """
        preds = np.empty(
            shape=(self.n_actual_samples, self.num_classes), dtype="float")

        end = 0
        
        for v in range(self.n_actual_samples):
            start = end
            end = start + self.num_segs[v]
            
            '''
            # remove the last one for long utterances
            if self.num_segs[v] > 1:
                end = end - 1
                
            preds[v] = np.average(seg_preds[start:end], axis=0)
            
            if self.num_segs[v] > 1:
                end = end + 1
            
            
            # choose the most certain one
            tmp_seg = -1
            for seg in range(self.num_segs[v]):
                end_seg = start + seg
                if np.max(seg_preds[end_seg]) - np.min(seg_preds[end_seg]) > tmp_seg:
                    tmp_seg = np.max(seg_preds[end_seg]) - np.min(seg_preds[end_seg])
                    preds[v] = seg_preds[end_seg]
            '''  
            preds[v] = np.average(seg_preds[start:end], axis=0)
                                 
        preds = np.argmax(preds, axis=1)
        return preds


    def weighted_accuracy(self, utt_preds):
        """
        Calculate accuracy score given the predictions.

        Parameters
        ----------
        utt_preds : ndarray
            Processed predictions.

        Returns
        -------
        float
            Accuracy score.

        """

        acc = (self.actual_target == utt_preds).sum() / self.n_actual_samples
        return acc


    def unweighted_accuracy(self, utt_preds):
        """
        Calculate unweighted accuracy score given the predictions.

        Parameters
        ----------
        utt_preds : ndarray
            Processed predictions.

        Returns
        -------
        float
            Unweighted Accuracy (UA) score.

        """
        class_acc = 0
        n_classes = 0
        
        for c in range(self.num_classes):
            class_pred = np.multiply((self.actual_target == utt_preds),
                                     (self.actual_target == c)).sum()

        
            if (self.actual_target == c).sum() > 0:    
                class_pred /= (self.actual_target == c).sum()
                n_classes += 1
                class_acc += class_pred
        
        return class_acc / n_classes

    
    def confusion_matrix_iemocap(self, utt_preds):
        """Compute confusion matrix given the predictions.

        Parameters
        ----------
        utt_preds : ndarray
            Processed predictions.

        """
        conf = confusion_matrix(self.actual_target, utt_preds)
        
        # Make confusion matrix into data frame for readability
        # IEMOCAP情感标签顺序: {'ang': 0, 'sad': 1, 'hap': 2, 'neu': 3, 'exc': 4, 'fru': 5}
        # conf_fmt = pd.DataFrame({"ang": conf[:, 0], "sad": conf[:, 1],
        #                      "hap": conf[:, 2], "neu": conf[:, 3],
        #                      "exc": conf[:, 4], "fru": conf[:, 5]})
        conf_fmt = pd.DataFrame({"ang": conf[:, 0], "sad": conf[:, 1],
                             "h+e": conf[:, 2], "neu": conf[:, 3]})
        # conf_fmt = conf_fmt.to_string(index=False)
        
        # 只返回混淆矩阵numpy数组，不再返回元组
        return conf
        
    def confusion_matrix_meld(self, utt_preds):
        """Compute confusion matrix for MELD dataset given the predictions.

        Parameters
        ----------
        utt_preds : ndarray
            Processed predictions.

        """
        conf = confusion_matrix(self.actual_target, utt_preds)
        
        # Make confusion matrix into data frame for readability
        # MELD情感标签顺序: {'anger': 0, 'sadness':1, 'joy':2, 'neutral':3, 'surprise':4, 'fear':5, 'disgust':6}
        conf_fmt = pd.DataFrame({"anger": conf[:, 0], "sadness": conf[:, 1],
                             "joy": conf[:, 2], "neutral": conf[:, 3],
                             "surprise": conf[:, 4], "fear": conf[:, 5],
                             "disgust": conf[:, 6]})
        conf_fmt = conf_fmt.to_string(index=False)
        
        # 只返回混淆矩阵numpy数组，不再返回元组
        return conf


class SERDataset:
    """
    Wrapper for both `TrainDataset` and `TestDataset`, which loads and pre-process
    speech spectorgrams into `Dataset` objects.
    
    This also assign the dataset into train, validation, test dataset based on IEMOCAP cross-validation
    arrangement. There are 10 speakers in total (5 sessions x  2 speakers per session) and the IDs assigned
    are 1F, 1M, 2F, 2M, 3F, 3M, 4F, 4M, 5F, 5M.

    Parameters
    ----------
    features_data
        Spectrograms extracted using `extract_features.py`, labels
    num_classes
        Number of emotion classes
    val_speaker_id
        ID of speaker to be used as validation in kfold cross-validation
    test_speaker_id
        ID of speaker to be used as test in kfold cross-validation 
    oversample : bool
        Set 'True' to apply random dataset oversampling to balance the classes
        
    """
    def __init__(self, features_data, num_classes = 7, val_speaker_id='1M', test_speaker_id='1F', oversample=False):
        """
        features_data format: dictionary
            {speaker_id: (data_tot, labels_tot, labels_segs_tot, segs)}

                [0] data_tot: all spectrogram segments, shape = (N_segment, Channels, Freq., Time)
                [1] labels_tot: label for each utterance
                [2] labels_seg_tot: labels for each segments (each utterance might be split into multiple segments)
                [3] segs: number of segments for each utterance
        """

        # get training spectrograms
        train_spec_data, train_mfcc_data, train_audio_data, train_seg_labels, train_labels = None, None, None, None, None
        for speaker_id in features_data.keys():
            if speaker_id in [val_speaker_id, test_speaker_id]:
                continue
            # Concatenate spectrograms from speakers in training set
            if train_mfcc_data is None:
                train_spec_data = features_data[speaker_id]['seg_spec'].astype(np.float32)
                train_mfcc_data = features_data[speaker_id]['seg_mfcc'].astype(np.float32)
                train_audio_data = features_data[speaker_id]['seg_audio'].astype(np.float32)
            else:
                train_spec_data = np.concatenate((train_spec_data, features_data[speaker_id]['seg_spec'].astype(np.float32) ), axis=0)
                train_mfcc_data = np.concatenate((train_mfcc_data, features_data[speaker_id]['seg_mfcc'].astype(np.float32) ), axis=0)
                train_audio_data = np.concatenate((train_audio_data, features_data[speaker_id]['seg_audio'].astype(np.float32) ), axis=0)

            #Concatenate the corresponding labels
            if train_seg_labels is None:
                train_seg_labels = features_data[speaker_id]['seg_label'].astype(np.int64)
                train_labels = features_data[speaker_id]['utter_label'].astype(np.int64)
            else:
                train_seg_labels = np.concatenate((train_seg_labels, features_data[speaker_id]['seg_label'].astype(np.int64)), axis=0)
                train_labels = np.concatenate((train_labels, features_data[speaker_id]['utter_label'].astype(np.int64)), axis=0)
                                             
        self.train_spec_data = train_spec_data
        self.train_mfcc_data = train_mfcc_data
        self.train_audio_data = train_audio_data
        self.train_seg_labels = train_seg_labels
        self.train_labels    = train_labels
        self.num_classes     = num_classes
        #### >>> 合并val和test数据集
        
        # 原始代码
        #get validation spectrograms
        self.val_spec_data  = features_data[val_speaker_id]['seg_spec'].astype(np.float32)
        self.val_mfcc_data  = features_data[val_speaker_id]['seg_mfcc'].astype(np.float32)
        self.val_audio_data  = features_data[val_speaker_id]['seg_audio'].astype(np.float32)
        self.val_seg_labels = features_data[val_speaker_id]['seg_label'].astype(np.int64)
        self.val_labels     = features_data[val_speaker_id]['utter_label'].astype(np.int64)
        self.val_num_segs   = features_data[val_speaker_id]['seg_num']

        #get test spectrograms
        self.test_spec_data  = features_data[test_speaker_id]['seg_spec'].astype(np.float32)
        self.test_mfcc_data  = features_data[test_speaker_id]['seg_mfcc'].astype(np.float32)
        self.test_audio_data  = features_data[test_speaker_id]['seg_audio'].astype(np.float32)
        self.test_seg_labels = features_data[test_speaker_id]['seg_label'].astype(np.int64)
        self.test_labels     = features_data[test_speaker_id]['utter_label'].astype(np.int64)
        self.test_num_segs   = features_data[test_speaker_id]['seg_num']
        
        # 新增合并代码
        # self.val_spec_data = np.concatenate((self.val_spec_data, self.test_spec_data), axis=0)
        # self.val_mfcc_data = np.concatenate((self.val_mfcc_data, self.test_mfcc_data), axis=0)
        # self.val_audio_data = np.concatenate((self.val_audio_data, self.test_audio_data), axis=0)
        # self.val_seg_labels = np.concatenate((self.val_seg_labels, self.test_seg_labels), axis=0)
        # self.val_labels = np.concatenate((self.val_labels, self.test_labels), axis=0)
        # self.val_num_segs = np.concatenate((self.val_num_segs, self.test_num_segs), axis=0)
        
        # self.test_spec_data  = self.val_spec_data
        # self.test_mfcc_data  = self.val_mfcc_data
        # self.test_audio_data  = self.val_audio_data
        # self.test_seg_labels = self.val_seg_labels
        # self.test_labels     = self.val_labels
        # self.test_num_segs   = self.val_num_segs
        ### <<< 合并val和test数据集
        '''
        # used when training with leave-one-session-out validation strategy
        self.val_spec_data = np.concatenate((self.val_spec_data, self.test_spec_data), axis=0)
        self.val_mfcc_data = np.concatenate((self.val_mfcc_data, self.test_mfcc_data), axis=0)
        self.val_audio_data = np.concatenate((self.val_audio_data, self.test_audio_data), axis=0)
        self.val_seg_labels = np.concatenate((self.val_seg_labels, self.test_seg_labels), axis=0)
        self.val_labels = np.concatenate((self.val_labels, self.test_labels), axis=0)
        self.val_num_segs = np.concatenate((self.val_num_segs, self.test_num_segs), axis=0)
        
        self.test_spec_data  = self.val_spec_data
        self.test_mfcc_data  = self.val_mfcc_data
        self.test_audio_data  = self.val_audio_data
        self.test_seg_labels = self.val_seg_labels
        self.test_labels     = self.val_labels
        self.test_num_segs   = self.val_num_segs
        '''

        #Normalize dataset to the range of [0, 1] suitable as image pixel
        self._normalize('minmax')

        #Random oversampling on training dataset
        if oversample == True:
            print('\nPerform training dataset oversampling')
            datar, labelr = random_oversample(self.train_spec_data, self.train_labels)
            datar, labelr = random_oversample(datar,labelr)
            self.train_spec_data = datar
            self.train_labels = labelr

        
        train_spec_data_shape = self.train_spec_data.shape
        val_spec_data_shape = self.val_spec_data.shape
        test_spec_data_shape = self.test_spec_data.shape

        #convert normalized spectrogram to 3 channel image, apply AlexNet image pre-processing
        # NOTE
        self.train_spec_data = self._spec_to_gray(self.train_spec_data)
        self.val_spec_data = self._spec_to_gray(self.val_spec_data)
        self.test_spec_data = self._spec_to_gray(self.test_spec_data)
        self.num_in_ch = 1
        # NOTE
        # self.train_spec_data = self._spec_to_rgb_my(self.train_spec_data)
        # gc.collect()
        # self.val_spec_data = self._spec_to_rgb_my(self.val_spec_data)
        # gc.collect()
        # self.test_spec_data = self._spec_to_rgb_my(self.test_spec_data)
        # gc.collect()
        self.num_in_ch = 3

        #self.train_data = train_spec_data, train_mfcc_data
        self.train_data = defaultdict()
        self.train_data["seg_spec"] = self.train_spec_data
        self.train_data["seg_mfcc"] = self.train_mfcc_data
        self.train_data["seg_audio"] = self.train_audio_data
        self.train_data["seg_label"] = self.train_seg_labels
        # self.train_data["utter_label"] = self.train_labels

        #self.val_data = self.val_spec_data, self.val_mfcc_data
        self.val_data = defaultdict()
        self.val_data["seg_spec"] = self.val_spec_data
        self.val_data["seg_mfcc"] = self.val_mfcc_data       
        self.val_data["seg_audio"] = self.val_audio_data  
        self.val_data["seg_label"] = self.val_seg_labels
        self.val_data["utter_label"] = self.val_labels
        self.val_data["seg_num"] = self.val_num_segs
                
        #self.test_data = self.test_spec_data, self.test_mfcc_data
        self.test_data = defaultdict()
        # self.test_data["seg_spec"] = self.test_spec_data
        # self.test_data["seg_mfcc"] = self.test_mfcc_data
        # self.test_data["seg_audio"] = self.test_audio_data
        # self.test_data["seg_label"] = self.test_seg_labels
        # self.test_data["utter_label"] = self.test_labels
        # self.test_data["seg_num"] = self.test_num_segs

        self.test_data["seg_spec"] = np.concatenate([self.test_spec_data, self.val_spec_data], axis=0)
        self.test_data["seg_mfcc"] = np.concatenate([self.test_mfcc_data, self.val_mfcc_data], axis=0)
        self.test_data["seg_audio"] = np.concatenate([self.test_audio_data, self.val_audio_data], axis=0)
        self.test_data["seg_label"] = np.concatenate([self.test_seg_labels, self.val_seg_labels], axis=0)
        self.test_data["utter_label"] = np.concatenate([self.test_labels, self.val_labels], axis=0)
        self.test_data["seg_num"] = np.concatenate([self.test_num_segs, self.val_num_segs], axis=0)
        
        # print(len(self.train_data['seg_spec']))
        # print(len(self.val_data['seg_spec']))
        # print(len(self.test_data['seg_spec']))
        
        assert len(self.train_spec_data) == train_spec_data_shape[0]
        assert len(self.val_spec_data) == val_spec_data_shape[0]
        assert len(self.test_spec_data) == test_spec_data_shape[0]

        assert val_spec_data_shape[0] == self.val_seg_labels.shape[0] == sum(self.val_num_segs)
        assert self.val_labels.shape[0] == self.val_num_segs.shape[0]
        assert test_spec_data_shape[0] == self.test_seg_labels.shape[0] == sum(self.test_num_segs)
        assert self.test_labels.shape[0] == self.test_num_segs.shape[0]
        
            
        # 创建表格展示数据集信息
        table = Table(title="数据集信息", show_header=True, header_style="bold magenta")
        table.add_column("数据类型", style="cyan")
        table.add_column("形状/值", style="green")
        
        # 添加验证和测试说话人ID
        table.add_row("val ID", str(val_speaker_id))
        table.add_row("test ID", str(test_speaker_id))
        
        # 添加训练数据信息
        table.add_row("train data shape", str(train_spec_data_shape))
        table.add_row("train labels", str(self.train_labels.shape))
        table.add_row("train seg labels", str(self.train_seg_labels.shape))
        
        # 添加验证数据信息
        table.add_row("val data shape", str(val_spec_data_shape))
        table.add_row("val labels shape", str(self.val_labels.shape))
        table.add_row("val seg labels shape", str(self.val_seg_labels.shape))
        table.add_row("val num seg shape", str(self.val_num_segs.shape))
        
        # 添加测试数据信息
        table.add_row("test data shape", str(test_spec_data_shape))
        table.add_row("test labels shape", str(self.test_labels.shape))
        table.add_row("test seg labels shape", str(self.test_seg_labels.shape))
        table.add_row("test num seg shape", str(self.test_num_segs.shape))
        
        # 使用Logger输出表格
        Logger.log_panel(Panel(table, border_style="blue", expand=False))

    
    def _normalize(self, scaling):
        
        '''
        calculate normalization factor from training dataset and apply to
           the whole dataset
        '''
        
        #get data range
        input_range = self._get_data_range()

        #re-arrange array from (N, C, F, T) to (C, -1, F)
        nsegs = self.train_spec_data.shape[0]
        nch   = self.train_spec_data.shape[1]
        nfreq = self.train_spec_data.shape[2]
        ntime = self.train_spec_data.shape[3]
        rearrange = lambda x: x.transpose(1,0,3,2).reshape(nch,-1,nfreq)
        self.train_spec_data = rearrange(self.train_spec_data)
        self.val_spec_data   = rearrange(self.val_spec_data)
        self.test_spec_data  = rearrange(self.test_spec_data)
        
        #scaler type
        scaler = eval(SCALER_TYPE[scaling])

        for ch in range(nch):
            #get scaling values from training data
            scale_values = scaler.fit(self.train_spec_data[ch])
            
            #apply to all
            self.train_spec_data[ch] = scaler.transform(self.train_spec_data[ch])
            self.val_spec_data[ch] = scaler.transform(self.val_spec_data[ch])
            self.test_spec_data[ch] = scaler.transform(self.test_spec_data[ch])
        
        #Shape the data back to (N,C,F,T)
        rearrange = lambda x: x.reshape(nch,-1,ntime,nfreq).transpose(1,0,3,2)
        self.train_spec_data = rearrange(self.train_spec_data)
        self.val_spec_data   = rearrange(self.val_spec_data)
        self.test_spec_data  = rearrange(self.test_spec_data)

        # print(f'\nDataset normalized with {scaling} scaler')
        # print(f'\tRange before normalization: {input_range}')
        # print(f'\tRange after  normalization: {self._get_data_range()}')
        
        gc.collect()

    def _get_data_range(self):
        #get data range
        trmin = np.min(self.train_spec_data)
        evmin = np.min(self.val_spec_data)
        tsmin = np.min(self.test_spec_data)
        dmin = np.min(np.array([trmin, evmin, tsmin]))

        trmax = np.max(self.train_spec_data)
        evmax = np.max(self.val_spec_data)
        tsmax = np.max(self.test_spec_data)
        dmax = np.max(np.array([trmax, evmax, tsmax]))
        
        return [dmin, dmax]

    def _spec_to_rgb(self,data):

        """
        Convert normalized spectrogram to pseudo-RGB image based on pyplot color map
            and apply AlexNet image pre-processing
        
        Input: data
                - shape (N,C,H,W) = (num_spectrogram_segments, 1, Freq, Time)
                - data range [0.0, 1.0]
        """

        #AlexNet preprocessing
        alexnet_preprocess = transforms.Compose([
                transforms.Resize(224),
                #transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ]) 

        # Get the color map to convert normalized spectrum to RGB
        cm = plt.get_cmap('jet') #brg #gist_heat #brg #bwr

        #Flip the frequency axis to orientate image upward, remove Channel axis
        data = np.flip(data,axis=2)
        
        data = np.squeeze(data, axis=1) 

        data_tensor = list()

        for i, seg in enumerate(data):
            seg = np.clip(seg, 0.0, 1.0)
            seg_rgb = (cm(seg)[:,:,:3]*255.0).astype(np.uint8)
            
            img = Image.fromarray(seg_rgb, mode='RGB')

            data_tensor.append(alexnet_preprocess(img))
        
        return data_tensor


    def _spec_to_gray(self,data):

        """
        Convert normalized spectrogram to 3-channel gray image (identical data on each channel)
            and apply AlexNet image pre-processing
        
        Input: data
                - shape (N,C,H,W) = (num_spectrogram_segments, 1, Freq, Time)
                - data range [0.0, 1.0]
        """

        #AlexNet preprocessing
        alexnet_preprocess = transforms.Compose([
                transforms.Resize(256),
                #transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ]) 

        #Convert format to uint8, flip the frequency axis to orientate image upward, 
        #   duplicate into 3 channels
        data = np.clip(data,0.0, 1.0)
        data = (data*255.0).astype(np.uint8)
        data = np.flip(data,axis=2)
        data = np.moveaxis(data,1,-1)
        data = np.repeat(data,3,axis=-1)
       
        data_tensor = list()
        for i, seg in enumerate(data):
            with Image.fromarray(seg, mode='RGB') as img:
                data_tensor.append(alexnet_preprocess(img))
            if i % 100 == 0:
                gc.collect()
            
        return data_tensor  
    
    
    
    
    
    
    
    
    def get_train_dataset(self):
        #print(self.train_mfcc_data.shape)
        #print(self.train_seg_labels.shape)
        #print(self.train_labels.shape)
        return TrainDataset(
            self.train_data, num_classes=self.num_classes)
    
    def get_val_dataset(self):
        return TestDataset(
            self.val_data, num_classes=self.num_classes)
    
    def get_test_dataset(self):
        return TestDataset(
            self.test_data, num_classes=self.num_classes)
        

    def save_mfcc_pic(self, save_pic = False):
        if save_pic == False:
            print("x" * 10, " MFCC未保存为图片 ", "x" * 10)
            return
        
        def process_and_transform_mfcc(data):
            def process_single_mfcc(index):
                try:
                    seg_mfcc = data["seg_mfcc"][index]

                    # 归一化处理
                    min_val, max_val = np.min(seg_mfcc), np.max(seg_mfcc)
                    seg_mfcc_normalized = (seg_mfcc - min_val) / (max_val - min_val)

                    # 模拟保存为图片
                    buffer = BytesIO()
                    plt.imsave(buffer, seg_mfcc_normalized, cmap='jet', format='png')
                    buffer.seek(0)

                    # 模拟读取图片
                    img = plt.imread(buffer)
                    img = img[:, :, :3]  # 只保留 RGB 通道
                    buffer.close()

                    # 转为单通道灰度图
                    single_channel_img = np.dot(img, [0.2989, 0.5870, 0.1140])

                    # 更新原始数据（可选）
                    data['seg_mfcc'][index] = np.array(single_channel_img, dtype=np.float32)

                except Exception as e:
                    print(f"Error processing MFCC index {index}: {e}")

            # 使用多线程加速处理
            with ThreadPoolExecutor(max_workers=8) as executor:
                executor.map(process_single_mfcc, range(len(data['seg_mfcc'])))


        print("*" * 10," MFCC图片保存开始 ", "*" * 10)
        
        # 处理训练数据
        process_and_transform_mfcc(self.train_data)
        gc.collect()  # Collect garbage to free up memory

        # 处理验证数据
        process_and_transform_mfcc(self.val_data)
        gc.collect()  # Collect garbage to free up memory

        # 处理测试数据
        process_and_transform_mfcc(self.test_data)
        gc.collect()  # Collect garbage to free up memory
        
        
        print("*" * 10," MFCC图片保存完成 ", "*" * 10)
        print()
    
    
    def _spec_to_rgb_my(self, data, save_images=True):
        if save_images == False:
            print("x" * 10, " spec未保存为图片 ", "x" * 10)    
        else:
            print("*" * 10," spec图片保存开始 ", "*" * 10)
        
        # AlexNet preprocessing
        alexnet_preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]) 

        # Get the color map to convert normalized spectrum to RGB
        cm = plt.get_cmap('jet')

        # Flip the frequency axis to orientate image upward, remove Channel axis
        data = np.flip(data, axis=2)
        data = np.squeeze(data, axis=1)

        def process_segment(index):
            try:
                # Clip and normalize data
                seg = np.clip(data[index], 0.0, 1.0)

                if save_images:
                    # Simulate saving to an image and reading back
                    with BytesIO() as buffer:
                        plt.imsave(buffer, seg, cmap='jet', format='png')
                        buffer.seek(0)
                        img = plt.imread(buffer)
                    img = img[:, :, :3]  # Retain RGB channels
                    buffer.close()
                else:
                    # Directly simulate the RGB transformation without I/O
                    img = (cm(seg)[:, :, :3] * 255.0).astype(np.uint8)  # Convert to RGB values

                # Convert to PIL image and preprocess
                img = Image.fromarray(img, mode='RGB')
                preprocessed_img = alexnet_preprocess(img)

                del seg, img  # Free memory
                return preprocessed_img

            except Exception as e:
                print(f"Error processing spectrogram segment {index}: {e}")
                return None

        # Parallel processing of segments
        with ThreadPoolExecutor(max_workers=8) as executor:
            future_to_index = {executor.submit(process_segment, i): i for i in range(len(data))}
            data_tensor = []
            count = 0
            for future in tqdm(as_completed(future_to_index)):
                index = future_to_index[future]
                count += 1
                try:
                    result = future.result()
                    if result is not None:
                        data_tensor.append(result)
                    del result
                except Exception as exc:
                    print(f"Segment {index} generated an exception: {exc}")
                del index

                if count % 100 == 0:
                    gc.collect()

        # Filter out any None values if any segment processing failed
        data_tensor = [tensor for tensor in data_tensor if tensor is not None]

        gc.collect()  # Collect garbage to free up memory


        if save_images == True:
            print("*" * 10," spec图片保存完成 ", "*" * 10)
            print()
        
        return data_tensor


def random_oversample(data, labels):
    print('\tOversampling method: Random Oversampling')
    ros = RandomOverSampler(random_state=0,sampling_strategy='minority')

    n_samples = data.shape[0]
    fh = data.shape[2]
    fw = data.shape[3]
    n_features= fh*fw
        
    data = np.squeeze(data,axis=1)
    data = np.reshape(data,(n_samples, n_features))
    data_resampled, label_resampled = ros.fit_resample(data, labels)
    n_samples = data_resampled.shape[0]
    data_resampled = np.reshape(data_resampled,(n_samples,fh,fw))
    data_resampled = np.expand_dims(data_resampled, axis=1)
    
    return data_resampled, label_resampled


