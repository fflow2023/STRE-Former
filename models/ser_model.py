"""
        spec(200, 300) --> transformer --> alexnet --> 128 ----------------------------
    /                                                   |                              \ 
wav -   mfcc(40, 300)  --> transformer --> lstm -----> 128 ---------------------------------------> 384 --> 256 --> 128 --> 4
    \                                                   | X                            /
        wav2vec(149,768) --> transformer ---------> (256, 256) --> (256) --> 128 ------

"""
"""
AIO -- All Model in One
"""

#models\ser_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings

# 忽略 Hugging Face Transformers 的特定警告
warnings.filterwarnings("ignore", message="Some weights of the model checkpoint")
warnings.filterwarnings("ignore", message="You should probably TRAIN this model")

from transformers import  Wav2Vec2Model
from models.ser_spec import SER_AlexNet
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor
from PIL import Image
from matplotlib.colors import Normalize
from matplotlib.cm import get_cmap
import numpy as np
import os
from config import *
from tools import *
from rich import print

class Ser_Model(nn.Module):
    def __init__(self, args2: Args2=None):
        super(Ser_Model, self).__init__()
        self.args2 = args2
        
         # === 新增骨干网络参数解析 ===
        self.backbone_type = args2.backbone
        self.layers = args2.layers
        # print(f"[DBG] layers   = {self.layers}")
        
        if(self.backbone_type=="transformer"):
            # print(f"[DBG] backbone = {self.backbone_type} (T)")
            from models.transformers_encoder.transformer import TransformerEncoder as TransformerEncoder_traditional
            from models.transformers_encoder_RoPE_2D.transformer import TransformerEncoder as TransformerEncoder_RoPE_2D
            from models.transformers_encoder_RoPE_Fib.transformer import TransformerEncoder as TransformerEncoder_RoPE_Fib
            from models.transformers_encoder_RoPE_Fib_Plus_RoPE.transformer import TransformerEncoder as TransformerEncoder_RoPE_Fib_Plus_RoPE
            from models.transformers_encoder_ALiBi.transformer import TransformerEncoder as TransformerEncoder_ALiBi
        else:            
            # print(f"[DBG] backbone = {self.backbone_type} (ST)")
            from models.transformers_encoder.Swintransformer import TransformerEncoder as TransformerEncoder_traditional
            from models.transformers_encoder_RoPE_2D.Swintransformer import TransformerEncoder as TransformerEncoder_RoPE_2D
            from models.transformers_encoder_RoPE_Fib.Swintransformer import TransformerEncoder as TransformerEncoder_RoPE_Fib
            from models.transformers_encoder_RoPE_Fib_Plus_RoPE.Swintransformer import TransformerEncoder as TransformerEncoder_RoPE_Fib_Plus_RoPE
            # ALiBi 始终使用标准 Transformer（无 Swin 变体）
            from models.transformers_encoder_ALiBi.transformer import TransformerEncoder as TransformerEncoder_ALiBi
        
        self.use_spec    = 'S' in args2.input_feature
        self.use_mfcc    = 'M' in args2.input_feature
        self.use_wav2vec = 'W' in args2.input_feature

        # CNN for Spectrogram
        self.alexnet_model = SER_AlexNet(num_classes=NUM_CLASSES, in_ch=3, pretrained=True)
        
        self.post_spec_dropout = nn.Dropout(p=0.1)
        self.post_spec_layer   = nn.Linear(9216, 128) # 9216 for cnn, 32768 for ltsm s, 65536 for lstm l
        
        # LSTM for MFCC
        if self.args2.coding_way_mfcc in ('APE', 'ALiBi'):
            self.lstm_mfcc         = nn.LSTM(input_size=MFCC_PIC_SIZE[0], hidden_size=MFCC_PIC_SIZE[1], num_layers=2, batch_first=True, dropout=0.5, bidirectional = True)
        elif self.args2.coding_way_mfcc == 'RoPE':
            self.lstm_mfcc         = nn.LSTM(input_size=40, hidden_size=LSTM_MFCC_HIDDEN_SIZE * 2, num_layers=2, batch_first=True, dropout=0.5, bidirectional = True)
        else:
            self.lstm_mfcc         = nn.LSTM(input_size=MFCC_PATCH_SIZE[0] * MFCC_PATCH_SIZE[1], hidden_size=LSTM_MFCC_HIDDEN_SIZE, num_layers=2, batch_first=True, dropout=0.5, bidirectional = True)
        self.post_mfcc_dropout = nn.Dropout(p=0.1)
        
        if self.args2.coding_way_mfcc in ('APE', 'ALiBi'):
            self.post_mfcc_layer   = nn.Linear(3200, 128)
        elif self.args2.coding_way_mfcc == 'RoPE':
            self.post_mfcc_layer   = nn.Linear(122880, 128)
        else:
            self.post_mfcc_layer   = nn.Linear((MFCC_PIC_SIZE[0] // MFCC_PATCH_SIZE[0]) * (MFCC_PIC_SIZE[1] // MFCC_PATCH_SIZE[1]) * 2 * LSTM_MFCC_HIDDEN_SIZE, 128)
        
        # Spectrogram + MFCC
        self.post_spec_mfcc_att_dropout = nn.Dropout(p=0.1)
        if self.args2.coding_way_mfcc in ('APE', 'ALiBi'):
            self.post_spec_mfcc_att_layer   = nn.Linear(256, 768)
        elif self.args2.coding_way_mfcc == 'RoPE':
            self.post_spec_mfcc_att_layer   = nn.Linear(256, 1064)
        else:
            self.post_spec_mfcc_att_layer   = nn.Linear(256, WAV2VEC_PIC_SIZE[0] // WAV2VEC_PATCH_SIZE[0] * WAV2VEC_PIC_SIZE[1] // WAV2VEC_PATCH_SIZE[1])
        
        # WAV2VEC 2.0
        self.wav2vec2_model: Wav2Vec2Model = Wav2Vec2Model.from_pretrained(WAV2VEC2_MODEL_PATH)
        self.post_wav_dropout = nn.Dropout(p=0.1)
        if self.args2.coding_way_wav2vec in ('APE', 'ALiBi'): # 768
            self.post_wav_layer   = nn.Linear(149, 128)
            if self.args2.coding_way_mfcc == 'RoPE':
                self.mfcc_adjust = nn.Linear(1064,768) # Adjust MFCC to match Wav2Vec2 output
            elif self.args2.coding_way_mfcc == 'Fib':
                self.mfcc_adjust = nn.Linear(432,768) # Adjust MFCC to match Wav2Vec2 output
            elif self.args2.coding_way_mfcc == 'Fib+RoPE':
                self.mfcc_adjust = nn.Linear(432,768) # Adjust MFCC to match Wav2Vec2 output
            elif self.args2.coding_way_mfcc in ('APE', 'ALiBi'):
                self.mfcc_adjust = nn.Linear(768,768) # Adjust MFCC to match Wav2Vec2 output


        elif self.args2.coding_way_wav2vec == 'RoPE': # 1064
            self.post_wav_layer   = nn.Linear(768, 128)
            if self.args2.coding_way_mfcc in ('APE', 'ALiBi'):
                self.mfcc_adjust = nn.Linear(768,1064) # Adjust MFCC to match Wav2Vec2 output
            elif self.args2.coding_way_mfcc == 'Fib':
                self.mfcc_adjust = nn.Linear(432,1064) # Adjust MFCC to match Wav2Vec2 output
            elif self.args2.coding_way_mfcc == 'RoPE':
                self.mfcc_adjust = nn.Linear(1064,1064) # Adjust MFCC to match Wav2Vec2 outpu
            elif self.args2.coding_way_mfcc == 'Fib+RoPE':
                self.mfcc_adjust = nn.Linear(432,1064) # Adjust MFCC to match Wav2Vec2 output

        else: # 432
            self.post_wav_layer   = nn.Linear(WAV2VEC_PATCH_SIZE[0] * WAV2VEC_PATCH_SIZE[1], 128)
            if self.args2.coding_way_mfcc in ('APE', 'ALiBi'):
                self.mfcc_adjust = nn.Linear(768,432) # Adjust MFCC to match Wav2Vec2 output
            elif self.args2.coding_way_mfcc == 'RoPE':
                self.mfcc_adjust = nn.Linear(1064,432) # Adjust MFCC to match Wav2Vec2 output
            else:  # F/FR
                self.mfcc_adjust = nn.Linear(432,432) # Adjust MFCC to match Wav2Vec2 output
        
        # Combination
        self.post_att_dropout = nn.Dropout(p=0.1)
        self.post_att_layer_1 = nn.Linear((self.use_spec + self.use_mfcc + self.use_wav2vec) * 128, 128)
        self.post_att_layer_2 = nn.Linear(128, 128)
        self.post_att_layer_3 = nn.Linear(128, NUM_CLASSES)
        
        # 设置device
        self.device = torch.device(f"cuda:{args2.gpu_id}" if args2.use_gpu else "cpu")
 
        if self.args2.coding_way_spec == 'Fib':
            self.transformer_encoder_spec    = TransformerEncoder_RoPE_Fib(num_heads=1, layers=self.layers, pic_size=SPEC_PIC_SIZE, patch_size=SPEC_PATCH_SIZE, device=self.device, ratio=args2.ratio, use_fts=args2.use_fts)
        elif self.args2.coding_way_spec == 'RoPE':
            self.transformer_encoder_spec    = TransformerEncoder_RoPE_2D(embed_dim=384, num_heads=4, layers=self.layers, in_channels=3, patch_size=SPEC_PATCH_SIZE)
        elif self.args2.coding_way_spec == 'APE':
            self.transformer_encoder_spec    = TransformerEncoder_traditional(embed_dim=384, num_heads=4, layers=self.layers)
        elif self.args2.coding_way_spec == 'Fib+RoPE':
            self.transformer_encoder_spec    = TransformerEncoder_RoPE_Fib_Plus_RoPE(num_heads=1, layers=self.layers, pic_size=SPEC_PIC_SIZE, patch_size=SPEC_PATCH_SIZE, device=self.device, ratio=args2.ratio, use_fts=args2.use_fts)
        elif self.args2.coding_way_spec == 'ALiBi':
            self.transformer_encoder_spec    = TransformerEncoder_ALiBi(embed_dim=384, num_heads=4, layers=self.layers)
        else:
            raise ValueError('coding_way must be Fib or RoPE or APE or Fib+RoPE or ALiBi')
        

        if self.args2.coding_way_mfcc == 'Fib':
            self.transformer_encoder_mfcc    = TransformerEncoder_RoPE_Fib(num_heads=1, layers=self.layers, pic_size=MFCC_PIC_SIZE, patch_size=MFCC_PATCH_SIZE, in_channels=1, device=self.device, ratio=args2.ratio, use_fts=args2.use_fts)
        elif self.args2.coding_way_mfcc == 'RoPE':
            self.transformer_encoder_mfcc    = TransformerEncoder_RoPE_2D(embed_dim=40, num_heads=4, layers=self.layers, in_channels=1, patch_size=MFCC_PATCH_SIZE) 
        elif self.args2.coding_way_mfcc == 'APE':
            self.transformer_encoder_mfcc    = TransformerEncoder_traditional(embed_dim=40, num_heads=4, layers=self.layers)
        elif self.args2.coding_way_mfcc == 'Fib+RoPE':
            self.transformer_encoder_mfcc    = TransformerEncoder_RoPE_Fib_Plus_RoPE(num_heads=1, layers=self.layers, pic_size=MFCC_PIC_SIZE, patch_size=MFCC_PATCH_SIZE, in_channels=1, device=self.device, ratio=args2.ratio, use_fts=args2.use_fts)
        elif self.args2.coding_way_mfcc == 'ALiBi':
            self.transformer_encoder_mfcc    = TransformerEncoder_ALiBi(embed_dim=40, num_heads=4, layers=self.layers)
        else:
            raise ValueError('coding_way must be Fib or RoPE or APE or Fib+RoPE or ALiBi')
        
        if self.args2.coding_way_wav2vec == 'Fib':
            self.transformer_encoder_wav2vec = TransformerEncoder_RoPE_Fib(num_heads=1, layers=self.layers, pic_size=WAV2VEC_PIC_SIZE, patch_size=WAV2VEC_PATCH_SIZE, in_channels=1, device=self.device, ratio=args2.ratio, use_fts=args2.use_fts)
        elif self.args2.coding_way_wav2vec == 'RoPE':
            self.transformer_encoder_wav2vec = TransformerEncoder_RoPE_2D(embed_dim=768, num_heads=4, layers=self.layers, in_channels=1, patch_size=WAV2VEC_PATCH_SIZE)
        elif self.args2.coding_way_wav2vec == 'APE':
            self.transformer_encoder_wav2vec = TransformerEncoder_traditional(embed_dim=768, num_heads=4, layers=self.layers)
        elif self.args2.coding_way_wav2vec == 'Fib+RoPE':
            self.transformer_encoder_wav2vec = TransformerEncoder_RoPE_Fib_Plus_RoPE(num_heads=1, layers=self.layers, pic_size=WAV2VEC_PIC_SIZE, patch_size=WAV2VEC_PATCH_SIZE, in_channels=1, device=self.device, ratio=args2.ratio, use_fts=args2.use_fts)
        elif self.args2.coding_way_wav2vec == 'ALiBi':
            self.transformer_encoder_wav2vec = TransformerEncoder_ALiBi(embed_dim=768, num_heads=4, layers=self.layers)
        else:
            raise ValueError('coding_way must be Fib or RoPE or APE or Fib+RoPE or ALiBi')
        

        
        # # 冻结掉wav2vec的所有参数，探究输出元素的梯度影响
        # for param in self.wav2vec2_model.parameters():
        #     param.requires_grad = False
    
    def forward(self, audio_spec: torch.Tensor, audio_mfcc: torch.Tensor, audio_wav: torch.Tensor):
        # audio_spec: [batch, 3, 256, 384]
        # audio_mfcc: [batch, 300, 88]
        # audio_wav: [batch, 48000]
        
        # spec
        audio_spec = self.transformer_encoder_spec(audio_spec) # [batch, 3, 256, 384] --> [batch, 384, 256]
        audio_spec = torch.unsqueeze(audio_spec, 1) # [batch, 1, 384, 256]
        audio_spec = audio_spec.repeat(1, 3, 1, 1)   # [batch, 3, 384, 256]
        
        audio_spec, output_spec_t = self.alexnet_model(audio_spec) # [batch, 256, 6, 6], []
        audio_spec = audio_spec.reshape(audio_spec.shape[0], audio_spec.shape[1], -1) # [batch, 256, 36]  
        audio_spec_ = torch.flatten(audio_spec, 1) # [batch, 9216]  
        audio_spec_d = self.post_spec_dropout(audio_spec_) # [batch, 9216]  
        audio_spec_p = F.relu(self.post_spec_layer(audio_spec_d), inplace=True) # [batch, 128]  
        
        # mfcc
        audio_mfcc = F.normalize(audio_mfcc, p=2, dim=2)
        audio_mfcc = audio_mfcc.unsqueeze(1) # [batch, 1, 300, 40]
        audio_mfcc = self.transformer_encoder_mfcc(audio_mfcc) # [batch, 60, 200]
        audio_mfcc, _ = self.lstm_mfcc(audio_mfcc) # [batch, 60, 512]  
        #+ audio_mfcc = self.att(audio_mfcc)
        audio_mfcc_ = torch.flatten(audio_mfcc, 1) # [batch, 30720]  
        audio_mfcc_att_d = self.post_mfcc_dropout(audio_mfcc_) # [batch, 30720]  
        audio_mfcc_p = F.relu(self.post_mfcc_layer(audio_mfcc_att_d), inplace=True) # [batch, 128]  
        
        # FOR WAV2VEC2.0 WEIGHTS 
        spec_mfcc = torch.cat([audio_spec_p, audio_mfcc_p], dim=-1) # [batch, 256] 
        audio_spec_mfcc_att_d = self.post_spec_mfcc_att_dropout(spec_mfcc)# [batch, 256] 
        audio_spec_mfcc_att_p = F.relu(self.post_spec_mfcc_att_layer(audio_spec_mfcc_att_d), inplace=True)# [batch, 432] 
        audio_spec_mfcc_att_p = audio_spec_mfcc_att_p.reshape(audio_spec_mfcc_att_p.shape[0], 1, -1)# [batch, 1, 432] 
        #+ audio_spec_mfcc_att_2 = F.softmax(audio_spec_mfcc_att_1, dim=2)
        
        # wav2vec 2.0 
        audio_wav = self.wav2vec2_model(audio_wav.to(self.device)).last_hidden_state # [batch, 149, 768] 
        audio_wav = audio_wav.unsqueeze(1) # [batch, 1, 149, 768]
        audio_wav = self.transformer_encoder_wav2vec(audio_wav) # [batch, 432, 256]
        

        # print(audio_spec_mfcc_att_p.shape, audio_wav.shape)
        audio_spec_mfcc_att_p = self.mfcc_adjust(audio_spec_mfcc_att_p)
        audio_wav = torch.matmul(audio_spec_mfcc_att_p, audio_wav) # [batch, 1, 256] 
        audio_wav = audio_wav.reshape(audio_wav.shape[0], -1) # [batch, 256] 
        #audio_wav = torch.mean(audio_wav, dim=1)
        audio_wav_d = self.post_wav_dropout(audio_wav) # [batch, 256] 
        audio_wav_p = F.relu(self.post_wav_layer(audio_wav_d), inplace=True) # [batch, 128] 
        
        # # output
        # audio_att = torch.cat([
        #     audio_spec_p if self.use_spec else None,
        #     audio_mfcc_p if self.use_mfcc else None,
        #     audio_wav_p if self.use_wav2vec else None
        # ], dim=-1)
                # output
        # 使用列表收集所有特征，然后过滤None值
        features = []
        if self.use_spec:
            features.append(audio_spec_p)
        if self.use_mfcc:
            features.append(audio_mfcc_p)
        if self.use_wav2vec:
            features.append(audio_wav_p)
            
        audio_att = torch.cat(features, dim=-1)
        
        
        audio_att_d_1 = self.post_att_dropout(audio_att) # [batch, 384] 
        audio_att_1 = F.relu(self.post_att_layer_1(audio_att_d_1), inplace=True) # [batch, 128] 
        audio_att_d_2 = self.post_att_dropout(audio_att_1) # [batch, 128] 
        audio_att_2 = F.relu(self.post_att_layer_2(audio_att_d_2), inplace=True)  # [batch, 128] 
        output_att = self.post_att_layer_3(audio_att_2) # [batch, 4] 
  
  
  
        output = {
            # 'F1': audio_wav_p,
            'F2': audio_att_1,
            'F3': audio_att_2,
            'F4': output_att,
            'M': output_att
        }            
        

        return output
 