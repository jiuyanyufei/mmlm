"""
音频编码器模块

这个模块实现了音频编码器，用于将音频数据转换为模型可以处理的特征表示。
基于预训练的wav2vec2模型，提取音频特征并映射到模型隐藏空间。
"""

import torch
import torch.nn as nn
from transformers import Wav2Vec2Model, Wav2Vec2Config
from config import ModelConfig

class AudioEncoder(nn.Module):
    """
    音频编码器
    
    使用预训练的wav2vec2模型处理音频输入，并将其映射到模型的隐藏空间。
    """
    
    def __init__(self, config: ModelConfig):
        """
        初始化音频编码器
        
        Args:
            config: 模型配置
        """
        super().__init__()
        
        # 加载预训练的wav2vec2模型
        self.audio_model = Wav2Vec2Model.from_pretrained(config.pretrained_audio_model)
        
        # 冻结部分预训练模型参数以提高训练效率
        for param in self.audio_model.feature_extractor.parameters():
            param.requires_grad = False
        
        # 音频特征映射层 - 将wav2vec2特征映射到模型隐藏空间
        self.audio_projection = nn.Linear(
            self.audio_model.config.hidden_size,  # wav2vec2隐藏层大小
            config.hidden_size                    # 模型隐藏层大小
        )
        
        # 位置编码 - 为音频序列添加位置信息
        self.position_embeddings = nn.Embedding(
            config.max_audio_length,  # 最大音频长度
            config.hidden_size        # 隐藏层大小
        )
        
        # 层归一化
        self.layer_norm = nn.LayerNorm(config.hidden_size)
        
        # Dropout正则化
        self.dropout = nn.Dropout(config.dropout_prob)
    
    def forward(self, audio_inputs: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            audio_inputs: 音频输入张量，形状为[batch_size, sequence_length]
            
        Returns:
            torch.Tensor: 音频特征表示，形状为[batch_size, sequence_length, hidden_size]
        """
        # 步骤1: 通过wav2vec2模型提取音频特征
        # audio_inputs形状: [batch_size, sequence_length]
        audio_outputs = self.audio_model(audio_inputs).last_hidden_state
        # audio_outputs形状: [batch_size, sequence_length, wav2vec2_hidden_size]
        
        # 步骤2: 将音频特征映射到模型隐藏空间
        audio_features = self.audio_projection(audio_outputs)
        # audio_features形状: [batch_size, sequence_length, hidden_size]
        
        # 步骤3: 添加位置编码
        batch_size, seq_length = audio_features.shape[:2]
        position_ids = torch.arange(seq_length, device=audio_features.device).unsqueeze(0).expand(batch_size, -1)
        position_embeddings = self.position_embeddings(position_ids)
        # position_embeddings形状: [batch_size, sequence_length, hidden_size]
        
        # 步骤4: 将特征与位置编码相加
        audio_features = audio_features + position_embeddings
        
        # 步骤5: 应用层归一化和Dropout
        audio_features = self.layer_norm(audio_features)
        audio_features = self.dropout(audio_features)
        
        return audio_features