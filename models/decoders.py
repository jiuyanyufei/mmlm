"""
多模态解码器模块

这个模块实现了多种模态的解码器，用于生成不同类型的输出：
1. ImageDecoder - 生成图像输出
2. VideoDecoder - 生成视频输出
3. AudioDecoder - 生成音频输出

这些解码器将模型的隐藏表示转换为相应模态的输出。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from config import ModelConfig

class ImageDecoder(nn.Module):
    """
    图像解码器
    
    将模型的隐藏表示解码为图像输出。
    使用转置卷积网络逐步上采样特征图，生成图像。
    """
    
    def __init__(self, config: ModelConfig):
        """
        初始化图像解码器
        
        Args:
            config: 模型配置
        """
        super().__init__()
        
        # 隐藏表示到初始特征图的映射
        self.initial_projection = nn.Linear(config.hidden_size, 4 * 4 * 512)
        
        # 转置卷积层，逐步上采样
        self.decoder = nn.Sequential(
            # 第1层: 4x4x512 -> 8x8x256
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            
            # 第2层: 8x8x256 -> 16x16x128
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            # 第3层: 16x16x128 -> 32x32x64
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            # 第4层: 32x32x64 -> 64x64x32
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            # 第5层: 64x64x32 -> 128x128x16
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            
            # 第6层: 128x128x16 -> 224x224x3 (最终图像输出)
            nn.ConvTranspose2d(16, 3, kernel_size=4, stride=2, padding=1),
            nn.Tanh()  # 输出范围[-1, 1]
        )
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            hidden_states: 模型隐藏状态，形状为[batch_size, hidden_size]
            
        Returns:
            torch.Tensor: 生成的图像，形状为[batch_size, 3, 224, 224]
        """
        batch_size = hidden_states.shape[0]
        
        # 步骤1: 将隐藏状态映射到初始特征图
        x = self.initial_projection(hidden_states)
        # x形状: [batch_size, 4*4*512]
        
        # 步骤2: 重塑为4D张量，用于转置卷积
        x = x.view(batch_size, 512, 4, 4)
        # x形状: [batch_size, 512, 4, 4]
        
        # 步骤3: 通过解码器生成图像
        x = self.decoder(x)
        # x形状: [batch_size, 3, 224, 224]
        
        return x


class VideoDecoder(nn.Module):
    """
    视频解码器
    
    将模型的隐藏表示解码为视频输出。
    使用3D转置卷积网络生成时间维度和空间维度的特征。
    """
    
    def __init__(self, config: ModelConfig):
        """
        初始化视频解码器
        
        Args:
            config: 模型配置
        """
        super().__init__()
        
        # 隐藏表示到初始特征体的映射
        self.initial_projection = nn.Linear(config.hidden_size, 2 * 4 * 4 * 512)
        
        # 3D转置卷积层，同时上采样时间和空间维度
        self.decoder = nn.Sequential(
            # 第1层: 2x4x4x512 -> 4x8x8x256
            nn.ConvTranspose3d(512, 256, kernel_size=(2, 4, 4), stride=(2, 2, 2), padding=(0, 1, 1)),
            nn.BatchNorm3d(256),
            nn.ReLU(inplace=True),
            
            # 第2层: 4x8x8x256 -> 8x16x16x128
            nn.ConvTranspose3d(256, 128, kernel_size=(2, 4, 4), stride=(2, 2, 2), padding=(0, 1, 1)),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            
            # 第3层: 8x16x16x128 -> 8x32x32x64
            nn.ConvTranspose3d(128, 64, kernel_size=(1, 4, 4), stride=(1, 2, 2), padding=(0, 1, 1)),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            
            # 第4层: 8x32x32x64 -> 8x64x64x32
            nn.ConvTranspose3d(64, 32, kernel_size=(1, 4, 4), stride=(1, 2, 2), padding=(0, 1, 1)),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            
            # 第5层: 8x64x64x32 -> 8x128x128x16
            nn.ConvTranspose3d(32, 16, kernel_size=(1, 4, 4), stride=(1, 2, 2), padding=(0, 1, 1)),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),
            
            # 第6层: 8x128x128x16 -> 8x224x224x3 (最终视频输出)
            nn.ConvTranspose3d(16, 3, kernel_size=(1, 4, 4), stride=(1, 2, 2), padding=(0, 1, 1)),
            nn.Tanh()  # 输出范围[-1, 1]
        )
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            hidden_states: 模型隐藏状态，形状为[batch_size, hidden_size]
            
        Returns:
            torch.Tensor: 生成的视频，形状为[batch_size, 8, 3, 224, 224]
        """
        batch_size = hidden_states.shape[0]
        
        # 步骤1: 将隐藏状态映射到初始特征体
        x = self.initial_projection(hidden_states)
        # x形状: [batch_size, 2*4*4*512]
        
        # 步骤2: 重塑为5D张量，用于3D转置卷积
        x = x.view(batch_size, 512, 2, 4, 4)
        # x形状: [batch_size, 512, 2, 4, 4]
        
        # 步骤3: 通过解码器生成视频
        x = self.decoder(x)
        # x形状: [batch_size, 3, 8, 224, 224]
        
        # 步骤4: 调整维度顺序为[batch_size, frames, channels, height, width]
        x = x.permute(0, 2, 1, 3, 4)
        # x形状: [batch_size, 8, 3, 224, 224]
        
        return x


class AudioDecoder(nn.Module):
    """
    音频解码器
    
    将模型的隐藏表示解码为音频输出。
    使用1D转置卷积网络生成音频波形。
    """
    
    def __init__(self, config: ModelConfig):
        """
        初始化音频解码器
        
        Args:
            config: 模型配置
        """
        super().__init__()
        
        # 隐藏表示到初始特征序列的映射
        self.initial_projection = nn.Linear(config.hidden_size, 256 * 32)
        
        # 1D转置卷积层，逐步上采样音频序列
        self.decoder = nn.Sequential(
            # 第1层: 32x256 -> 64x128
            nn.ConvTranspose1d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            
            # 第2层: 64x128 -> 128x64
            nn.ConvTranspose1d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            
            # 第3层: 128x64 -> 256x32
            nn.ConvTranspose1d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            
            # 第4层: 256x32 -> 512x16
            nn.ConvTranspose1d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(16),
            nn.ReLU(inplace=True),
            
            # 第5层: 512x16 -> 1024x8
            nn.ConvTranspose1d(16, 8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(8),
            nn.ReLU(inplace=True),
            
            # 第6层: 1024x8 -> 2048x4
            nn.ConvTranspose1d(8, 4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(4),
            nn.ReLU(inplace=True),
            
            # 第7层: 2048x4 -> 4096x2
            nn.ConvTranspose1d(4, 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(2),
            nn.ReLU(inplace=True),
            
            # 第8层: 4096x2 -> 8192x1
            nn.ConvTranspose1d(2, 1, kernel_size=4, stride=2, padding=1),
            nn.Tanh()  # 输出范围[-1, 1]
        )
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            hidden_states: 模型隐藏状态，形状为[batch_size, hidden_size]
            
        Returns:
            torch.Tensor: 生成的音频，形状为[batch_size, 1, 8192]
        """
        batch_size = hidden_states.shape[0]
        
        # 步骤1: 将隐藏状态映射到初始特征序列
        x = self.initial_projection(hidden_states)
        # x形状: [batch_size, 256*32]
        
        # 步骤2: 重塑为3D张量，用于1D转置卷积
        x = x.view(batch_size, 256, 32)
        # x形状: [batch_size, 256, 32]
        
        # 步骤3: 通过解码器生成音频
        x = self.decoder(x)
        # x形状: [batch_size, 1, 8192]
        
        return x