"""
视觉编码器模块

这个文件定义了两个主要的视觉编码器:
1. ImageEncoder - 处理单张图像
2. VideoEncoder - 处理视频帧序列

这些编码器将图像和视频转换为可以与文本特征融合的向量表示。
"""
# PyTorch是深度学习的核心库
import torch
import torch.nn as nn
# 类型提示，帮助IDE和开发者理解代码
from typing import Dict, List, Optional, Union, Any, Tuple
# 从Hugging Face的transformers库导入Vision Transformer相关模型
from transformers import ViTModel, ViTConfig


class ImageEncoder(nn.Module):
    """
    图像编码器
    
    使用Vision Transformer (ViT)将图像编码为特征向量序列。
    ViT将图像分割成小块(patches)，然后像处理文本标记一样处理这些图像块。
    
    工作流程:
    1. 将图像分割成固定大小的块
    2. 线性投影每个块
    3. 添加位置编码
    4. 通过Transformer编码器处理序列
    5. 输出特征向量序列
    """
    
    def __init__(
        self,
        hidden_size: int = 1024,         # 隐藏层维度大小
        vision_embed_dim: int = 1024,    # 视觉特征嵌入维度
        vision_patch_size: int = 14,     # 图像块大小，如14x14像素
        vision_image_size: int = 224,    # 输入图像大小，如224x224像素
        vision_layers: int = 12,         # Transformer编码器层数
        vision_heads: int = 16,          # 注意力头数量
        projection_dim: int = 1024,      # 输出特征维度
        pretrained: bool = True,         # 是否使用预训练模型
        model_name: str = "google/vit-base-patch16-224",  # 预训练模型名称
    ):
        """
        初始化图像编码器
        
        Args:
            hidden_size: 模型隐藏层大小，影响模型容量
            vision_embed_dim: 视觉特征的嵌入维度，决定特征的丰富程度
            vision_patch_size: 图像被分割的块大小，较小的块能捕获更细节的信息
            vision_image_size: 输入图像的大小，模型期望的输入尺寸
            vision_layers: Transformer编码器的层数，更多层可以学习更复杂的特征
            vision_heads: 注意力机制中的头数，多头注意力可以关注不同的特征
            projection_dim: 输出特征的维度，通常与文本模型的维度匹配
            pretrained: 是否加载预训练权重，通常使用预训练模型效果更好
            model_name: Hugging Face模型库中的预训练模型名称
        """
        super().__init__()
        
        # 第1部分: 初始化Vision Transformer模型
        if pretrained:
            # 使用预训练ViT模型 - 这通常是在大规模数据集(如ImageNet)上预训练的
            # 使用预训练模型可以大大加快训练速度并提高性能
            self.vision_model = ViTModel.from_pretrained(model_name)
        else:
            # 创建新的ViT模型 - 如果需要从头开始训练
            # 配置ViT模型的各种参数
            config = ViTConfig(
                hidden_size=vision_embed_dim,           # 隐藏层大小
                num_hidden_layers=vision_layers,        # Transformer层数
                num_attention_heads=vision_heads,       # 注意力头数
                intermediate_size=vision_embed_dim * 4, # 前馈网络中间层大小(通常是隐藏层的4倍)
                patch_size=vision_patch_size,           # 图像块大小
                image_size=vision_image_size,           # 输入图像大小
            )
            self.vision_model = ViTModel(config)
        
        # 第2部分: 特征维度调整
        # 投影层，将视觉特征投影到与文本模型相同的维度
        # 这对于后续的多模态融合非常重要
        if vision_embed_dim != projection_dim:
            # 如果视觉模型的输出维度与目标维度不同，使用线性层进行转换
            self.projection = nn.Linear(vision_embed_dim, projection_dim)
        else:
            # 如果维度已经匹配，使用恒等映射(不做任何改变)
            self.projection = nn.Identity()
    
    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        前向传播 - 处理输入图像并生成特征表示
        
        Args:
            pixel_values: 图像像素值，形状为 [batch_size, channels, height, width]
                          通常是归一化到[-1,1]或[0,1]范围的RGB图像
            
        Returns:
            torch.Tensor: 图像特征序列，形状为 [batch_size, seq_len, hidden_size]
                         其中seq_len = 1 + (image_size/patch_size)^2
                         第一个向量是特殊的[CLS]标记，后面是每个图像块的特征
        """
        # 第1步: 通过Vision Transformer处理图像
        # ViT将图像分割成块，添加位置编码，然后通过Transformer处理
        vision_outputs = self.vision_model(pixel_values=pixel_values)
        
        # 获取最后一层的隐藏状态作为视觉特征
        # 形状: [batch_size, seq_len, vision_embed_dim]
        # seq_len = 1 + (image_size/patch_size)^2，其中1是[CLS]标记
        vision_embeds = vision_outputs.last_hidden_state
        
        # 第2步: 投影到目标维度
        # 确保视觉特征的维度与后续处理所需的维度一致
        # 形状: [batch_size, seq_len, projection_dim]
        projected_embeds = self.projection(vision_embeds)
        
        return projected_embeds


class VideoEncoder(nn.Module):
    """
    视频编码器
    
    处理视频帧序列，将视频转换为特征向量序列。
    该编码器首先使用ImageEncoder处理每一帧，然后通过时序融合层
    捕获帧之间的时间关系。
    
    工作流程:
    1. 使用ImageEncoder处理每一帧
    2. 提取每帧的[CLS]标记特征
    3. 通过时序融合层处理帧序列
    4. 输出融合后的视频特征
    """
    
    def __init__(
        self,
        hidden_size: int = 1024,         # 隐藏层维度大小
        vision_embed_dim: int = 1024,    # 视觉特征嵌入维度
        vision_patch_size: int = 14,     # 图像块大小
        vision_image_size: int = 224,    # 输入图像大小
        vision_layers: int = 12,         # 视觉Transformer层数
        vision_heads: int = 16,          # 视觉注意力头数
        projection_dim: int = 1024,      # 输出特征维度
        pretrained: bool = True,         # 是否使用预训练模型
        model_name: str = "google/vit-base-patch16-224",  # 预训练模型名称
    ):
        """
        初始化视频编码器
        
        Args:
            hidden_size: 模型隐藏层大小，影响模型容量
            vision_embed_dim: 视觉特征的嵌入维度
            vision_patch_size: 图像被分割的块大小
            vision_image_size: 输入图像的大小
            vision_layers: 视觉Transformer的层数
            vision_heads: 视觉注意力机制中的头数
            projection_dim: 输出特征的维度，通常与文本模型的维度匹配
            pretrained: 是否加载预训练权重
            model_name: 预训练视觉模型的名称
        """
        super().__init__()
        
        # 第1部分: 图像编码器 - 用于处理视频中的每一帧
        # 重用ImageEncoder类来处理单帧图像
        self.image_encoder = ImageEncoder(
            hidden_size=hidden_size,             # 隐藏层大小
            vision_embed_dim=vision_embed_dim,   # 视觉嵌入维度
            vision_patch_size=vision_patch_size, # 图像块大小
            vision_image_size=vision_image_size, # 输入图像大小
            vision_layers=vision_layers,         # Transformer层数
            vision_heads=vision_heads,           # 注意力头数
            projection_dim=projection_dim,       # 输出特征维度
            pretrained=pretrained,               # 是否使用预训练模型
            model_name=model_name,               # 预训练模型名称
        )
        
        # 第2部分: 时序融合层 - 捕获帧之间的时间关系
        # 使用Transformer编码器来处理帧序列
        self.temporal_fusion = nn.TransformerEncoder(
            # 创建一个Transformer编码器层
            encoder_layer=nn.TransformerEncoderLayer(
                d_model=projection_dim,           # 模型维度
                nhead=8,                          # 注意力头数量(8是常用值)
                dim_feedforward=projection_dim * 4, # 前馈网络维度(通常是模型维度的4倍)
                batch_first=True,                 # 批次维度在前
            ),
            num_layers=2,                         # 使用2层Transformer编码器
        )
    
    def forward(self, video_frames: torch.Tensor) -> torch.Tensor:
        """
        前向传播 - 处理视频帧序列并生成特征表示
        
        Args:
            video_frames: 视频帧序列，形状为 [batch_size, num_frames, channels, height, width]
                         例如，一个批次的8帧RGB视频，每帧224x224像素
            
        Returns:
            torch.Tensor: 视频特征序列，形状为 [batch_size, num_frames, hidden_size]
                         表示整个视频的时序特征
        """
        # 第1步: 获取视频维度信息
        batch_size, num_frames, channels, height, width = video_frames.shape
        
        # 第2步: 重塑视频张量以便批量处理所有帧
        # 将batch_size和num_frames维度合并，这样可以一次性处理所有帧
        # 从[batch_size, num_frames, channels, height, width]
        # 变为[batch_size * num_frames, channels, height, width]
        frames = video_frames.view(-1, channels, height, width)
        
        # 第3步: 使用图像编码器处理每一帧
        # 每帧都会被转换为特征序列
        # 形状: [batch_size * num_frames, seq_len, hidden_size]
        frame_features = self.image_encoder(frames)
        
        # 第4步: 提取每帧的[CLS]标记特征
        # [CLS]标记是ViT输出的第一个标记，包含整个图像的全局表示
        # 形状: [batch_size * num_frames, hidden_size]
        cls_features = frame_features[:, 0]  # 索引0表示[CLS]标记
        
        # 第5步: 重塑特征以恢复批次和帧维度
        # 从[batch_size * num_frames, hidden_size]
        # 变为[batch_size, num_frames, hidden_size]
        cls_features = cls_features.view(batch_size, num_frames, -1)
        
        # 第6步: 通过时序融合层处理帧序列
        # Transformer编码器捕获帧之间的时间关系
        # 形状: [batch_size, num_frames, hidden_size]
        video_features = self.temporal_fusion(cls_features)
        
        return video_features