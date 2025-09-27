"""
视觉编码器模块
"""
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Union, Any, Tuple
from transformers import ViTModel, ViTConfig


class ImageEncoder(nn.Module):
    """图像编码器"""
    
    def __init__(
        self,
        hidden_size: int = 1024,
        vision_embed_dim: int = 1024,
        vision_patch_size: int = 14,
        vision_image_size: int = 224,
        vision_layers: int = 12,
        vision_heads: int = 16,
        projection_dim: int = 1024,
        pretrained: bool = True,
        model_name: str = "google/vit-base-patch16-224",
    ):
        """
        初始化图像编码器
        
        Args:
            hidden_size: 隐藏层大小
            vision_embed_dim: 视觉嵌入维度
            vision_patch_size: 视觉补丁大小
            vision_image_size: 视觉图像大小
            vision_layers: 视觉层数
            vision_heads: 视觉注意力头数
            projection_dim: 投影维度
            pretrained: 是否使用预训练模型
            model_name: 预训练模型名称
        """
        super().__init__()
        
        if pretrained:
            # 使用预训练ViT模型
            self.vision_model = ViTModel.from_pretrained(model_name)
        else:
            # 创建新的ViT模型
            config = ViTConfig(
                hidden_size=vision_embed_dim,
                num_hidden_layers=vision_layers,
                num_attention_heads=vision_heads,
                intermediate_size=vision_embed_dim * 4,
                patch_size=vision_patch_size,
                image_size=vision_image_size,
            )
            self.vision_model = ViTModel(config)
        
        # 投影层，将视觉特征投影到与文本相同的维度
        if vision_embed_dim != projection_dim:
            self.projection = nn.Linear(vision_embed_dim, projection_dim)
        else:
            self.projection = nn.Identity()
    
    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            pixel_values: 图像像素值 [batch_size, channels, height, width]
            
        Returns:
            torch.Tensor: 图像特征 [batch_size, seq_len, hidden_size]
        """
        # 获取视觉特征
        vision_outputs = self.vision_model(pixel_values=pixel_values)
        vision_embeds = vision_outputs.last_hidden_state  # [batch_size, seq_len, vision_embed_dim]
        
        # 投影到目标维度
        projected_embeds = self.projection(vision_embeds)  # [batch_size, seq_len, projection_dim]
        
        return projected_embeds


class VideoEncoder(nn.Module):
    """视频编码器"""
    
    def __init__(
        self,
        hidden_size: int = 1024,
        vision_embed_dim: int = 1024,
        vision_patch_size: int = 14,
        vision_image_size: int = 224,
        vision_layers: int = 12,
        vision_heads: int = 16,
        projection_dim: int = 1024,
        pretrained: bool = True,
        model_name: str = "google/vit-base-patch16-224",
    ):
        """
        初始化视频编码器
        
        Args:
            hidden_size: 隐藏层大小
            vision_embed_dim: 视觉嵌入维度
            vision_patch_size: 视觉补丁大小
            vision_image_size: 视觉图像大小
            vision_layers: 视觉层数
            vision_heads: 视觉注意力头数
            projection_dim: 投影维度
            pretrained: 是否使用预训练模型
            model_name: 预训练模型名称
        """
        super().__init__()
        
        # 使用图像编码器处理每一帧
        self.image_encoder = ImageEncoder(
            hidden_size=hidden_size,
            vision_embed_dim=vision_embed_dim,
            vision_patch_size=vision_patch_size,
            vision_image_size=vision_image_size,
            vision_layers=vision_layers,
            vision_heads=vision_heads,
            projection_dim=projection_dim,
            pretrained=pretrained,
            model_name=model_name,
        )
        
        # 时序融合层
        self.temporal_fusion = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(
                d_model=projection_dim,
                nhead=8,
                dim_feedforward=projection_dim * 4,
                batch_first=True,
            ),
            num_layers=2,
        )
    
    def forward(self, video_frames: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            video_frames: 视频帧 [batch_size, num_frames, channels, height, width]
            
        Returns:
            torch.Tensor: 视频特征 [batch_size, seq_len, hidden_size]
        """
        batch_size, num_frames, channels, height, width = video_frames.shape
        
        # 重塑为 [batch_size * num_frames, channels, height, width]
        frames = video_frames.view(-1, channels, height, width)
        
        # 使用图像编码器处理每一帧
        frame_features = self.image_encoder(frames)  # [batch_size * num_frames, seq_len, hidden_size]
        
        # 提取 [CLS] 标记特征
        cls_features = frame_features[:, 0]  # [batch_size * num_frames, hidden_size]
        
        # 重塑为 [batch_size, num_frames, hidden_size]
        cls_features = cls_features.view(batch_size, num_frames, -1)
        
        # 时序融合
        video_features = self.temporal_fusion(cls_features)  # [batch_size, num_frames, hidden_size]
        
        return video_features