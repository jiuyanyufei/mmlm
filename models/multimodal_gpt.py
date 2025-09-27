"""
多模态GPT模型
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Union, Any, Tuple
from transformers import GPT2Model, GPT2Config, GPT2LMHeadModel

from models.vision_encoder import ImageEncoder, VideoEncoder


class MultiModalGPT(nn.Module):
    """多模态GPT模型"""
    
    def __init__(
        self,
        hidden_size: int = 1024,
        num_hidden_layers: int = 24,
        num_attention_heads: int = 16,
        intermediate_size: int = 4096,
        hidden_act: str = "gelu",
        hidden_dropout_prob: float = 0.1,
        attention_probs_dropout_prob: float = 0.1,
        max_position_embeddings: int = 2048,
        vocab_size: int = 50257,  # GPT-2 词汇表大小
        vision_embed_dim: int = 1024,
        vision_patch_size: int = 14,
        vision_image_size: int = 224,
        vision_layers: int = 12,
        vision_heads: int = 16,
        video_frames: int = 8,
        projection_dim: int = 1024,
        pretrained_text_model: str = "gpt2-medium",
        pretrained_vision_model: str = "google/vit-base-patch16-224",
    ):
        """
        初始化多模态GPT模型
        
        Args:
            hidden_size: 隐藏层大小
            num_hidden_layers: 隐藏层数量
            num_attention_heads: 注意力头数量
            intermediate_size: 中间层大小
            hidden_act: 隐藏层激活函数
            hidden_dropout_prob: 隐藏层dropout概率
            attention_probs_dropout_prob: 注意力概率dropout概率
            max_position_embeddings: 最大位置嵌入数
            vocab_size: 词汇表大小
            vision_embed_dim: 视觉嵌入维度
            vision_patch_size: 视觉补丁大小
            vision_image_size: 视觉图像大小
            vision_layers: 视觉层数
            vision_heads: 视觉注意力头数
            video_frames: 视频帧数
            projection_dim: 投影维度
            pretrained_text_model: 预训练文本模型名称
            pretrained_vision_model: 预训练视觉模型名称
        """
        super().__init__()
        
        # 图像编码器
        self.image_encoder = ImageEncoder(
            hidden_size=hidden_size,
            vision_embed_dim=vision_embed_dim,
            vision_patch_size=vision_patch_size,
            vision_image_size=vision_image_size,
            vision_layers=vision_layers,
            vision_heads=vision_heads,
            projection_dim=projection_dim,
            pretrained=True,
            model_name=pretrained_vision_model,
        )
        
        # 视频编码器
        self.video_encoder = VideoEncoder(
            hidden_size=hidden_size,
            vision_embed_dim=vision_embed_dim,
            vision_patch_size=vision_patch_size,
            vision_image_size=vision_image_size,
            vision_layers=vision_layers,
            vision_heads=vision_heads,
            projection_dim=projection_dim,
            pretrained=True,
            model_name=pretrained_vision_model,
        )
        
        # 文本模型
        self.text_model = GPT2LMHeadModel.from_pretrained(pretrained_text_model)
        
        # 确保文本模型的隐藏层大小与投影维度匹配
        text_hidden_size = self.text_model.config.hidden_size
        if text_hidden_size != projection_dim:
            # 调整投影维度以匹配文本模型
            self.image_encoder.projection = nn.Linear(vision_embed_dim, text_hidden_size)
            projection_dim = text_hidden_size
        
        # 模态类型嵌入
        self.modality_type_embeddings = nn.Embedding(3, projection_dim)  # 0: 文本, 1: 图像, 2: 视频
        
        # 模态融合层
        self.modality_fusion = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(
                d_model=projection_dim,
                nhead=8,
                dim_feedforward=projection_dim * 4,
                batch_first=True,
            ),
            num_layers=2,
        )
    
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        image: Optional[torch.Tensor] = None,
        video: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        前向传播
        
        Args:
            input_ids: 输入ID [batch_size, seq_len]
            attention_mask: 注意力掩码 [batch_size, seq_len]
            image: 图像特征 [batch_size, channels, height, width]
            video: 视频特征 [batch_size, num_frames, channels, height, width]
            labels: 标签 [batch_size, seq_len]
            
        Returns:
            Dict[str, torch.Tensor]: 包含损失和logits的字典
        """
        batch_size = input_ids.shape[0] if input_ids is not None else (
            image.shape[0] if image is not None else video.shape[0]
        )
        
        # 处理文本输入
        if input_ids is not None:
            # 获取文本嵌入
            text_outputs = self.text_model.transformer(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True,
            )
            text_embeds = text_outputs.last_hidden_state  # [batch_size, seq_len, hidden_size]
            
            # 添加模态类型嵌入
            text_type_embeds = self.modality_type_embeddings(
                torch.zeros(batch_size, text_embeds.shape[1], dtype=torch.long, device=text_embeds.device)
            )
            text_embeds = text_embeds + text_type_embeds
        else:
            text_embeds = None
        
        # 处理图像输入
        if image is not None:
            # 获取图像嵌入
            image_embeds = self.image_encoder(image)  # [batch_size, img_seq_len, hidden_size]
            
            # 添加模态类型嵌入
            image_type_embeds = self.modality_type_embeddings(
                torch.ones(batch_size, image_embeds.shape[1], dtype=torch.long, device=image_embeds.device)
            )
            image_embeds = image_embeds + image_type_embeds
        else:
            image_embeds = None
        
        # 处理视频输入
        if video is not None:
            # 获取视频嵌入
            video_embeds = self.video_encoder(video)  # [batch_size, vid_seq_len, hidden_size]
            
            # 添加模态类型嵌入
            video_type_embeds = self.modality_type_embeddings(
                torch.full((batch_size, video_embeds.shape[1]), 2, dtype=torch.long, device=video_embeds.device)
            )
            video_embeds = video_embeds + video_type_embeds
        else:
            video_embeds = None
        
        # 融合多模态特征
        all_embeds = []
        all_masks = []
        
        if text_embeds is not None:
            all_embeds.append(text_embeds)
            all_masks.append(attention_mask)
        
        if image_embeds is not None:
            all_embeds.append(image_embeds)
            # 为图像创建全1掩码
            img_mask = torch.ones(batch_size, image_embeds.shape[1], device=image_embeds.device)
            all_masks.append(img_mask)
        
        if video_embeds is not None:
            all_embeds.append(video_embeds)
            # 为视频创建全1掩码
            vid_mask = torch.ones(batch_size, video_embeds.shape[1], device=video_embeds.device)
            all_masks.append(vid_mask)
        
        # 如果有多个模态，进行融合
        if len(all_embeds) > 1:
            # 拼接所有嵌入
            multimodal_embeds = torch.cat(all_embeds, dim=1)  # [batch_size, total_seq_len, hidden_size]
            multimodal_mask = torch.cat(all_masks, dim=1)  # [batch_size, total_seq_len]
            
            # 创建注意力掩码
            extended_mask = multimodal_mask.unsqueeze(1).unsqueeze(2)  # [batch_size, 1, 1, total_seq_len]
            extended_mask = (1.0 - extended_mask) * -10000.0  # 将0转换为大的负值
            
            # 模态融合
            fused_embeds = self.modality_fusion(multimodal_embeds)
        else:
            # 只有一个模态，直接使用
            fused_embeds = all_embeds[0]
            multimodal_mask = all_masks[0]
        
        # 将融合后的特征输入到GPT模型的语言模型头
        lm_logits = self.text_model.lm_head(fused_embeds)
        
        outputs = {"logits": lm_logits}
        
        # 计算损失（如果提供了标签）
        if labels is not None:
            # 只计算文本部分的损失
            text_len = text_embeds.shape[1] if text_embeds is not None else 0
            
            # 确保标签与logits形状匹配
            if text_len < lm_logits.shape[1]:
                # 如果融合后的序列长于文本序列，只使用文本部分的logits计算损失
                text_logits = lm_logits[:, -text_len:, :]
            else:
                text_logits = lm_logits
            
            # 计算损失
            shift_logits = text_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            
            outputs["loss"] = loss
        
        return outputs