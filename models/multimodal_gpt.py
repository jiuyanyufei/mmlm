"""
多模态GPT模型

这个文件定义了多模态GPT模型的核心架构，它能够处理文本、图像和视频输入，
并将它们融合在一起生成文本输出。模型基于GPT-2架构，并添加了视觉编码器
来处理图像和视频输入。
"""
# PyTorch是深度学习的核心库
import torch
import torch.nn as nn
import torch.nn.functional as F
# 类型提示，帮助IDE和开发者理解代码
from typing import Dict, List, Optional, Union, Any, Tuple
# 从Hugging Face的transformers库导入GPT2相关模型
from transformers import GPT2Model, GPT2Config, GPT2LMHeadModel

# 导入我们自定义的视觉编码器，用于处理图像和视频
from models.vision_encoder import ImageEncoder, VideoEncoder


class MultiModalGPT(nn.Module):
    """
    多模态GPT模型
    
    这个模型结合了:
    1. 图像编码器 - 处理图像输入
    2. 视频编码器 - 处理视频输入
    3. GPT2语言模型 - 处理文本输入和生成
    4. 多模态融合机制 - 将不同模态的特征融合在一起
    
    模型工作流程:
    1. 分别编码文本、图像和视频输入
    2. 为每种模态添加模态类型嵌入
    3. 融合所有模态的特征
    4. 使用GPT2的语言模型头生成输出
    """
    
    def __init__(
        self,
        hidden_size: int = 1024,              # 隐藏层维度大小
        num_hidden_layers: int = 24,          # Transformer层数
        num_attention_heads: int = 16,        # 注意力头数量
        intermediate_size: int = 4096,        # 前馈网络中间层大小
        hidden_act: str = "gelu",             # 激活函数类型
        hidden_dropout_prob: float = 0.1,     # 隐藏层dropout比率
        attention_probs_dropout_prob: float = 0.1,  # 注意力dropout比率
        max_position_embeddings: int = 2048,  # 最大位置嵌入数量
        vocab_size: int = 50257,              # GPT-2词汇表大小
        vision_embed_dim: int = 1024,         # 视觉特征嵌入维度
        vision_patch_size: int = 14,          # 视觉补丁大小(ViT将图像分割成小块)
        vision_image_size: int = 224,         # 输入图像大小
        vision_layers: int = 12,              # 视觉Transformer层数
        vision_heads: int = 16,               # 视觉注意力头数量
        video_frames: int = 8,                # 处理的视频帧数
        projection_dim: int = 1024,           # 特征投影维度
        pretrained_text_model: str = "gpt2-medium",  # 预训练文本模型名称
        pretrained_vision_model: str = "google/vit-base-patch16-224",  # 预训练视觉模型
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
        
        # 第1部分: 图像编码器 - 使用Vision Transformer (ViT)处理图像
        # 这个编码器将图像转换为一系列特征向量
        self.image_encoder = ImageEncoder(
            hidden_size=hidden_size,             # 隐藏层大小
            vision_embed_dim=vision_embed_dim,   # 视觉嵌入维度
            vision_patch_size=vision_patch_size, # 图像被分割成的补丁大小
            vision_image_size=vision_image_size, # 输入图像大小
            vision_layers=vision_layers,         # Transformer层数
            vision_heads=vision_heads,           # 注意力头数量
            projection_dim=projection_dim,       # 输出特征维度
            pretrained=True,                     # 使用预训练模型
            model_name=pretrained_vision_model,  # 预训练模型名称
        )
        
        # 第2部分: 视频编码器 - 处理视频帧序列
        # 这个编码器首先使用图像编码器处理每一帧，然后添加时序信息
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
        
        # 第3部分: 文本模型 - 使用GPT2处理文本
        # 从Hugging Face加载预训练的GPT2模型
        self.text_model = GPT2LMHeadModel.from_pretrained(pretrained_text_model)
        
        # 第4部分: 特征维度对齐
        # 确保所有模态的特征维度一致，这对于后续融合非常重要
        text_hidden_size = self.text_model.config.hidden_size
        if text_hidden_size != projection_dim:
            # 如果文本模型的维度与投影维度不匹配，调整投影层
            self.image_encoder.projection = nn.Linear(vision_embed_dim, text_hidden_size)
            projection_dim = text_hidden_size
        
        # 第5部分: 模态类型嵌入 - 帮助模型区分不同类型的输入
        # 创建一个嵌入层，为每种模态类型分配一个可学习的向量
        # 0: 文本, 1: 图像, 2: 视频
        self.modality_type_embeddings = nn.Embedding(3, projection_dim)
        
        # 第6部分: 模态融合层 - 使用Transformer编码器融合不同模态
        # 这个模块将所有模态的特征融合在一起，使模型能够理解跨模态关系
        self.modality_fusion = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(
                d_model=projection_dim,           # 模型维度
                nhead=8,                          # 注意力头数量
                dim_feedforward=projection_dim * 4, # 前馈网络维度
                batch_first=True,                 # 批次维度在前
            ),
            num_layers=2,                         # Transformer层数
        )
    
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,      # 文本输入ID
        attention_mask: Optional[torch.Tensor] = None, # 文本注意力掩码
        image: Optional[torch.Tensor] = None,          # 图像输入
        video: Optional[torch.Tensor] = None,          # 视频输入
        labels: Optional[torch.Tensor] = None,         # 用于计算损失的标签
    ) -> Dict[str, torch.Tensor]:
        """
        模型的前向传播函数 - 处理输入并生成输出
        
        这个函数是模型的核心，它:
        1. 分别处理文本、图像和视频输入
        2. 为每种模态添加类型嵌入
        3. 融合所有模态的特征
        4. 生成输出logits和计算损失
        
        Args:
            input_ids: 文本输入ID [batch_size, seq_len]
            attention_mask: 文本注意力掩码，用于处理变长序列 [batch_size, seq_len]
            image: 图像特征 [batch_size, channels, height, width]
            video: 视频特征 [batch_size, num_frames, channels, height, width]
            labels: 用于计算损失的标签 [batch_size, seq_len]
            
        Returns:
            Dict[str, torch.Tensor]: 包含损失和logits的字典
        """
        # 第1步: 确定批次大小 - 从任何提供的输入中获取
        batch_size = input_ids.shape[0] if input_ids is not None else (
            image.shape[0] if image is not None else video.shape[0]
        )
        
        # 第2步: 处理文本输入
        if input_ids is not None:
            # 使用GPT2的transformer部分获取文本嵌入
            text_outputs = self.text_model.transformer(
                input_ids=input_ids,              # 输入ID
                attention_mask=attention_mask,    # 注意力掩码
                return_dict=True,                 # 返回字典格式的输出
            )
            # 获取最后一层的隐藏状态作为文本嵌入
            text_embeds = text_outputs.last_hidden_state  # [batch_size, seq_len, hidden_size]
            
            # 添加模态类型嵌入 - 帮助模型区分这是文本输入(类型0)
            # 创建一个全0张量，表示文本类型
            text_type_ids = torch.zeros(
                batch_size, text_embeds.shape[1], 
                dtype=torch.long, device=text_embeds.device
            )
            # 获取对应的嵌入向量
            text_type_embeds = self.modality_type_embeddings(text_type_ids)
            # 将类型嵌入添加到文本嵌入中
            text_embeds = text_embeds + text_type_embeds
        else:
            text_embeds = None
        
        # 第3步: 处理图像输入
        if image is not None:
            # 使用图像编码器处理图像
            image_embeds = self.image_encoder(image)  # [batch_size, img_seq_len, hidden_size]
            
            # 添加模态类型嵌入 - 帮助模型区分这是图像输入(类型1)
            # 创建一个全1张量，表示图像类型
            image_type_ids = torch.ones(
                batch_size, image_embeds.shape[1], 
                dtype=torch.long, device=image_embeds.device
            )
            # 获取对应的嵌入向量
            image_type_embeds = self.modality_type_embeddings(image_type_ids)
            # 将类型嵌入添加到图像嵌入中
            image_embeds = image_embeds + image_type_embeds
        else:
            image_embeds = None
        
        # 第4步: 处理视频输入
        if video is not None:
            # 使用视频编码器处理视频
            video_embeds = self.video_encoder(video)  # [batch_size, vid_seq_len, hidden_size]
            
            # 添加模态类型嵌入 - 帮助模型区分这是视频输入(类型2)
            # 创建一个全2张量，表示视频类型
            video_type_ids = torch.full(
                (batch_size, video_embeds.shape[1]), 
                2, dtype=torch.long, device=video_embeds.device
            )
            # 获取对应的嵌入向量
            video_type_embeds = self.modality_type_embeddings(video_type_ids)
            # 将类型嵌入添加到视频嵌入中
            video_embeds = video_embeds + video_type_embeds
        else:
            video_embeds = None
        
        # 第5步: 准备融合多模态特征
        # 创建列表存储所有模态的嵌入和掩码
        all_embeds = []  # 存储所有模态的嵌入
        all_masks = []   # 存储所有模态的注意力掩码
        
        # 添加文本嵌入和掩码(如果有)
        if text_embeds is not None:
            all_embeds.append(text_embeds)
            all_masks.append(attention_mask)
        
        # 添加图像嵌入和掩码(如果有)
        if image_embeds is not None:
            all_embeds.append(image_embeds)
            # 为图像创建全1掩码，表示所有图像标记都应该被注意到
            img_mask = torch.ones(batch_size, image_embeds.shape[1], device=image_embeds.device)
            all_masks.append(img_mask)
        
        # 添加视频嵌入和掩码(如果有)
        if video_embeds is not None:
            all_embeds.append(video_embeds)
            # 为视频创建全1掩码，表示所有视频标记都应该被注意到
            vid_mask = torch.ones(batch_size, video_embeds.shape[1], device=video_embeds.device)
            all_masks.append(vid_mask)
        
        # 第6步: 融合多模态特征
        if len(all_embeds) > 1:
            # 如果有多个模态，将它们拼接在一起
            # 在序列长度维度(dim=1)上拼接所有嵌入
            multimodal_embeds = torch.cat(all_embeds, dim=1)  # [batch_size, total_seq_len, hidden_size]
            # 同样拼接所有掩码
            multimodal_mask = torch.cat(all_masks, dim=1)  # [batch_size, total_seq_len]
            
            # 创建Transformer注意力掩码
            # 扩展维度以匹配注意力机制的需求
            extended_mask = multimodal_mask.unsqueeze(1).unsqueeze(2)  # [batch_size, 1, 1, total_seq_len]
            # 将0转换为大的负值(-10000.0)，这样在softmax后会变成接近0的值
            extended_mask = (1.0 - extended_mask) * -10000.0
            
            # 使用Transformer编码器融合所有模态
            fused_embeds = self.modality_fusion(multimodal_embeds)
        else:
            # 如果只有一个模态，直接使用它
            fused_embeds = all_embeds[0]
            multimodal_mask = all_masks[0]
        
        # 第7步: 生成输出logits
        # 使用GPT2的语言模型头将融合后的特征转换为词汇表上的概率分布
        lm_logits = self.text_model.lm_head(fused_embeds)  # [batch_size, seq_len, vocab_size]
        
        # 创建输出字典
        outputs = {"logits": lm_logits}
        
        # 第8步: 计算损失(如果提供了标签)
        if labels is not None:
            # 确定文本部分的长度
            text_len = text_embeds.shape[1] if text_embeds is not None else 0
            
            # 确保标签与logits形状匹配
            if text_len < lm_logits.shape[1]:
                # 如果融合后的序列长于文本序列，只使用相关部分的logits计算损失
                # 这通常是序列的后半部分，对应于我们想要预测的文本
                text_logits = lm_logits[:, -text_len:, :]
            else:
                text_logits = lm_logits
            
            # 计算语言模型损失
            # 在语言模型中，我们预测下一个标记，所以需要错开输入和标签
            # 输入: [A, B, C, D] -> 预测: [B, C, D, E]
            # 因此我们使用[:-1]和[1:]来创建这种错位
            shift_logits = text_logits[..., :-1, :].contiguous()  # 去掉最后一个位置
            shift_labels = labels[..., 1:].contiguous()           # 去掉第一个位置
            
            # 使用交叉熵损失函数
            loss_fct = nn.CrossEntropyLoss()
            # 重塑张量以适应损失函数的输入要求
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),  # [batch_size*seq_len, vocab_size]
                shift_labels.view(-1)                          # [batch_size*seq_len]
            )
            
            # 将损失添加到输出字典
            outputs["loss"] = loss
        
        return outputs