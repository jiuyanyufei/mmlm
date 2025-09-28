"""
配置文件

这个文件定义了多模态大模型的配置类:
1. ModelConfig - 模型结构配置
2. TrainingConfig - 训练参数配置
3. DataConfig - 数据处理配置

这些配置类使用Python的dataclass装饰器，提供了类型提示和默认值。
"""

from dataclasses import dataclass, field
from typing import List, Optional, Union, Dict, Any


@dataclass
class ModelConfig:
    """
    模型配置
    
    定义多模态大模型的结构参数，包括:
    - 预训练模型选择
    - 模型尺寸参数
    - 多模态配置
    - 输出模态配置
    """
    # 基础配置
    pretrained_text_model: str = "gpt2-medium"  # 预训练文本模型
    pretrained_vision_model: str = "google/vit-base-patch16-224"  # 预训练视觉模型
    pretrained_audio_model: str = "facebook/wav2vec2-base-960h"  # 预训练音频模型
    
    # 模型结构配置
    hidden_size: int = 768  # 隐藏层大小
    num_attention_heads: int = 12  # 注意力头数
    num_hidden_layers: int = 12  # 隐藏层数
    intermediate_size: int = 3072  # 中间层大小
    
    # 多模态配置
    max_text_length: int = 512  # 最大文本长度
    max_image_length: int = 50  # 最大图像序列长度
    max_video_frames: int = 16  # 最大视频帧数
    max_audio_length: int = 16000  # 最大音频长度（1秒）
    
    # 多模态输出配置
    enable_multimodal_output: bool = True  # 是否启用多模态输出
    output_modality_types: List[str] = field(default_factory=lambda: ["text", "image", "video", "audio"])  # 支持的输出模态类型
    modality_type_vocab_size: int = 4  # 模态类型词汇表大小
    
    # 图像解码器配置
    image_decoder_layers: int = 8  # 图像解码器层数
    image_decoder_heads: int = 16  # 图像解码器注意力头数
    
    # 视频解码器配置
    video_decoder_layers: int = 8  # 视频解码器层数
    video_decoder_heads: int = 16  # 视频解码器注意力头数
    
    # 音频解码器配置
    audio_decoder_layers: int = 6  # 音频解码器层数
    audio_decoder_heads: int = 12  # 音频解码器注意力头数
    
    # 训练配置
    dropout_prob: float = 0.1  # Dropout概率


@dataclass
class TrainingConfig:
    """
    训练配置
    
    定义模型训练的参数，包括:
    - 优化器设置
    - 学习率调度
    - 批次大小
    - 训练轮数
    - 评估和保存策略
    """
    # 基础训练参数
    output_dir: str = "./outputs"  # 输出目录
    num_train_epochs: int = 3  # 训练轮数
    per_device_train_batch_size: int = 8  # 每个设备的训练批次大小
    per_device_eval_batch_size: int = 8  # 每个设备的评估批次大小
    gradient_accumulation_steps: int = 4  # 梯度累积步数
    
    # 优化器参数
    learning_rate: float = 5e-5  # 学习率
    weight_decay: float = 0.01  # 权重衰减
    adam_beta1: float = 0.9  # Adam优化器beta1
    adam_beta2: float = 0.999  # Adam优化器beta2
    adam_epsilon: float = 1e-8  # Adam优化器epsilon
    max_grad_norm: float = 1.0  # 梯度裁剪最大范数
    
    # 学习率调度
    warmup_ratio: float = 0.1  # 预热比例
    lr_scheduler_type: str = "linear"  # 学习率调度类型
    
    # 训练控制
    logging_steps: int = 100  # 日志记录步数
    save_steps: int = 1000  # 保存模型步数
    eval_steps: int = 1000  # 评估步数
    save_total_limit: int = 3  # 保存的检查点总数限制
    
    # 混合精度训练
    fp16: bool = False  # 是否使用混合精度训练
    fp16_opt_level: str = "O1"  # 混合精度优化级别
    
    # 分布式训练
    local_rank: int = -1  # 本地排名
    
    # 实验跟踪
    wandb: bool = False  # 是否使用Weights & Biases
    wandb_project: str = "mmlm"  # Weights & Biases项目名称
    
    # 多模态输出训练
    modality_loss_weights: Dict[str, float] = field(default_factory=lambda: {
        "text": 1.0,  # 文本损失权重
        "image": 1.0,  # 图像损失权重
        "video": 1.0,  # 视频损失权重
        "audio": 1.0,  # 音频损失权重
        "modality_type": 0.5  # 模态类型预测损失权重
    })


@dataclass
class DataConfig:
    """
    数据配置
    
    定义数据处理和加载的参数，包括:
    - 数据路径
    - 处理参数
    - 数据增强设置
    """
    # 数据路径
    train_file: str = ""  # 训练数据文件
    validation_file: Optional[str] = None  # 验证数据文件
    data_dir: str = "./data"  # 数据目录
    
    # 文本处理
    max_text_length: int = 512  # 最大文本长度
    
    # 图像处理
    image_size: int = 224  # 图像大小
    image_mean: List[float] = field(default_factory=lambda: [0.485, 0.456, 0.406])  # 图像均值
    image_std: List[float] = field(default_factory=lambda: [0.229, 0.224, 0.225])  # 图像标准差
    
    # 视频处理
    num_frames: int = 8  # 视频帧数
    frame_size: int = 224  # 视频帧大小
    video_sample_rate: int = 4  # 视频采样率
    
    # 音频处理
    audio_sample_rate: int = 16000  # 音频采样率
    max_audio_length: int = 10  # 最大音频长度(秒)
    audio_feature_type: str = "waveform"  # 音频特征类型
    
    # 数据加载
    num_workers: int = 4  # 数据加载线程数
    pin_memory: bool = True  # 是否将数据固定在内存中
    
    # 多模态输出
    enable_multimodal_output: bool = True  # 是否启用多模态输出
    output_modality_ratio: Dict[str, float] = field(default_factory=lambda: {
        "text": 0.7,  # 文本输出比例
        "image": 0.1,  # 图像输出比例
        "video": 0.1,  # 视频输出比例
        "audio": 0.1   # 音频输出比例
    })