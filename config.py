"""
多模态大模型配置文件
"""
import os
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Union


@dataclass
class ModelConfig:
    """模型配置"""
    # 基础配置
    model_name: str = "mmlm"
    model_type: str = "gpt"  # gpt, llama, etc.
    
    # 模型结构参数
    hidden_size: int = 1024
    num_hidden_layers: int = 24
    num_attention_heads: int = 16
    intermediate_size: int = 4096
    hidden_act: str = "gelu"
    hidden_dropout_prob: float = 0.1
    attention_probs_dropout_prob: float = 0.1
    max_position_embeddings: int = 2048
    
    # 词汇表大小
    vocab_size: int = 50257  # GPT-2 词汇表大小
    
    # 多模态配置
    vision_embed_dim: int = 1024
    vision_patch_size: int = 14
    vision_image_size: int = 224
    vision_layers: int = 12
    vision_heads: int = 16
    
    # 视频配置
    video_frames: int = 8
    video_frame_size: int = 224
    
    # 投影层配置
    projection_dim: int = 1024
    
    # 训练配置
    max_seq_length: int = 1024
    
    def to_dict(self) -> Dict[str, Any]:
        """将配置转换为字典"""
        return {k: v for k, v in self.__dict__.items()}


@dataclass
class TrainingConfig:
    """训练配置"""
    # 基础训练参数
    output_dir: str = "./outputs"
    overwrite_output_dir: bool = True
    do_train: bool = True
    do_eval: bool = True
    
    # 训练超参数
    per_device_train_batch_size: int = 8
    per_device_eval_batch_size: int = 8
    gradient_accumulation_steps: int = 4
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    num_train_epochs: int = 3
    
    # 学习率调度
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = 0.1
    
    # 混合精度训练
    fp16: bool = True
    
    # 分布式训练
    local_rank: int = -1
    
    # 保存和评估策略
    save_strategy: str = "steps"
    save_steps: int = 1000
    eval_strategy: str = "steps"
    eval_steps: int = 1000
    
    # 日志
    logging_dir: str = "./logs"
    logging_strategy: str = "steps"
    logging_steps: int = 100
    
    # 数据集配置
    train_file: Optional[str] = None
    validation_file: Optional[str] = None
    
    # 随机种子
    seed: int = 42
    
    def to_dict(self) -> Dict[str, Any]:
        """将配置转换为字典"""
        return {k: v for k, v in self.__dict__.items()}


@dataclass
class DataConfig:
    """数据配置"""
    # 数据路径
    data_dir: str = "./data"
    
    # 图像配置
    image_size: int = 224
    image_column: str = "image"
    
    # 视频配置
    video_column: str = "video"
    num_frames: int = 8
    frame_sample_rate: int = 4
    
    # 文本配置
    text_column: str = "text"
    max_text_length: int = 512
    
    # 数据处理
    preprocessing_num_workers: int = 4
    
    def to_dict(self) -> Dict[str, Any]:
        """将配置转换为字典"""
        return {k: v for k, v in self.__dict__.items()}


@dataclass
class InferenceConfig:
    """推理配置"""
    # 基础配置
    checkpoint_path: str = "./outputs/checkpoint-final"
    device: str = "cuda"
    
    # 生成参数
    max_new_tokens: int = 100
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    repetition_penalty: float = 1.2
    do_sample: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """将配置转换为字典"""
        return {k: v for k, v in self.__dict__.items()}


# 默认配置实例
MODEL_CONFIG = ModelConfig()
TRAINING_CONFIG = TrainingConfig()
DATA_CONFIG = DataConfig()
INFERENCE_CONFIG = InferenceConfig()