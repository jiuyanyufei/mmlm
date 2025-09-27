"""
数据集定义
"""
import os
import json
import torch
from torch.utils.data import Dataset
from typing import Dict, List, Optional, Union, Any, Callable
import pandas as pd

from utils.logger import logger
from data.processors import MultiModalProcessor


class MultiModalDataset(Dataset):
    """多模态数据集"""
    
    def __init__(
        self,
        data_file: str,
        processor: MultiModalProcessor,
        data_dir: Optional[str] = None,
        max_samples: Optional[int] = None,
    ):
        """
        初始化多模态数据集
        
        Args:
            data_file: 数据文件路径，支持json和csv格式
            processor: 多模态处理器
            data_dir: 数据目录，用于解析相对路径
            max_samples: 最大样本数，用于调试
        """
        self.processor = processor
        self.data_dir = data_dir or ""
        
        # 加载数据
        self.samples = self._load_data(data_file)
        
        # 限制样本数量
        if max_samples is not None and max_samples > 0:
            self.samples = self.samples[:max_samples]
        
        logger.info(f"加载了 {len(self.samples)} 个样本")
    
    def __len__(self) -> int:
        """返回数据集大小"""
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """获取数据样本"""
        sample = self.samples[idx]
        
        # 准备输入
        text = sample.get("text")
        
        # 处理图像路径
        image_path = sample.get("image")
        if image_path and self.data_dir and not os.path.isabs(image_path):
            image_path = os.path.join(self.data_dir, image_path)
        
        # 处理视频路径
        video_path = sample.get("video")
        if video_path and self.data_dir and not os.path.isabs(video_path):
            video_path = os.path.join(self.data_dir, video_path)
        
        # 处理样本
        features = self.processor.process_sample(
            text=text,
            image_path=image_path,
            video_path=video_path,
        )
        
        # 添加标签（如果有）
        if "label" in sample:
            features["label"] = sample["label"]
        
        return features
    
    def _load_data(self, data_file: str) -> List[Dict[str, Any]]:
        """
        加载数据文件
        
        Args:
            data_file: 数据文件路径
            
        Returns:
            List[Dict[str, Any]]: 数据样本列表
        """
        if not os.path.exists(data_file):
            raise FileNotFoundError(f"数据文件 {data_file} 不存在")
        
        file_ext = os.path.splitext(data_file)[1].lower()
        
        if file_ext == ".json":
            # 加载JSON文件
            with open(data_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            # 处理不同的JSON格式
            if isinstance(data, list):
                return data
            elif isinstance(data, dict) and "data" in data:
                return data["data"]
            else:
                raise ValueError(f"不支持的JSON格式: {data_file}")
        
        elif file_ext == ".csv":
            # 加载CSV文件
            df = pd.read_csv(data_file)
            return df.to_dict("records")
        
        elif file_ext == ".tsv":
            # 加载TSV文件
            df = pd.read_csv(data_file, sep="\t")
            return df.to_dict("records")
        
        else:
            raise ValueError(f"不支持的文件格式: {data_file}")


class MultiModalInstructionDataset(MultiModalDataset):
    """多模态指令数据集"""
    
    def __init__(
        self,
        data_file: str,
        processor: MultiModalProcessor,
        data_dir: Optional[str] = None,
        max_samples: Optional[int] = None,
        instruction_template: str = "以下是一个包含{modality}的任务。请根据{modality}回答问题：\n{instruction}\n",
    ):
        """
        初始化多模态指令数据集
        
        Args:
            data_file: 数据文件路径
            processor: 多模态处理器
            data_dir: 数据目录
            max_samples: 最大样本数
            instruction_template: 指令模板
        """
        super().__init__(data_file, processor, data_dir, max_samples)
        self.instruction_template = instruction_template
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """获取数据样本"""
        sample = self.samples[idx]
        
        # 获取指令和回答
        instruction = sample.get("instruction", "")
        answer = sample.get("answer", "")
        
        # 确定模态类型
        modalities = []
        if "image" in sample:
            modalities.append("图像")
        if "video" in sample:
            modalities.append("视频")
        modality_str = "和".join(modalities) if modalities else "文本"
        
        # 构建输入文本
        input_text = self.instruction_template.format(
            modality=modality_str,
            instruction=instruction,
        )
        
        # 处理图像路径
        image_path = sample.get("image")
        if image_path and self.data_dir and not os.path.isabs(image_path):
            image_path = os.path.join(self.data_dir, image_path)
        
        # 处理视频路径
        video_path = sample.get("video")
        if video_path and self.data_dir and not os.path.isabs(video_path):
            video_path = os.path.join(self.data_dir, video_path)
        
        # 处理样本
        features = self.processor.process_sample(
            text=input_text,
            image_path=image_path,
            video_path=video_path,
        )
        
        # 添加标签（回答）
        features["labels"] = self.processor.text_processor(answer)["input_ids"]
        
        return features


def create_data_loaders(
    train_file: str,
    processor: MultiModalProcessor,
    validation_file: Optional[str] = None,
    data_dir: Optional[str] = None,
    train_batch_size: int = 8,
    eval_batch_size: int = 8,
    num_workers: int = 4,
    is_instruction: bool = True,
    max_train_samples: Optional[int] = None,
    max_eval_samples: Optional[int] = None,
):
    """
    创建数据加载器
    
    Args:
        train_file: 训练数据文件
        processor: 多模态处理器
        validation_file: 验证数据文件
        data_dir: 数据目录
        train_batch_size: 训练批次大小
        eval_batch_size: 评估批次大小
        num_workers: 数据加载工作进程数
        is_instruction: 是否为指令数据集
        max_train_samples: 最大训练样本数
        max_eval_samples: 最大评估样本数
        
    Returns:
        tuple: (train_dataloader, eval_dataloader)
    """
    # 选择数据集类
    dataset_class = MultiModalInstructionDataset if is_instruction else MultiModalDataset
    
    # 创建训练数据集
    train_dataset = dataset_class(
        data_file=train_file,
        processor=processor,
        data_dir=data_dir,
        max_samples=max_train_samples,
    )
    
    # 创建训练数据加载器
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    # 创建评估数据加载器（如果有验证文件）
    eval_dataloader = None
    if validation_file:
        eval_dataset = dataset_class(
            data_file=validation_file,
            processor=processor,
            data_dir=data_dir,
            max_samples=max_eval_samples,
        )
        
        eval_dataloader = torch.utils.data.DataLoader(
            eval_dataset,
            batch_size=eval_batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )
    
    return train_dataloader, eval_dataloader