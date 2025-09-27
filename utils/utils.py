"""
通用工具函数
"""
import os
import random
import numpy as np
import torch
import yaml
from typing import Dict, Any, Union, Optional


def set_seed(seed: int = 42) -> None:
    """
    设置随机种子以确保可重现性
    
    Args:
        seed: 随机种子
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # 设置确定性算法
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_config_to_yaml(config: Dict[str, Any], file_path: str) -> None:
    """
    将配置保存为YAML文件
    
    Args:
        config: 配置字典
        file_path: 保存路径
    """
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False)


def load_config_from_yaml(file_path: str) -> Dict[str, Any]:
    """
    从YAML文件加载配置
    
    Args:
        file_path: YAML文件路径
        
    Returns:
        Dict[str, Any]: 配置字典
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def get_device() -> torch.device:
    """
    获取可用的设备
    
    Returns:
        torch.device: 可用的设备
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def count_parameters(model: torch.nn.Module) -> int:
    """
    计算模型参数数量
    
    Args:
        model: PyTorch模型
        
    Returns:
        int: 参数数量
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def format_time(seconds: float) -> str:
    """
    格式化时间
    
    Args:
        seconds: 秒数
        
    Returns:
        str: 格式化后的时间字符串
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"