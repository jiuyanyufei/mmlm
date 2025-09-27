"""
数据处理器
"""
import os
import torch
import numpy as np
from PIL import Image
from typing import Dict, List, Union, Any, Tuple, Optional
import decord
from decord import VideoReader
import torchvision.transforms as T
from transformers import PreTrainedTokenizer

from utils.logger import logger


class ImageProcessor:
    """图像处理器"""
    
    def __init__(self, image_size: int = 224):
        """
        初始化图像处理器
        
        Args:
            image_size: 图像大小
        """
        self.image_size = image_size
        self.transform = T.Compose([
            T.Resize((image_size, image_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    
    def __call__(self, image_path: str) -> torch.Tensor:
        """
        处理图像
        
        Args:
            image_path: 图像路径
            
        Returns:
            torch.Tensor: 处理后的图像张量
        """
        try:
            image = Image.open(image_path).convert("RGB")
            return self.transform(image)
        except Exception as e:
            logger.error(f"处理图像 {image_path} 时出错: {e}")
            # 返回一个空白图像
            return torch.zeros(3, self.image_size, self.image_size)
    
    def process_batch(self, image_paths: List[str]) -> torch.Tensor:
        """
        批量处理图像
        
        Args:
            image_paths: 图像路径列表
            
        Returns:
            torch.Tensor: 处理后的图像张量批次
        """
        return torch.stack([self(path) for path in image_paths])


class VideoProcessor:
    """视频处理器"""
    
    def __init__(
        self,
        num_frames: int = 8,
        frame_size: int = 224,
        sample_rate: int = 4,
    ):
        """
        初始化视频处理器
        
        Args:
            num_frames: 采样帧数
            frame_size: 帧大小
            sample_rate: 采样率
        """
        self.num_frames = num_frames
        self.frame_size = frame_size
        self.sample_rate = sample_rate
        
        # 图像变换
        self.transform = T.Compose([
            T.ToPILImage(),
            T.Resize((frame_size, frame_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    
    def __call__(self, video_path: str) -> torch.Tensor:
        """
        处理视频
        
        Args:
            video_path: 视频路径
            
        Returns:
            torch.Tensor: 处理后的视频帧张量 [T, C, H, W]
        """
        try:
            # 使用decord加载视频
            decord.bridge.set_bridge('torch')
            vr = VideoReader(video_path)
            
            # 计算采样索引
            total_frames = len(vr)
            indices = self._sample_frames(total_frames)
            
            # 读取帧
            frames = vr.get_batch(indices)  # [T, H, W, C]
            
            # 处理帧
            processed_frames = []
            for frame in frames:
                processed_frame = self.transform(frame)
                processed_frames.append(processed_frame)
            
            # 堆叠帧
            return torch.stack(processed_frames)  # [T, C, H, W]
        
        except Exception as e:
            logger.error(f"处理视频 {video_path} 时出错: {e}")
            # 返回空白视频帧
            return torch.zeros(self.num_frames, 3, self.frame_size, self.frame_size)
    
    def _sample_frames(self, total_frames: int) -> List[int]:
        """
        采样视频帧索引
        
        Args:
            total_frames: 视频总帧数
            
        Returns:
            List[int]: 采样帧索引
        """
        if total_frames <= self.num_frames:
            # 如果总帧数小于等于采样帧数，则使用所有帧
            return list(range(total_frames))
        
        # 计算采样间隔
        if self.sample_rate > 0:
            # 固定采样率
            start_idx = max(0, (total_frames - self.num_frames * self.sample_rate) // 2)
            indices = [start_idx + i * self.sample_rate for i in range(self.num_frames)]
            indices = [min(i, total_frames - 1) for i in indices]
        else:
            # 均匀采样
            indices = np.linspace(0, total_frames - 1, self.num_frames, dtype=int).tolist()
        
        return indices
    
    def process_batch(self, video_paths: List[str]) -> torch.Tensor:
        """
        批量处理视频
        
        Args:
            video_paths: 视频路径列表
            
        Returns:
            torch.Tensor: 处理后的视频张量批次 [B, T, C, H, W]
        """
        return torch.stack([self(path) for path in video_paths])


class TextProcessor:
    """文本处理器"""
    
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        max_length: int = 512,
    ):
        """
        初始化文本处理器
        
        Args:
            tokenizer: 分词器
            max_length: 最大文本长度
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __call__(
        self,
        text: str,
        return_tensors: str = "pt",
    ) -> Dict[str, torch.Tensor]:
        """
        处理文本
        
        Args:
            text: 输入文本
            return_tensors: 返回张量类型
            
        Returns:
            Dict[str, torch.Tensor]: 处理后的文本特征
        """
        return self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors=return_tensors,
        )
    
    def process_batch(
        self,
        texts: List[str],
        return_tensors: str = "pt",
    ) -> Dict[str, torch.Tensor]:
        """
        批量处理文本
        
        Args:
            texts: 文本列表
            return_tensors: 返回张量类型
            
        Returns:
            Dict[str, torch.Tensor]: 处理后的文本特征批次
        """
        return self.tokenizer(
            texts,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors=return_tensors,
        )


class MultiModalProcessor:
    """多模态处理器"""
    
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        image_size: int = 224,
        num_frames: int = 8,
        frame_size: int = 224,
        video_sample_rate: int = 4,
        max_text_length: int = 512,
    ):
        """
        初始化多模态处理器
        
        Args:
            tokenizer: 分词器
            image_size: 图像大小
            num_frames: 视频帧数
            frame_size: 视频帧大小
            video_sample_rate: 视频采样率
            max_text_length: 最大文本长度
        """
        self.image_processor = ImageProcessor(image_size=image_size)
        self.video_processor = VideoProcessor(
            num_frames=num_frames,
            frame_size=frame_size,
            sample_rate=video_sample_rate,
        )
        self.text_processor = TextProcessor(
            tokenizer=tokenizer,
            max_length=max_text_length,
        )
    
    def process_sample(
        self,
        text: Optional[str] = None,
        image_path: Optional[str] = None,
        video_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        处理单个样本
        
        Args:
            text: 文本
            image_path: 图像路径
            video_path: 视频路径
            
        Returns:
            Dict[str, Any]: 处理后的特征
        """
        features = {}
        
        # 处理文本
        if text is not None:
            text_features = self.text_processor(text)
            features.update(text_features)
        
        # 处理图像
        if image_path is not None:
            image_features = self.image_processor(image_path)
            features["image"] = image_features
        
        # 处理视频
        if video_path is not None:
            video_features = self.video_processor(video_path)
            features["video"] = video_features
        
        return features
    
    def process_batch(
        self,
        texts: Optional[List[str]] = None,
        image_paths: Optional[List[str]] = None,
        video_paths: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        批量处理样本
        
        Args:
            texts: 文本列表
            image_paths: 图像路径列表
            video_paths: 视频路径列表
            
        Returns:
            Dict[str, Any]: 处理后的特征批次
        """
        features = {}
        
        # 处理文本
        if texts is not None:
            text_features = self.text_processor.process_batch(texts)
            features.update(text_features)
        
        # 处理图像
        if image_paths is not None:
            image_features = self.image_processor.process_batch(image_paths)
            features["image"] = image_features
        
        # 处理视频
        if video_paths is not None:
            video_features = self.video_processor.process_batch(video_paths)
            features["video"] = video_features
        
        return features