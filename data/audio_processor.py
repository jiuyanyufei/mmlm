"""
音频处理器

这个文件实现了音频数据的处理功能，包括:
1. 加载音频文件
2. 重采样音频到指定采样率
3. 将音频转换为模型可用的特征表示
4. 批量处理多个音频文件
"""

import os                                  # 操作系统接口
import torch                               # PyTorch深度学习库
import numpy as np                         # 数值计算库
import librosa                             # 音频处理库
import soundfile as sf                     # 音频文件读写库
from typing import Dict, List, Union, Any, Tuple, Optional  # 类型提示

from utils.logger import logger            # 日志工具


class AudioProcessor:
    """
    音频处理器
    
    处理音频数据，将其转换为模型可用的特征表示。
    支持:
    1. 加载音频文件
    2. 重采样音频
    3. 提取音频特征
    4. 批量处理多个音频文件
    """
    
    def __init__(
        self,
        sample_rate: int = 16000,          # 采样率
        max_length: int = 10,              # 最大音频长度(秒)
        n_fft: int = 400,                  # FFT窗口大小
        hop_length: int = 160,             # 帧移
        n_mels: int = 80,                  # Mel滤波器组数量
        feature_type: str = "waveform",    # 特征类型: "waveform"或"mel"
    ):
        """
        初始化音频处理器
        
        Args:
            sample_rate: 目标采样率，默认为16000Hz
            max_length: 最大音频长度(秒)，默认为10秒
            n_fft: FFT窗口大小，默认为400
            hop_length: 帧移，默认为160
            n_mels: Mel滤波器组数量，默认为80
            feature_type: 特征类型，"waveform"表示原始波形，"mel"表示Mel频谱图
        """
        self.sample_rate = sample_rate
        self.max_length = max_length
        self.max_samples = max_length * sample_rate  # 最大采样点数
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.feature_type = feature_type
    
    def load_audio(self, audio_path: str) -> np.ndarray:
        """
        加载音频文件
        
        Args:
            audio_path: 音频文件路径
            
        Returns:
            np.ndarray: 加载的音频数据，形状为[num_samples]
        """
        try:
            # 使用librosa加载音频文件
            audio, sr = librosa.load(audio_path, sr=self.sample_rate, mono=True)
            
            # 记录日志
            logger.debug(f"加载音频文件: {audio_path}, 采样率: {sr}, 长度: {len(audio)/sr:.2f}秒")
            
            return audio
        except Exception as e:
            # 记录错误日志
            logger.error(f"加载音频文件失败: {audio_path}, 错误: {str(e)}")
            # 返回空音频
            return np.zeros(1)
    
    def extract_features(self, audio: np.ndarray) -> np.ndarray:
        """
        从音频数据中提取特征
        
        Args:
            audio: 音频数据，形状为[num_samples]
            
        Returns:
            np.ndarray: 提取的特征
                如果feature_type为"waveform"，形状为[1, num_samples]
                如果feature_type为"mel"，形状为[1, n_mels, num_frames]
        """
        # 确保音频长度不超过最大长度
        if len(audio) > self.max_samples:
            # 截断过长的音频
            audio = audio[:self.max_samples]
        elif len(audio) < self.max_samples:
            # 填充过短的音频
            padding = np.zeros(self.max_samples - len(audio))
            audio = np.concatenate([audio, padding])
        
        # 根据特征类型提取不同的特征
        if self.feature_type == "waveform":
            # 直接使用原始波形作为特征
            features = audio.reshape(1, -1)  # 添加通道维度
        elif self.feature_type == "mel":
            # 提取Mel频谱图特征
            mel_spectrogram = librosa.feature.melspectrogram(
                y=audio,
                sr=self.sample_rate,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                n_mels=self.n_mels
            )
            # 转换为分贝单位
            features = librosa.power_to_db(mel_spectrogram, ref=np.max)
            # 添加通道维度
            features = features.reshape(1, self.n_mels, -1)
        else:
            # 不支持的特征类型，使用原始波形
            logger.warning(f"不支持的特征类型: {self.feature_type}，使用原始波形")
            features = audio.reshape(1, -1)
        
        return features
    
    def __call__(self, audio_path: str) -> torch.Tensor:
        """
        处理单个音频文件
        
        Args:
            audio_path: 音频文件路径
            
        Returns:
            torch.Tensor: 处理后的音频特征
        """
        # 加载音频文件
        audio = self.load_audio(audio_path)
        
        # 提取特征
        features = self.extract_features(audio)
        
        # 转换为PyTorch张量
        features_tensor = torch.tensor(features, dtype=torch.float)
        
        return features_tensor
    
    def process_batch(self, audio_paths: List[str]) -> torch.Tensor:
        """
        批量处理多个音频文件
        
        Args:
            audio_paths: 音频文件路径列表
            
        Returns:
            torch.Tensor: 批量处理后的音频特征，形状为[batch_size, ...]
        """
        # 创建列表存储所有处理后的特征
        features_list = []
        
        # 处理每个音频文件
        for audio_path in audio_paths:
            # 调用__call__方法处理单个音频文件
            features = self(audio_path)
            # 添加到列表中
            features_list.append(features)
        
        # 堆叠所有特征，形成批次
        features_batch = torch.stack(features_list, dim=0)
        
        return features_batch
    
    def save_audio(self, audio: torch.Tensor, output_path: str) -> None:
        """
        保存音频数据到文件
        
        Args:
            audio: 音频数据，形状为[1, num_samples]或[num_samples]
            output_path: 输出文件路径
        """
        try:
            # 确保输出目录存在
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # 转换为numpy数组
            if isinstance(audio, torch.Tensor):
                audio_np = audio.detach().cpu().numpy()
            else:
                audio_np = audio
            
            # 如果是2D张量，取第一个通道
            if len(audio_np.shape) > 1 and audio_np.shape[0] == 1:
                audio_np = audio_np[0]
            
            # 归一化音频
            if np.abs(audio_np).max() > 1.0:
                audio_np = audio_np / np.abs(audio_np).max()
            
            # 保存音频文件
            sf.write(output_path, audio_np, self.sample_rate)
            
            # 记录日志
            logger.debug(f"保存音频文件: {output_path}, 采样率: {self.sample_rate}, 长度: {len(audio_np)/self.sample_rate:.2f}秒")
        except Exception as e:
            # 记录错误日志
            logger.error(f"保存音频文件失败: {output_path}, 错误: {str(e)}")