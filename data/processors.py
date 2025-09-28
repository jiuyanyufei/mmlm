"""
数据处理器

这个文件包含了处理多种模态数据的处理器类:
1. ImageProcessor - 处理图像数据
2. VideoProcessor - 处理视频数据
3. TextProcessor - 处理文本数据
4. AudioProcessor - 处理音频数据
5. MultiModalProcessor - 整合以上处理器，处理多模态输入和输出

这些处理器将原始数据转换为模型可以使用的张量格式，并支持任意形态的输入和输出。
"""
import os                                  # 操作系统接口
import torch                               # PyTorch深度学习库
import numpy as np                         # 数值计算库
from PIL import Image                      # 图像处理库
from typing import Dict, List, Union, Any, Tuple, Optional  # 类型提示
import decord                              # 高效视频解码库
from decord import VideoReader             # 视频读取器
import torchvision.transforms as T         # PyTorch视觉变换工具
from transformers import PreTrainedTokenizer  # Hugging Face分词器

from utils.logger import logger            # 日志工具
from data.audio_processor import AudioProcessor  # 导入音频处理器


class ImageProcessor:
    """
    图像处理器
    
    负责将原始图像文件转换为模型可用的标准化张量。
    处理步骤包括:
    1. 加载图像文件
    2. 调整大小到统一尺寸
    3. 转换为张量
    4. 标准化像素值
    """
    
    def __init__(self, image_size: int = 224):
        """
        初始化图像处理器
        
        Args:
            image_size: 输出图像的大小(高度和宽度)，默认为224x224像素
                       这个尺寸通常由预训练视觉模型决定
        """
        self.image_size = image_size
        
        # 创建图像变换管道
        self.transform = T.Compose([
            # 步骤1: 调整图像大小到统一尺寸
            T.Resize((image_size, image_size)),
            
            # 步骤2: 将PIL图像转换为张量 [0, 1]
            # 结果形状: [C, H, W] = [3, image_size, image_size]
            T.ToTensor(),
            
            # 步骤3: 标准化像素值 - 使用ImageNet预训练模型的均值和标准差
            # 这有助于模型更快收敛
            T.Normalize(
                # RGB通道的均值
                mean=[0.485, 0.456, 0.406],
                # RGB通道的标准差
                std=[0.229, 0.224, 0.225]
            ),
        ])
    
    def __call__(self, image_path: str) -> torch.Tensor:
        """
        处理单张图像
        
        这个方法使处理器可以像函数一样被调用: processor(image_path)
        
        Args:
            image_path: 图像文件的路径
            
        Returns:
            torch.Tensor: 处理后的图像张量，形状为[3, image_size, image_size]
                         其中3表示RGB三个通道
        """
        try:
            # 步骤1: 打开图像文件并确保它是RGB格式
            # 有些图像可能是灰度图或带Alpha通道，convert确保统一格式
            image = Image.open(image_path).convert("RGB")
            
            # 步骤2: 应用变换管道
            return self.transform(image)
            
        except Exception as e:
            # 如果处理失败(文件不存在、损坏等)，记录错误并返回零张量
            logger.error(f"处理图像 {image_path} 时出错: {e}")
            
            # 返回一个全零张量作为替代，避免程序崩溃
            # 形状: [3, image_size, image_size]
            return torch.zeros(3, self.image_size, self.image_size)
    
    def process_batch(self, image_paths: List[str]) -> torch.Tensor:
        """
        批量处理多张图像
        
        Args:
            image_paths: 图像文件路径列表
            
        Returns:
            torch.Tensor: 处理后的图像张量批次，形状为[B, 3, image_size, image_size]
                         其中B是批次大小(图像数量)
        """
        # 对每个路径调用__call__方法处理图像，然后将结果堆叠成批次
        # 列表推导式: [self(path) for path in image_paths] 创建处理后图像的列表
        # torch.stack: 将列表中的张量堆叠成一个新的维度(批次维度)
        return torch.stack([self(path) for path in image_paths])


class VideoProcessor:
    """
    视频处理器
    
    负责将原始视频文件转换为模型可用的帧序列张量。
    处理步骤包括:
    1. 加载视频文件
    2. 采样指定数量的帧
    3. 调整帧大小
    4. 标准化像素值
    5. 转换为张量序列
    """
    
    def __init__(
        self,
        num_frames: int = 8,      # 要采样的帧数
        frame_size: int = 224,    # 输出帧的大小
        sample_rate: int = 4,     # 采样率(每隔多少帧采样一次)
    ):
        """
        初始化视频处理器
        
        Args:
            num_frames: 从视频中采样的帧数，默认为8帧
                       这决定了模型处理的时间范围
            frame_size: 输出帧的大小(高度和宽度)，默认为224x224像素
                       通常由预训练视觉模型决定
            sample_rate: 视频帧采样率，默认为4(每隔4帧采样一次)
                        值为0时表示均匀采样
        """
        self.num_frames = num_frames
        self.frame_size = frame_size
        self.sample_rate = sample_rate
        
        # 创建帧变换管道
        self.transform = T.Compose([
            # 步骤1: 将numpy数组转换为PIL图像
            # decord返回的帧是numpy数组或torch张量
            T.ToPILImage(),
            
            # 步骤2: 调整帧大小到统一尺寸
            T.Resize((frame_size, frame_size)),
            
            # 步骤3: 将PIL图像转换为张量 [0, 1]
            T.ToTensor(),
            
            # 步骤4: 标准化像素值 - 使用ImageNet预训练模型的均值和标准差
            T.Normalize(
                mean=[0.485, 0.456, 0.406],  # RGB通道的均值
                std=[0.229, 0.224, 0.225]    # RGB通道的标准差
            ),
        ])
    
    def __call__(self, video_path: str) -> torch.Tensor:
        """
        处理单个视频
        
        这个方法使处理器可以像函数一样被调用: processor(video_path)
        
        Args:
            video_path: 视频文件的路径
            
        Returns:
            torch.Tensor: 处理后的视频帧张量，形状为[T, C, H, W]
                         其中T是帧数，C是通道数(3表示RGB)，H和W是帧的高度和宽度
        """
        try:
            # 步骤1: 配置decord使用PyTorch作为数据桥接
            # 这样decord返回的数据可以直接与PyTorch兼容
            decord.bridge.set_bridge('torch')
            
            # 步骤2: 使用decord加载视频文件
            # decord是一个高效的视频解码库，比OpenCV更快
            vr = VideoReader(video_path)
            
            # 步骤3: 计算要采样的帧索引
            total_frames = len(vr)  # 视频总帧数
            indices = self._sample_frames(total_frames)  # 调用采样方法获取索引
            
            # 步骤4: 一次性读取所有采样帧
            # 形状: [T, H, W, C] - T是帧数，H是高度，W是宽度，C是通道数
            frames = vr.get_batch(indices)
            
            # 步骤5: 处理每一帧
            processed_frames = []
            for frame in frames:
                # 应用变换管道到每一帧
                processed_frame = self.transform(frame)  # 形状: [C, H, W]
                processed_frames.append(processed_frame)
            
            # 步骤6: 将处理后的帧堆叠成序列
            # 形状: [T, C, H, W] - T是帧数，C是通道数，H和W是帧的高度和宽度
            return torch.stack(processed_frames)
        
        except Exception as e:
            # 如果处理失败(文件不存在、损坏等)，记录错误并返回零张量
            logger.error(f"处理视频 {video_path} 时出错: {e}")
            
            # 返回一个全零张量作为替代，避免程序崩溃
            # 形状: [num_frames, 3, frame_size, frame_size]
            return torch.zeros(self.num_frames, 3, self.frame_size, self.frame_size)
    
    def _sample_frames(self, total_frames: int) -> List[int]:
        """
        采样视频帧索引 - 从视频中选择要处理的帧
        
        Args:
            total_frames: 视频的总帧数
            
        Returns:
            List[int]: 采样帧的索引列表
        """
        # 情况1: 视频帧数不足
        if total_frames <= self.num_frames:
            # 如果总帧数小于等于需要的帧数，则使用所有可用帧
            return list(range(total_frames))
        
        # 情况2: 视频帧数充足，需要采样
        if self.sample_rate > 0:
            # 方法1: 固定采样率采样
            # 计算起始索引，使采样居中
            start_idx = max(0, (total_frames - self.num_frames * self.sample_rate) // 2)
            
            # 按固定间隔采样
            # 例如: sample_rate=4，从start_idx开始，每隔4帧采样一次
            indices = [start_idx + i * self.sample_rate for i in range(self.num_frames)]
            
            # 确保索引不超出视频范围
            indices = [min(i, total_frames - 1) for i in indices]
        else:
            # 方法2: 均匀采样
            # 在整个视频长度上均匀分布采样点
            # 例如: 对于100帧的视频采样8帧，会选择索引约为[0, 14, 28, 42, 57, 71, 85, 99]的帧
            indices = np.linspace(0, total_frames - 1, self.num_frames, dtype=int).tolist()
        
        return indices
    
    def process_batch(self, video_paths: List[str]) -> torch.Tensor:
        """
        批量处理多个视频
        
        Args:
            video_paths: 视频文件路径列表
            
        Returns:
            torch.Tensor: 处理后的视频张量批次，形状为[B, T, C, H, W]
                         其中B是批次大小(视频数量)，T是帧数，
                         C是通道数(3表示RGB)，H和W是帧的高度和宽度
        """
        # 对每个路径调用__call__方法处理视频，然后将结果堆叠成批次
        # 列表推导式: [self(path) for path in video_paths] 创建处理后视频的列表
        # torch.stack: 将列表中的张量堆叠成一个新的维度(批次维度)
        return torch.stack([self(path) for path in video_paths])


class TextProcessor:
    """
    文本处理器
    
    负责将原始文本转换为模型可用的标记ID序列。
    处理步骤包括:
    1. 使用分词器将文本分割成标记(tokens)
    2. 将标记转换为ID
    3. 添加特殊标记(如[CLS], [SEP])
    4. 填充或截断到固定长度
    5. 创建注意力掩码
    """
    
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,  # Hugging Face分词器
        max_length: int = 512,           # 最大序列长度
    ):
        """
        初始化文本处理器
        
        Args:
            tokenizer: Hugging Face预训练分词器，如GPT2Tokenizer
                      分词器负责将文本分割成标记并转换为ID
            max_length: 处理后的最大序列长度，默认为512
                       超过此长度的文本将被截断，不足的将被填充
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __call__(
        self,
        text: str,                  # 输入文本
        return_tensors: str = "pt", # 返回张量类型("pt"表示PyTorch)
    ) -> Dict[str, torch.Tensor]:
        """
        处理单个文本
        
        这个方法使处理器可以像函数一样被调用: processor(text)
        
        Args:
            text: 要处理的输入文本字符串
            return_tensors: 返回张量的类型，"pt"表示PyTorch张量
            
        Returns:
            Dict[str, torch.Tensor]: 包含处理后文本特征的字典，通常包含:
                - input_ids: 标记ID序列
                - attention_mask: 注意力掩码，指示哪些标记是真实内容(1)，哪些是填充(0)
                - 可能还有token_type_ids等其他特征，取决于分词器类型
        """
        # 使用Hugging Face分词器处理文本
        return self.tokenizer(
            text,                       # 输入文本
            max_length=self.max_length, # 最大长度
            padding="max_length",       # 填充策略：填充到max_length
            truncation=True,            # 启用截断：超长文本将被截断
            return_tensors=return_tensors, # 返回张量类型
        )
    
    def process_batch(
        self,
        texts: List[str],             # 文本列表
        return_tensors: str = "pt",   # 返回张量类型
    ) -> Dict[str, torch.Tensor]:
        """
        批量处理多个文本
        
        Args:
            texts: 要处理的文本字符串列表
            return_tensors: 返回张量的类型，"pt"表示PyTorch张量
            
        Returns:
            Dict[str, torch.Tensor]: 包含处理后文本特征的字典，每个张量的第一维是批次大小
                - input_ids: 形状为[batch_size, max_length]的标记ID序列
                - attention_mask: 形状为[batch_size, max_length]的注意力掩码
                - 可能还有其他特征
        """
        # 使用Hugging Face分词器批量处理文本
        # 分词器会自动处理批次中不同长度的文本，并创建适当的填充和掩码
        return self.tokenizer(
            texts,                      # 文本列表
            max_length=self.max_length, # 最大长度
            padding="max_length",       # 填充策略：填充到max_length
            truncation=True,            # 启用截断：超长文本将被截断
            return_tensors=return_tensors, # 返回张量类型
        )


class MultiModalProcessor:
    """
    多模态处理器
    
    整合文本、图像、视频和音频处理器，提供统一的接口处理多模态输入和输出。
    这个处理器可以同时处理:
    1. 文本数据
    2. 图像数据
    3. 视频数据
    4. 音频数据
    或者它们的任意组合，并支持任意模态的输出
    """
    
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,  # 文本分词器
        image_size: int = 224,           # 图像大小
        num_frames: int = 8,             # 视频帧数
        frame_size: int = 224,           # 视频帧大小
        video_sample_rate: int = 4,      # 视频采样率
        max_text_length: int = 512,      # 最大文本长度
        audio_sample_rate: int = 16000,  # 音频采样率
        max_audio_length: int = 10,      # 最大音频长度(秒)
    ):
        """
        初始化多模态处理器
        
        Args:
            tokenizer: Hugging Face预训练分词器，用于处理文本
            image_size: 处理后的图像大小，默认为224x224像素
            num_frames: 从视频中采样的帧数，默认为8帧
            frame_size: 处理后的视频帧大小，默认为224x224像素
            video_sample_rate: 视频帧采样率，默认为4
            max_text_length: 处理后的最大文本长度，默认为512
            audio_sample_rate: 音频采样率，默认为16000Hz
            max_audio_length: 最大音频长度，默认为10秒
        """
        # 创建各个模态的处理器
        
        # 图像处理器 - 处理图像输入
        self.image_processor = ImageProcessor(
            image_size=image_size
        )
        
        # 视频处理器 - 处理视频输入
        self.video_processor = VideoProcessor(
            num_frames=num_frames,           # 采样帧数
            frame_size=frame_size,           # 帧大小
            sample_rate=video_sample_rate,   # 采样率
        )
        
        # 文本处理器 - 处理文本输入
        self.text_processor = TextProcessor(
            tokenizer=tokenizer,             # 分词器
            max_length=max_text_length,      # 最大文本长度
        )
        
        # 音频处理器 - 处理音频输入
        self.audio_processor = AudioProcessor(
            sample_rate=audio_sample_rate,   # 采样率
            max_length=max_audio_length,     # 最大长度
        )
    
    def process_sample(
        self,
        text: Optional[str] = None,           # 可选的文本输入
        image_path: Optional[str] = None,      # 可选的图像路径
        video_path: Optional[str] = None,      # 可选的视频路径
        audio_path: Optional[str] = None,      # 可选的音频路径
        text_output: Optional[str] = None,     # 可选的文本输出
        image_output_path: Optional[str] = None, # 可选的图像输出路径
        video_output_path: Optional[str] = None, # 可选的视频输出路径
        audio_output_path: Optional[str] = None, # 可选的音频输出路径
        output_modality: Optional[int] = None,  # 输出模态类型
    ) -> Dict[str, Any]:
        """
        处理单个多模态样本
        
        可以同时处理文本、图像、视频和音频，或者它们的任意组合。
        每种模态都是可选的，只处理提供的模态。
        
        Args:
            text: 输入文本字符串
            image_path: 图像文件路径
            video_path: 视频文件路径
            audio_path: 音频文件路径
            text_output: 文本输出
            image_output_path: 图像输出文件路径
            video_output_path: 视频输出文件路径
            audio_output_path: 音频输出文件路径
            output_modality: 输出模态类型 (0:文本, 1:图像, 2:视频, 3:音频)
            
        Returns:
            Dict[str, Any]: 包含处理后特征的字典，可能包含:
                - input_ids, attention_mask: 文本特征
                - image: 图像特征
                - video: 视频特征
                - audio: 音频特征
                - labels: 文本输出特征
                - image_output: 图像输出特征
                - video_output: 视频输出特征
                - audio_output: 音频输出特征
                - output_modality: 输出模态类型
        """
        # 创建空字典存储所有特征
        features = {}
        
        # 步骤1: 处理输入文本(如果提供)
        if text is not None:
            # 使用文本处理器处理文本
            text_features = self.text_processor(text)
            # 将文本特征添加到总特征字典中
            features.update(text_features)
        
        # 步骤2: 处理输入图像(如果提供)
        if image_path is not None:
            # 使用图像处理器处理图像
            image_features = self.image_processor(image_path)
            # 将图像特征添加到总特征字典中
            features["image"] = image_features
        
        # 步骤3: 处理输入视频(如果提供)
        if video_path is not None:
            # 使用视频处理器处理视频
            video_features = self.video_processor(video_path)
            # 将视频特征添加到总特征字典中
            features["video"] = video_features
        
        # 步骤4: 处理输入音频(如果提供)
        if audio_path is not None:
            # 使用音频处理器处理音频
            audio_features = self.audio_processor(audio_path)
            # 将音频特征添加到总特征字典中
            features["audio"] = audio_features
        
        # 步骤5: 处理输出文本(如果提供)
        if text_output is not None:
            # 使用文本处理器处理输出文本
            text_output_features = self.text_processor(text_output)
            # 将文本输出特征添加到总特征字典中
            features["labels"] = text_output_features["input_ids"]
        
        # 步骤6: 处理输出图像(如果提供)
        if image_output_path is not None:
            # 使用图像处理器处理输出图像
            image_output_features = self.image_processor(image_output_path)
            # 将图像输出特征添加到总特征字典中
            features["image_output"] = image_output_features
        
        # 步骤7: 处理输出视频(如果提供)
        if video_output_path is not None:
            # 使用视频处理器处理输出视频
            video_output_features = self.video_processor(video_output_path)
            # 将视频输出特征添加到总特征字典中
            features["video_output"] = video_output_features
        
        # 步骤8: 处理输出音频(如果提供)
        if audio_output_path is not None:
            # 使用音频处理器处理输出音频
            audio_output_features = self.audio_processor(audio_output_path)
            # 将音频输出特征添加到总特征字典中
            features["audio_output"] = audio_output_features
        
        # 步骤9: 添加输出模态类型(如果提供)
        if output_modality is not None:
            features["output_modality"] = torch.tensor(output_modality, dtype=torch.long)
        
        return features
    
    def process_batch(
        self,
        texts: Optional[List[str]] = None,           # 可选的文本列表
        image_paths: Optional[List[str]] = None,     # 可选的图像路径列表
        video_paths: Optional[List[str]] = None,     # 可选的视频路径列表
        audio_paths: Optional[List[str]] = None,     # 可选的音频路径列表
        text_outputs: Optional[List[str]] = None,    # 可选的文本输出列表
        image_output_paths: Optional[List[str]] = None, # 可选的图像输出路径列表
        video_output_paths: Optional[List[str]] = None, # 可选的视频输出路径列表
        audio_output_paths: Optional[List[str]] = None, # 可选的音频输出路径列表
        output_modalities: Optional[List[int]] = None, # 输出模态类型列表
    ) -> Dict[str, Any]:
        """
        批量处理多个多模态样本
        
        可以同时处理多个文本、图像、视频和音频，或者它们的任意组合。
        每种模态都是可选的，只处理提供的模态。
        所有列表的长度应该相同，表示批次大小。
        
        Args:
            texts: 输入文本字符串列表
            image_paths: 图像文件路径列表
            video_paths: 视频文件路径列表
            audio_paths: 音频文件路径列表
            text_outputs: 文本输出列表
            image_output_paths: 图像输出文件路径列表
            video_output_paths: 视频输出文件路径列表
            audio_output_paths: 音频输出文件路径列表
            output_modalities: 输出模态类型列表 (0:文本, 1:图像, 2:视频, 3:音频)
            
        Returns:
            Dict[str, Any]: 包含处理后特征批次的字典，可能包含:
                - input_ids, attention_mask: 形状为[batch_size, ...]的文本特征
                - image: 形状为[batch_size, ...]的图像特征
                - video: 形状为[batch_size, ...]的视频特征
                - audio: 形状为[batch_size, ...]的音频特征
                - labels: 形状为[batch_size, ...]的文本输出特征
                - image_output: 形状为[batch_size, ...]的图像输出特征
                - video_output: 形状为[batch_size, ...]的视频输出特征
                - audio_output: 形状为[batch_size, ...]的音频输出特征
                - output_modality: 形状为[batch_size]的输出模态类型
        """
        # 创建空字典存储所有特征
        features = {}
        
        # 步骤1: 批量处理输入文本(如果提供)
        if texts is not None:
            # 使用文本处理器批量处理文本
            text_features = self.text_processor.process_batch(texts)
            # 将文本特征添加到总特征字典中
            features.update(text_features)
        
        # 步骤2: 批量处理输入图像(如果提供)
        if image_paths is not None:
            # 使用图像处理器批量处理图像
            image_features = self.image_processor.process_batch(image_paths)
            # 将图像特征添加到总特征字典中
            features["image"] = image_features
        
        # 步骤3: 批量处理输入视频(如果提供)
        if video_paths is not None:
            # 使用视频处理器批量处理视频
            video_features = self.video_processor.process_batch(video_paths)
            # 将视频特征添加到总特征字典中
            features["video"] = video_features
        
        # 步骤4: 批量处理输入音频(如果提供)
        if audio_paths is not None:
            # 使用音频处理器批量处理音频
            audio_features = self.audio_processor.process_batch(audio_paths)
            # 将音频特征添加到总特征字典中
            features["audio"] = audio_features
        
        # 步骤5: 批量处理输出文本(如果提供)
        if text_outputs is not None:
            # 使用文本处理器批量处理输出文本
            text_output_features = self.text_processor.process_batch(text_outputs)
            # 将文本输出特征添加到总特征字典中
            features["labels"] = text_output_features["input_ids"]
        
        # 步骤6: 批量处理输出图像(如果提供)
        if image_output_paths is not None:
            # 使用图像处理器批量处理输出图像
            image_output_features = self.image_processor.process_batch(image_output_paths)
            # 将图像输出特征添加到总特征字典中
            features["image_output"] = image_output_features
        
        # 步骤7: 批量处理输出视频(如果提供)
        if video_output_paths is not None:
            # 使用视频处理器批量处理输出视频
            video_output_features = self.video_processor.process_batch(video_output_paths)
            # 将视频输出特征添加到总特征字典中
            features["video_output"] = video_output_features
        
        # 步骤8: 批量处理输出音频(如果提供)
        if audio_output_paths is not None:
            # 使用音频处理器批量处理输出音频
            audio_output_features = self.audio_processor.process_batch(audio_output_paths)
            # 将音频输出特征添加到总特征字典中
            features["audio_output"] = audio_output_features
        
        # 步骤9: 添加输出模态类型(如果提供)
        if output_modalities is not None:
            features["output_modality"] = torch.tensor(output_modalities, dtype=torch.long)
        
        return features