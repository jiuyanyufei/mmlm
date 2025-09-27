"""
生成模块
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Union, Any, Tuple
from transformers import LogitsProcessorList, StoppingCriteriaList, StoppingCriteria
from transformers.generation import GenerationConfig

from models.multimodal_gpt import MultiModalGPT


class MultiModalGPTGeneration(MultiModalGPT):
    """多模态GPT生成模型"""
    
    def prepare_inputs_for_generation(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        image: Optional[torch.Tensor] = None,
        video: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        为生成准备输入
        
        Args:
            input_ids: 输入ID [batch_size, seq_len]
            attention_mask: 注意力掩码 [batch_size, seq_len]
            image: 图像特征 [batch_size, channels, height, width]
            video: 视频特征 [batch_size, num_frames, channels, height, width]
            
        Returns:
            Dict[str, Any]: 模型输入
        """
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "image": image,
            "video": video,
        }
    
    def generate(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        image: Optional[torch.Tensor] = None,
        video: Optional[torch.Tensor] = None,
        max_new_tokens: int = 100,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        repetition_penalty: float = 1.2,
        do_sample: bool = True,
        **kwargs,
    ) -> torch.Tensor:
        """
        生成文本
        
        Args:
            input_ids: 输入ID [batch_size, seq_len]
            attention_mask: 注意力掩码 [batch_size, seq_len]
            image: 图像特征 [batch_size, channels, height, width]
            video: 视频特征 [batch_size, num_frames, channels, height, width]
            max_new_tokens: 最大新标记数
            temperature: 温度
            top_p: 累积概率阈值
            top_k: 保留的最高概率标记数
            repetition_penalty: 重复惩罚
            do_sample: 是否采样
            
        Returns:
            torch.Tensor: 生成的文本ID [batch_size, seq_len]
        """
        batch_size = input_ids.shape[0] if input_ids is not None else (
            image.shape[0] if image is not None else video.shape[0]
        )
        device = next(self.parameters()).device
        
        # 如果没有输入ID，创建一个起始标记
        if input_ids is None:
            input_ids = torch.full(
                (batch_size, 1),
                self.text_model.config.bos_token_id,
                dtype=torch.long,
                device=device,
            )
        
        # 如果没有注意力掩码，创建一个全1掩码
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        
        # 创建生成配置
        generation_config = GenerationConfig(
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            do_sample=do_sample,
            pad_token_id=self.text_model.config.pad_token_id,
            bos_token_id=self.text_model.config.bos_token_id,
            eos_token_id=self.text_model.config.eos_token_id,
        )
        
        # 自回归生成
        for _ in range(max_new_tokens):
            # 准备模型输入
            model_inputs = self.prepare_inputs_for_generation(
                input_ids=input_ids,
                attention_mask=attention_mask,
                image=image,
                video=video,
            )
            
            # 前向传播
            outputs = self.forward(**model_inputs)
            logits = outputs["logits"]
            
            # 只使用最后一个时间步的logits
            next_token_logits = logits[:, -1, :]
            
            # 应用温度
            if temperature > 0:
                next_token_logits = next_token_logits / temperature
            
            # 应用重复惩罚
            if repetition_penalty > 1.0:
                for i in range(batch_size):
                    for token_id in set(input_ids[i].tolist()):
                        if token_id < next_token_logits.shape[-1]:
                            next_token_logits[i, token_id] /= repetition_penalty
            
            # 应用top_k和top_p过滤
            if do_sample:
                # top_k过滤
                if top_k > 0:
                    indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                    next_token_logits[indices_to_remove] = -float("inf")
                
                # top_p过滤
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    
                    # 移除累积概率大于top_p的标记
                    sorted_indices_to_remove = cumulative_probs > top_p
                    # 保留第一个超过阈值的标记
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    
                    # 散回原始索引
                    indices_to_remove = sorted_indices_to_remove.scatter(
                        dim=1, index=sorted_indices, src=sorted_indices_to_remove
                    )
                    next_token_logits[indices_to_remove] = -float("inf")
                
                # 采样
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                # 贪婪解码
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            
            # 更新输入ID和注意力掩码
            input_ids = torch.cat([input_ids, next_token], dim=-1)
            attention_mask = torch.cat(
                [attention_mask, attention_mask.new_ones((batch_size, 1))], dim=-1
            )
            
            # 检查是否生成了结束标记
            if (next_token == self.text_model.config.eos_token_id).all():
                break
        
        return input_ids


def greedy_search(
    model: MultiModalGPT,
    input_ids: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    image: Optional[torch.Tensor] = None,
    video: Optional[torch.Tensor] = None,
    max_new_tokens: int = 100,
    **kwargs,
) -> torch.Tensor:
    """
    贪婪搜索生成
    
    Args:
        model: 多模态GPT模型
        input_ids: 输入ID [batch_size, seq_len]
        attention_mask: 注意力掩码 [batch_size, seq_len]
        image: 图像特征 [batch_size, channels, height, width]
        video: 视频特征 [batch_size, num_frames, channels, height, width]
        max_new_tokens: 最大新标记数
        
    Returns:
        torch.Tensor: 生成的文本ID [batch_size, seq_len]
    """
    return model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        image=image,
        video=video,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        **kwargs,
    )


def sample(
    model: MultiModalGPT,
    input_ids: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    image: Optional[torch.Tensor] = None,
    video: Optional[torch.Tensor] = None,
    max_new_tokens: int = 100,
    temperature: float = 0.7,
    top_p: float = 0.9,
    top_k: int = 50,
    repetition_penalty: float = 1.2,
    **kwargs,
) -> torch.Tensor:
    """
    采样生成
    
    Args:
        model: 多模态GPT模型
        input_ids: 输入ID [batch_size, seq_len]
        attention_mask: 注意力掩码 [batch_size, seq_len]
        image: 图像特征 [batch_size, channels, height, width]
        video: 视频特征 [batch_size, num_frames, channels, height, width]
        max_new_tokens: 最大新标记数
        temperature: 温度
        top_p: 累积概率阈值
        top_k: 保留的最高概率标记数
        repetition_penalty: 重复惩罚
        
    Returns:
        torch.Tensor: 生成的文本ID [batch_size, seq_len]
    """
    return model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        image=image,
        video=video,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        repetition_penalty=repetition_penalty,
        do_sample=True,
        **kwargs,
    )