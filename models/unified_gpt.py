"""
统一多模态GPT模型 (纯解码器架构)

这个实现采用类似GPT的纯解码器架构处理多模态数据，所有模态统一表示为token序列。
关键特点：
1. 所有模态(文本/图像/视频/音频)都转换为token序列
2. 使用特殊token标识不同模态
3. 单一Transformer解码器处理所有模态
4. 自回归生成任意模态输出
"""

import torch
import torch.nn as nn
from transformers import GPT2Model, GPT2Config
from typing import Dict, List, Optional

class UnifiedMultimodalGPT(nn.Module):
    """
    统一多模态GPT模型
    
    采用纯解码器架构处理多模态输入和生成多模态输出。
    所有模态都转换为token序列，通过特殊token区分不同模态类型。
    """
    
    def __init__(
        self,
        text_vocab_size: int = 50257,       # 文本词汇表大小(GPT-2默认)
        image_vocab_size: int = 8192,       # 图像patch词汇表大小
        audio_vocab_size: int = 1024,       # 音频特征词汇表大小
        hidden_size: int = 768,             # 隐藏层维度
        num_layers: int = 12,              # Transformer层数
        num_heads: int = 12,               # 注意力头数
        max_seq_len: int = 1024,           # 最大序列长度
        dropout: float = 0.1               # Dropout概率
    ):
        """
        初始化统一多模态GPT模型
        
        Args:
            text_vocab_size: 文本词汇表大小
            image_vocab_size: 图像patch词汇表大小
            audio_vocab_size: 音频特征词汇表大小
            hidden_size: 隐藏层维度
            num_layers: Transformer层数
            num_heads: 注意力头数
            max_seq_len: 最大序列长度
            dropout: Dropout概率
        """
        super().__init__()
        
        # 1. 词汇表配置
        self.text_vocab_size = text_vocab_size
        self.image_vocab_size = image_vocab_size
        self.audio_vocab_size = audio_vocab_size
        
        # 特殊token ID分配
        self.modal_special_tokens = {
            "[TEXT]": 0,
            "[IMAGE]": 1, 
            "[VIDEO]": 2,
            "[AUDIO]": 3,
            "[SEP_MODAL]": 4
        }
        
        # 2. 总词汇表大小 = 文本 + 图像 + 音频 + 特殊token
        self.total_vocab_size = (
            text_vocab_size + 
            image_vocab_size + 
            audio_vocab_size + 
            len(self.modal_special_tokens)
        )
        
        # 3. 共享的token嵌入层
        self.token_embedding = nn.Embedding(
            self.total_vocab_size, hidden_size
        )
        
        # 4. 位置编码
        self.position_embedding = nn.Embedding(max_seq_len, hidden_size)
        
        # 5. Transformer解码器
        config = GPT2Config(
            n_embd=hidden_size,
            n_layer=num_layers,
            n_head=num_heads,
            vocab_size=self.total_vocab_size,
            n_positions=max_seq_len,
            resid_pdrop=dropout,
            embd_pdrop=dropout,
            attn_pdrop=dropout
        )
        self.transformer = GPT2Model(config)
        
        # 6. 输出层
        self.lm_head = nn.Linear(hidden_size, self.total_vocab_size, bias=False)
        
        # 7. 初始化权重
        self.init_weights()
    
    def init_weights(self):
        """初始化模型权重"""
        # 特殊token嵌入初始化
        nn.init.normal_(
            self.token_embedding.weight[
                :len(self.modal_special_tokens)
            ],
            mean=0.0,
            std=0.02
        )
        
        # 绑定输入输出嵌入权重(类似GPT-2)
        self.lm_head.weight = self.token_embedding.weight
    
    def get_modal_ranges(self):
        """
        获取各模态token的范围
        
        Returns:
            dict: 各模态token的起始和结束ID
        """
        return {
            "text": (len(self.modal_special_tokens), 
                    len(self.modal_special_tokens) + self.text_vocab_size),
            "image": (len(self.modal_special_tokens) + self.text_vocab_size,
                     len(self.modal_special_tokens) + self.text_vocab_size + self.image_vocab_size),
            "audio": (len(self.modal_special_tokens) + self.text_vocab_size + self.image_vocab_size,
                     len(self.modal_special_tokens) + self.text_vocab_size + self.image_vocab_size + self.audio_vocab_size)
        }
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        modal_types: Optional[List[str]] = None
    ) -> Dict[str, torch.Tensor]:
        """
        前向传播
        
        Args:
            input_ids: 输入token ID序列 [batch_size, seq_len]
            attention_mask: 注意力掩码 [batch_size, seq_len]
            modal_types: 每个token对应的模态类型列表
            
        Returns:
            Dict[str, torch.Tensor]: 包含logits的输出字典
        """
        # 1. 获取输入嵌入
        embeddings = self.token_embedding(input_ids)
        
        # 2. 添加位置编码
        position_ids = torch.arange(
            input_ids.size(1), 
            dtype=torch.long, 
            device=input_ids.device
        ).unsqueeze(0)
        position_embeddings = self.position_embedding(position_ids)
        embeddings += position_embeddings
        
        # 3. 通过Transformer
        transformer_output = self.transformer(
            inputs_embeds=embeddings,
            attention_mask=attention_mask,
            return_dict=True
        )
        hidden_states = transformer_output.last_hidden_state
        
        # 4. 计算logits
        logits = self.lm_head(hidden_states)
        
        return {
            "logits": logits,
            "hidden_states": hidden_states
        }
    
    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        max_length: int = 100,
        temperature: float = 1.0,
        top_k: int = 50,
        modal_constraints: Optional[Dict[str, bool]] = None
    ) -> torch.Tensor:
        """
        生成多模态序列
        
        Args:
            input_ids: 输入token ID序列 [batch_size, seq_len]
            attention_mask: 注意力掩码 [batch_size, seq_len]
            max_length: 最大生成长度
            temperature: 温度参数控制生成多样性
            top_k: top-k采样参数
            modal_constraints: 模态约束，如{"text": True, "image": False}
            
        Returns:
            torch.Tensor: 生成的token ID序列 [batch_size, generated_seq_len]
        """
        # 获取各模态token范围
        modal_ranges = self.get_modal_ranges()
        
        for _ in range(max_length):
            # 1. 获取当前输出logits
            outputs = self(input_ids, attention_mask)
            next_token_logits = outputs["logits"][:, -1, :]
            
            # 2. 应用模态约束
            if modal_constraints is not None:
                for modal, enabled in modal_constraints.items():
                    if not enabled:
                        start, end = modal_ranges[modal]
                        next_token_logits[:, start:end] = -float("inf")
            
            # 3. 温度调节和top-k采样
            next_token_logits = next_token_logits / temperature
            probs = torch.softmax(next_token_logits, dim=-1)
            
            # 4. 采样下一个token
            next_tokens = torch.multinomial(probs, num_samples=1)
            
            # 5. 更新输入序列
            input_ids = torch.cat([input_ids, next_tokens], dim=-1)
            
            # 6. 更新注意力掩码
            if attention_mask is not None:
                attention_mask = torch.cat([
                    attention_mask,
                    torch.ones((attention_mask.size(0), 1), device=attention_mask.device)
                ], dim=-1)
        
        return input_ids