"""
纯解码器架构多模态模型演示

这个脚本展示了如何使用统一token序列的纯解码器架构处理多模态数据。
演示内容包括:
1. 初始化多模态tokenizer
2. 准备多模态输入
3. 使用统一GPT模型处理
4. 生成多模态输出
"""

import torch
from PIL import Image
from transformers import GPT2Tokenizer
from models.unified_gpt import UnifiedMultimodalGPT
from models.tokenizers import MultimodalTokenizer

def main():
    # 1. 初始化组件
    print("初始化多模态组件...")
    
    # 文本tokenizer
    text_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    
    # 多模态tokenizer
    multimodal_tokenizer = MultimodalTokenizer(
        text_tokenizer=text_tokenizer,
        image_patch_size=16,
        audio_feature_dim=128
    )
    
    # 统一多模态模型
    model = UnifiedMultimodalGPT(
        text_vocab_size=text_tokenizer.vocab_size,
        image_vocab_size=256**3,  # RGB组合
        audio_vocab_size=10000,
        hidden_size=768,
        num_layers=6,  # 示例使用较小的模型
        num_heads=12
    )
    
    # 2. 准备多模态输入
    print("\n准备多模态输入...")
    
    # 示例文本
    text = "这是一只猫的照片"
    
    # 示例图像(使用随机图像作为演示)
    image = Image.new("RGB", (256, 256), color="red")
    
    # Tokenize多模态输入
    inputs = multimodal_tokenizer.tokenize_multimodal(
        text=text,
        image=image
    )
    
    print(f"生成的input_ids长度: {inputs['input_ids'].shape[1]}")
    
    # 3. 模型推理
    print("\n运行模型推理...")
    with torch.no_grad():
        outputs = model(**inputs)
    
    print(f"输出logits形状: {outputs['logits'].shape}")  # [batch, seq_len, vocab_size]
    
    # 4. 生成示例
    print("\n生成多模态输出...")
    
    # 设置生成约束(只生成文本)
    modal_constraints = {
        "text": True,
        "image": False,
        "audio": False
    }
    
    # 生成输出
    generated = model.generate(
        inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_length=50,
        modal_constraints=modal_constraints
    )
    
    # 解码生成的文本
    generated_text_ids = generated[0, inputs["input_ids"].shape[1]:] - multimodal_tokenizer.text_token_offset
    generated_text = text_tokenizer.decode(generated_text_ids)
    
    print(f"生成的文本: {generated_text}")

if __name__ == "__main__":
    print("="*50)
    print("纯解码器架构多模态模型演示")
    print("="*50)
    main()