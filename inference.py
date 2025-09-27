"""
多模态大模型推理脚本
"""
import os
import argparse
import torch
from PIL import Image
import cv2
from transformers import GPT2Tokenizer
from typing import Dict, List, Optional, Union, Any, Tuple

from config import InferenceConfig
from utils.logger import setup_logger, logger
from utils.utils import set_seed
from data.processors import ImageProcessor, VideoProcessor, TextProcessor
from models.multimodal_gpt import MultiModalGPT
from models.generation import MultiModalGPTGeneration


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="多模态大模型推理")
    
    # 模型参数
    parser.add_argument("--checkpoint_path", type=str, required=True, help="模型检查点路径")
    
    # 输入参数
    parser.add_argument("--text", type=str, help="输入文本")
    parser.add_argument("--image", type=str, help="输入图像路径")
    parser.add_argument("--video", type=str, help="输入视频路径")
    
    # 生成参数
    parser.add_argument("--max_new_tokens", type=int, default=100, help="最大新标记数")
    parser.add_argument("--temperature", type=float, default=0.7, help="温度")
    parser.add_argument("--top_p", type=float, default=0.9, help="累积概率阈值")
    parser.add_argument("--top_k", type=int, default=50, help="保留的最高概率标记数")
    parser.add_argument("--repetition_penalty", type=float, default=1.2, help="重复惩罚")
    parser.add_argument("--do_sample", action="store_true", help="是否采样")
    
    # 其他参数
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--device", type=str, default="cuda", help="设备")
    
    return parser.parse_args()


def load_model(checkpoint_path: str, device: str = "cuda") -> Tuple[MultiModalGPTGeneration, GPT2Tokenizer]:
    """
    加载模型
    
    Args:
        checkpoint_path: 模型检查点路径
        device: 设备
        
    Returns:
        Tuple[MultiModalGPTGeneration, GPT2Tokenizer]: 模型和分词器
    """
    # 加载分词器
    tokenizer = GPT2Tokenizer.from_pretrained(checkpoint_path)
    
    # 加载模型
    model = MultiModalGPTGeneration.from_pretrained(checkpoint_path)
    model.to(device)
    model.eval()
    
    return model, tokenizer


def process_inputs(
    text: Optional[str] = None,
    image_path: Optional[str] = None,
    video_path: Optional[str] = None,
    tokenizer: GPT2Tokenizer = None,
    device: str = "cuda",
) -> Dict[str, torch.Tensor]:
    """
    处理输入
    
    Args:
        text: 输入文本
        image_path: 输入图像路径
        video_path: 输入视频路径
        tokenizer: 分词器
        device: 设备
        
    Returns:
        Dict[str, torch.Tensor]: 处理后的输入
    """
    inputs = {}
    
    # 处理文本
    if text is not None and tokenizer is not None:
        text_inputs = tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        inputs["input_ids"] = text_inputs["input_ids"].to(device)
        inputs["attention_mask"] = text_inputs["attention_mask"].to(device)
    
    # 处理图像
    if image_path is not None:
        image_processor = ImageProcessor(image_size=224)
        image = image_processor(image_path)
        inputs["image"] = image.unsqueeze(0).to(device)  # 添加批次维度
    
    # 处理视频
    if video_path is not None:
        video_processor = VideoProcessor(num_frames=8, frame_size=224)
        video = video_processor(video_path)
        inputs["video"] = video.unsqueeze(0).to(device)  # 添加批次维度
    
    return inputs


def generate_response(
    model: MultiModalGPTGeneration,
    tokenizer: GPT2Tokenizer,
    inputs: Dict[str, torch.Tensor],
    max_new_tokens: int = 100,
    temperature: float = 0.7,
    top_p: float = 0.9,
    top_k: int = 50,
    repetition_penalty: float = 1.2,
    do_sample: bool = True,
) -> str:
    """
    生成响应
    
    Args:
        model: 模型
        tokenizer: 分词器
        inputs: 输入
        max_new_tokens: 最大新标记数
        temperature: 温度
        top_p: 累积概率阈值
        top_k: 保留的最高概率标记数
        repetition_penalty: 重复惩罚
        do_sample: 是否采样
        
    Returns:
        str: 生成的响应
    """
    # 生成文本
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            do_sample=do_sample,
        )
    
    # 解码生成的文本
    if "input_ids" in inputs:
        # 只保留新生成的标记
        input_length = inputs["input_ids"].shape[1]
        generated_ids = output_ids[:, input_length:]
    else:
        generated_ids = output_ids
    
    generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    
    return generated_text


def main():
    """主函数"""
    # 解析参数
    args = parse_args()
    
    # 设置随机种子
    set_seed(args.seed)
    
    # 设置日志
    logger = setup_logger()
    
    # 设置设备
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    logger.info(f"使用设备: {device}")
    
    # 加载模型
    logger.info(f"从 {args.checkpoint_path} 加载模型...")
    model, tokenizer = load_model(args.checkpoint_path, device)
    
    # 处理输入
    logger.info("处理输入...")
    inputs = process_inputs(
        text=args.text,
        image_path=args.image,
        video_path=args.video,
        tokenizer=tokenizer,
        device=device,
    )
    
    # 生成响应
    logger.info("生成响应...")
    response = generate_response(
        model=model,
        tokenizer=tokenizer,
        inputs=inputs,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        repetition_penalty=args.repetition_penalty,
        do_sample=args.do_sample,
    )
    
    # 打印响应
    print("\n生成的响应:")
    print("-" * 50)
    print(response)
    print("-" * 50)


if __name__ == "__main__":
    main()