"""
多模态大模型推理脚本

这个脚本用于使用训练好的多模态大模型进行推理。
主要功能包括:
1. 加载训练好的模型
2. 处理多模态输入（文本、图像、视频、音频）
3. 生成多模态输出（文本、图像、视频、音频）
4. 保存生成结果

模型可以根据输入自动决定输出哪种模态，支持文本、图像、视频和音频的生成。
"""

import os                       # 操作系统相关功能
import argparse                 # 命令行参数解析
import torch                    # PyTorch深度学习框架
import numpy as np              # 数值计算库
from PIL import Image           # 图像处理库
import matplotlib.pyplot as plt # 可视化库
from transformers import GPT2Tokenizer  # Hugging Face的GPT2分词器
from typing import Dict, List, Optional, Union, Any, Tuple  # 类型提示

# 导入自定义模块
from config import ModelConfig  # 模型配置类
from utils.logger import setup_logger, logger  # 日志工具
from utils.utils import set_seed  # 通用工具函数
from data.processors import MultiModalProcessor  # 多模态数据处理器
from models.multimodal_gpt import MultiModalGPT  # 多模态GPT模型


def parse_args():
    """
    解析命令行参数
    
    这个函数定义了推理脚本需要的所有命令行参数，包括模型路径、输入数据和输出设置等。
    
    Returns:
        argparse.Namespace: 包含所有解析后参数的命名空间对象
    """
    # 创建参数解析器
    parser = argparse.ArgumentParser(description="多模态大模型推理")
    
    # ===== 模型参数 =====
    parser.add_argument(
        "--model_path", 
        type=str, 
        required=True,
        help="训练好的模型路径"
    )
    parser.add_argument(
        "--pretrained_text_model", 
        type=str, 
        default="gpt2-medium", 
        help="预训练文本模型名称，如gpt2、gpt2-medium等"
    )
    parser.add_argument(
        "--pretrained_vision_model", 
        type=str, 
        default="google/vit-base-patch16-224", 
        help="预训练视觉模型名称，如ViT模型"
    )
    parser.add_argument(
        "--pretrained_audio_model", 
        type=str, 
        default="facebook/wav2vec2-base-960h", 
        help="预训练音频模型名称，如Wav2Vec2模型"
    )
    
    # ===== 输入参数 =====
    parser.add_argument(
        "--text", 
        type=str, 
        help="输入文本"
    )
    parser.add_argument(
        "--image", 
        type=str, 
        help="输入图像路径"
    )
    parser.add_argument(
        "--video", 
        type=str, 
        help="输入视频路径"
    )
    parser.add_argument(
        "--audio", 
        type=str, 
        help="输入音频路径"
    )
    
    # ===== 输出参数 =====
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="./outputs", 
        help="输出目录"
    )
    parser.add_argument(
        "--force_output_modality", 
        type=int, 
        default=None,
        help="强制输出模态类型 (0:文本, 1:图像, 2:视频, 3:音频)"
    )
    parser.add_argument(
        "--num_return_sequences", 
        type=int, 
        default=1,
        help="返回的序列数量"
    )
    parser.add_argument(
        "--max_length", 
        type=int, 
        default=50,
        help="生成的最大长度"
    )
    parser.add_argument(
        "--temperature", 
        type=float, 
        default=1.0,
        help="生成的温度参数"
    )
    parser.add_argument(
        "--top_p", 
        type=float, 
        default=0.9,
        help="生成的top-p参数"
    )
    
    # ===== 其他参数 =====
    parser.add_argument(
        "--seed", 
        type=int, 
        default=42, 
        help="随机种子，用于结果复现"
    )
    parser.add_argument(
        "--device", 
        type=str, 
        default="cuda" if torch.cuda.is_available() else "cpu", 
        help="计算设备"
    )
    
    # 解析参数并返回
    return parser.parse_args()


def load_model(args):
    """
    加载训练好的模型
    
    Args:
        args: 命令行参数
        
    Returns:
        tuple: (model, tokenizer, processor) - 模型、分词器和处理器
    """
    # 设置设备
    device = torch.device(args.device)
    
    # 加载分词器
    logger.info(f"加载分词器: {args.pretrained_text_model}")
    tokenizer = GPT2Tokenizer.from_pretrained(args.pretrained_text_model)
    
    # 添加特殊标记到分词器
    special_tokens = {
        "pad_token": "<pad>",  # 填充标记
        "bos_token": "<bos>",  # 序列开始标记
        "eos_token": "<eos>",  # 序列结束标记
    }
    tokenizer.add_special_tokens(special_tokens)
    
    # 创建模型配置
    model_config = ModelConfig(
        pretrained_text_model=args.pretrained_text_model,
        pretrained_vision_model=args.pretrained_vision_model,
        pretrained_audio_model=args.pretrained_audio_model,
        enable_multimodal_output=True,  # 启用多模态输出
    )
    
    # 创建模型
    logger.info(f"创建多模态GPT模型")
    model = MultiModalGPT(model_config)
    
    # 调整模型词汇表大小以匹配分词器
    model.text_model.resize_token_embeddings(len(tokenizer))
    
    # 加载训练好的模型权重
    logger.info(f"加载模型权重: {args.model_path}")
    model.load_state_dict(torch.load(os.path.join(args.model_path, "pytorch_model.bin"), map_location=device))
    
    # 将模型移动到设备
    model = model.to(device)
    model.eval()  # 设置为评估模式
    
    # 创建多模态处理器
    logger.info("创建多模态处理器")
    processor = MultiModalProcessor(
        tokenizer=tokenizer,
        image_size=224,
        num_frames=8,
        frame_size=224,
        video_sample_rate=4,
        max_text_length=512,
        audio_sample_rate=16000,
        max_audio_length=10,
    )
    
    return model, tokenizer, processor


def prepare_inputs(args, processor, device):
    """
    准备模型输入
    
    Args:
        args: 命令行参数
        processor: 多模态处理器
        device: 计算设备
        
    Returns:
        dict: 模型输入
    """
    # 处理输入
    features = processor.process_sample(
        text=args.text,
        image_path=args.image,
        video_path=args.video,
        audio_path=args.audio,
    )
    
    # 将特征移动到设备
    inputs = {}
    for k, v in features.items():
        if isinstance(v, torch.Tensor):
            inputs[k] = v.unsqueeze(0).to(device)  # 添加批次维度
    
    return inputs


def generate_output(model, inputs, args):
    """
    生成输出
    
    Args:
        model: 多模态GPT模型
        inputs: 模型输入
        args: 命令行参数
        
    Returns:
        dict: 生成的输出
    """
    # 设置生成参数
    generation_config = {
        "max_length": args.max_length,
        "num_return_sequences": args.num_return_sequences,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "do_sample": True,
    }
    
    # 如果指定了输出模态类型，添加到输入中
    if args.force_output_modality is not None:
        inputs["output_modality"] = torch.tensor([args.force_output_modality], device=inputs["input_ids"].device)
    
    # 生成输出
    with torch.no_grad():
        outputs = model(**inputs)
        
        # 如果没有指定输出模态类型，使用模型预测的模态类型
        if args.force_output_modality is None and "modality_logits" in outputs:
            modality_logits = outputs["modality_logits"]
            # 获取最可能的模态类型
            modality_type = torch.argmax(modality_logits, dim=-1)[0].item()
        else:
            modality_type = args.force_output_modality if args.force_output_modality is not None else 0
    
    # 根据模态类型返回不同的输出
    # 0: 文本, 1: 图像, 2: 视频, 3: 音频
    if modality_type == 0:
        # 文本输出
        return {
            "modality_type": "text",
            "text_logits": outputs["text_logits"],
        }
    elif modality_type == 1:
        # 图像输出
        return {
            "modality_type": "image",
            "image_output": outputs["image_output"],
        }
    elif modality_type == 2:
        # 视频输出
        return {
            "modality_type": "video",
            "video_output": outputs["video_output"],
        }
    elif modality_type == 3:
        # 音频输出
        return {
            "modality_type": "audio",
            "audio_output": outputs["audio_output"],
        }
    else:
        # 默认返回文本输出
        return {
            "modality_type": "text",
            "text_logits": outputs["text_logits"],
        }


def decode_text_output(text_logits, tokenizer):
    """
    解码文本输出
    
    Args:
        text_logits: 文本logits
        tokenizer: 分词器
        
    Returns:
        str: 解码后的文本
    """
    # 获取最可能的token ID
    token_ids = torch.argmax(text_logits, dim=-1)[0].cpu().numpy()
    
    # 解码token ID
    text = tokenizer.decode(token_ids, skip_special_tokens=True)
    
    return text


def save_image_output(image_output, output_path):
    """
    保存图像输出
    
    Args:
        image_output: 图像输出张量
        output_path: 输出路径
    """
    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # 将张量转换为PIL图像
    image = image_output[0].permute(1, 2, 0).cpu().numpy()
    
    # 归一化到[0, 1]范围
    image = (image + 1) / 2.0
    image = np.clip(image, 0, 1)
    
    # 保存图像
    plt.imsave(output_path, image)


def save_video_output(video_output, output_path):
    """
    保存视频输出
    
    Args:
        video_output: 视频输出张量
        output_path: 输出路径
    """
    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # 创建帧目录
    frames_dir = output_path.replace(".mp4", "_frames")
    os.makedirs(frames_dir, exist_ok=True)
    
    # 保存每一帧
    video = video_output[0].cpu().numpy()
    for i, frame in enumerate(video):
        # 将帧转换为PIL图像
        frame = frame.transpose(1, 2, 0)
        
        # 归一化到[0, 1]范围
        frame = (frame + 1) / 2.0
        frame = np.clip(frame, 0, 1)
        
        # 保存帧
        plt.imsave(os.path.join(frames_dir, f"frame_{i:03d}.png"), frame)
    
    # 使用ffmpeg将帧合成为视频
    try:
        import subprocess
        cmd = f"ffmpeg -y -framerate 24 -i {frames_dir}/frame_%03d.png -c:v libx264 -pix_fmt yuv420p {output_path}"
        subprocess.call(cmd, shell=True)
        logger.info(f"视频已保存到: {output_path}")
    except Exception as e:
        logger.error(f"合成视频失败: {str(e)}")
        logger.info(f"帧已保存到: {frames_dir}")


def save_audio_output(audio_output, output_path, sample_rate=16000):
    """
    保存音频输出
    
    Args:
        audio_output: 音频输出张量
        output_path: 输出路径
        sample_rate: 采样率
    """
    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # 将张量转换为numpy数组
    audio = audio_output[0, 0].cpu().numpy()
    
    # 归一化到[-1, 1]范围
    if np.abs(audio).max() > 1.0:
        audio = audio / np.abs(audio).max()
    
    # 保存音频
    try:
        import soundfile as sf
        sf.write(output_path, audio, sample_rate)
        logger.info(f"音频已保存到: {output_path}")
    except Exception as e:
        logger.error(f"保存音频失败: {str(e)}")
        # 保存为numpy数组
        np.save(output_path.replace(".wav", ".npy"), audio)
        logger.info(f"音频数组已保存到: {output_path.replace('.wav', '.npy')}")


def main():
    """
    主函数 - 推理流程的入口点
    """
    # 解析命令行参数
    args = parse_args()
    
    # 设置随机种子
    set_seed(args.seed)
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 设置日志
    log_file = os.path.join(args.output_dir, "inference.log")
    logger = setup_logger(log_file=log_file)
    
    # 加载模型
    model, tokenizer, processor = load_model(args)
    
    # 准备输入
    inputs = prepare_inputs(args, processor, args.device)
    
    # 生成输出
    outputs = generate_output(model, inputs, args)
    
    # 处理输出
    modality_type = outputs["modality_type"]
    logger.info(f"生成的输出模态类型: {modality_type}")
    
    if modality_type == "text":
        # 解码文本输出
        text = decode_text_output(outputs["text_logits"], tokenizer)
        logger.info(f"生成的文本: {text}")
        
        # 保存文本
        text_path = os.path.join(args.output_dir, "output.txt")
        with open(text_path, "w", encoding="utf-8") as f:
            f.write(text)
        logger.info(f"文本已保存到: {text_path}")
    
    elif modality_type == "image":
        # 保存图像输出
        image_path = os.path.join(args.output_dir, "output.png")
        save_image_output(outputs["image_output"], image_path)
        logger.info(f"图像已保存到: {image_path}")
    
    elif modality_type == "video":
        # 保存视频输出
        video_path = os.path.join(args.output_dir, "output.mp4")
        save_video_output(outputs["video_output"], video_path)
        logger.info(f"视频已保存到: {video_path}")
    
    elif modality_type == "audio":
        # 保存音频输出
        audio_path = os.path.join(args.output_dir, "output.wav")
        save_audio_output(outputs["audio_output"], audio_path)
        logger.info(f"音频已保存到: {audio_path}")
    
    logger.info("推理完成！")


if __name__ == "__main__":
    """
    脚本入口点
    """
    main()