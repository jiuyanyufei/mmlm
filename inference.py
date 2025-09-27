"""
多模态大模型推理脚本

这个脚本用于使用训练好的多模态大模型进行推理。
支持三种输入模式:
1. 纯文本输入
2. 图像+文本输入
3. 视频+文本输入

主要功能:
- 加载预训练的多模态模型
- 处理不同类型的输入(文本/图像/视频)
- 生成文本响应
- 支持多种生成策略(贪婪搜索/采样)
"""
import os                       # 操作系统相关功能
import argparse                 # 命令行参数解析
import torch                    # PyTorch深度学习框架
from PIL import Image           # 图像处理库
import cv2                      # OpenCV视频处理库
from transformers import GPT2Tokenizer  # Hugging Face的GPT2分词器
from typing import Dict, List, Optional, Union, Any, Tuple  # 类型提示

# 导入自定义模块
from config import InferenceConfig            # 推理配置
from utils.logger import setup_logger, logger # 日志工具
from utils.utils import set_seed              # 随机种子设置
from data.processors import ImageProcessor, VideoProcessor, TextProcessor  # 数据处理器
from models.multimodal_gpt import MultiModalGPT  # 多模态GPT模型
from models.generation import MultiModalGPTGeneration  # 生成模型


def parse_args():
    """
    解析命令行参数
    
    这个函数定义了推理脚本需要的所有命令行参数，包括模型路径、输入数据和生成参数等。
    使用argparse库来处理命令行输入，使脚本更加灵活和可配置。
    
    Returns:
        argparse.Namespace: 包含所有解析后参数的命名空间对象
    """
    # 创建参数解析器
    parser = argparse.ArgumentParser(description="多模态大模型推理")
    
    # ===== 模型参数 =====
    # 定义模型加载相关的参数
    parser.add_argument(
        "--checkpoint_path", 
        type=str, 
        required=True,  # 必须提供模型路径
        help="模型检查点路径，包含模型权重和配置"
    )
    
    # ===== 输入参数 =====
    # 这些参数定义了模型的输入
    # 注意：这三个参数都是可选的，但至少需要提供一个
    parser.add_argument(
        "--text", 
        type=str, 
        help="输入文本，如问题或指令"
    )
    parser.add_argument(
        "--image", 
        type=str, 
        help="输入图像的文件路径，支持jpg、png等格式"
    )
    parser.add_argument(
        "--video", 
        type=str, 
        help="输入视频的文件路径，支持mp4、avi等格式"
    )
    
    # ===== 生成参数 =====
    # 这些参数控制文本生成的行为
    parser.add_argument(
        "--max_new_tokens", 
        type=int, 
        default=100, 
        help="生成的最大新标记(token)数量"
    )
    parser.add_argument(
        "--temperature", 
        type=float, 
        default=0.7, 
        help="采样温度，控制生成的随机性，较高的值(如1.0)使输出更随机，较低的值(如0.1)使输出更确定"
    )
    parser.add_argument(
        "--top_p", 
        type=float, 
        default=0.9, 
        help="核采样(nucleus sampling)的累积概率阈值，只考虑概率总和达到此值的最高概率标记"
    )
    parser.add_argument(
        "--top_k", 
        type=int, 
        default=50, 
        help="只考虑概率最高的k个标记"
    )
    parser.add_argument(
        "--repetition_penalty", 
        type=float, 
        default=1.2, 
        help="重复惩罚因子，大于1的值会降低模型重复同一文本的可能性"
    )
    parser.add_argument(
        "--do_sample", 
        action="store_true",  # 布尔标志，存在则为True
        help="是否使用采样策略，如果不设置则使用贪婪解码"
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
        default="cuda", 
        help="计算设备，可以是'cuda'(GPU)或'cpu'"
    )
    
    # 解析参数并返回
    return parser.parse_args()


def load_model(checkpoint_path: str, device: str = "cuda") -> Tuple[MultiModalGPTGeneration, GPT2Tokenizer]:
    """
    加载预训练的多模态模型和分词器
    
    这个函数从指定的检查点路径加载模型和分词器，并将模型移动到指定设备上。
    同时将模型设置为评估模式，禁用dropout等训练特性。
    
    Args:
        checkpoint_path: 保存模型和分词器的目录路径
        device: 计算设备，可以是'cuda'(GPU)或'cpu'
        
    Returns:
        Tuple[MultiModalGPTGeneration, GPT2Tokenizer]: 
            - 加载好的模型实例
            - 对应的分词器实例
    """
    # 步骤1: 加载分词器
    # 分词器负责将文本转换为模型可以理解的标记(tokens)
    tokenizer = GPT2Tokenizer.from_pretrained(checkpoint_path)
    
    # 步骤2: 加载模型
    # 从保存的检查点加载模型权重和配置
    model = MultiModalGPTGeneration.from_pretrained(checkpoint_path)
    
    # 步骤3: 将模型移动到指定设备(GPU或CPU)
    model.to(device)
    
    # 步骤4: 设置模型为评估模式
    # 这会禁用dropout等只在训练时使用的功能
    model.eval()
    
    return model, tokenizer


def process_inputs(
    text: Optional[str] = None,           # 可选的文本输入
    image_path: Optional[str] = None,     # 可选的图像路径
    video_path: Optional[str] = None,     # 可选的视频路径
    tokenizer: GPT2Tokenizer = None,      # 文本分词器
    device: str = "cuda",                 # 计算设备
) -> Dict[str, torch.Tensor]:
    """
    处理多模态输入数据
    
    这个函数处理文本、图像和视频输入，将它们转换为模型可以处理的张量格式。
    每种模态都是可选的，函数会处理所有提供的输入类型。
    
    Args:
        text: 输入文本字符串
        image_path: 输入图像的文件路径
        video_path: 输入视频的文件路径
        tokenizer: 用于处理文本的分词器
        device: 计算设备，用于将张量移动到正确的设备上
        
    Returns:
        Dict[str, torch.Tensor]: 包含处理后输入的字典，可能包含:
            - input_ids: 文本标记ID
            - attention_mask: 注意力掩码
            - image: 处理后的图像张量
            - video: 处理后的视频张量
    """
    # 创建空字典存储所有处理后的输入
    inputs = {}
    
    # ===== 步骤1: 处理文本输入 =====
    if text is not None and tokenizer is not None:
        # 使用分词器将文本转换为标记ID
        text_inputs = tokenizer(
            text,                  # 输入文本
            return_tensors="pt",   # 返回PyTorch张量
            padding=True,          # 填充到最大长度
            truncation=True,       # 截断过长的文本
        )
        
        # 将文本特征移动到指定设备
        inputs["input_ids"] = text_inputs["input_ids"].to(device)           # 标记ID
        inputs["attention_mask"] = text_inputs["attention_mask"].to(device) # 注意力掩码
    
    # ===== 步骤2: 处理图像输入 =====
    if image_path is not None:
        # 创建图像处理器
        image_processor = ImageProcessor(image_size=224)  # 设置图像大小为224x224
        
        # 处理图像
        image = image_processor(image_path)  # 返回归一化的图像张量
        
        # 添加批次维度并移动到指定设备
        # unsqueeze(0)将形状从[C,H,W]变为[1,C,H,W]
        inputs["image"] = image.unsqueeze(0).to(device)
    
    # ===== 步骤3: 处理视频输入 =====
    if video_path is not None:
        # 创建视频处理器
        video_processor = VideoProcessor(
            num_frames=8,       # 采样8帧
            frame_size=224      # 每帧大小为224x224
        )
        
        # 处理视频
        video = video_processor(video_path)  # 返回形状为[F,C,H,W]的张量
        
        # 添加批次维度并移动到指定设备
        # unsqueeze(0)将形状从[F,C,H,W]变为[1,F,C,H,W]
        inputs["video"] = video.unsqueeze(0).to(device)
    
    return inputs


def generate_response(
    model: MultiModalGPTGeneration,       # 多模态生成模型
    tokenizer: GPT2Tokenizer,             # 分词器
    inputs: Dict[str, torch.Tensor],      # 模型输入
    max_new_tokens: int = 100,            # 最大生成标记数
    temperature: float = 0.7,             # 温度参数
    top_p: float = 0.9,                   # 核采样参数
    top_k: int = 50,                      # Top-K采样参数
    repetition_penalty: float = 1.2,      # 重复惩罚参数
    do_sample: bool = True,               # 是否使用采样
) -> str:
    """
    使用多模态模型生成文本响应
    
    这个函数接收处理好的输入，使用模型生成文本响应，并将生成的标记ID解码为可读文本。
    支持多种生成策略，包括贪婪搜索和各种采样方法。
    
    Args:
        model: 多模态生成模型实例
        tokenizer: 用于解码生成标记的分词器
        inputs: 包含处理好的输入的字典(文本/图像/视频)
        max_new_tokens: 最大生成的新标记数量
        temperature: 采样温度，控制随机性(较高=更随机)
        top_p: 核采样的累积概率阈值
        top_k: 只考虑概率最高的k个标记
        repetition_penalty: 重复惩罚因子，降低重复的可能性
        do_sample: 是否使用采样(True)或贪婪搜索(False)
        
    Returns:
        str: 生成的文本响应
    """
    # ===== 步骤1: 使用模型生成文本 =====
    # 使用torch.no_grad()禁用梯度计算，节省内存并加速推理
    with torch.no_grad():
        # 调用模型的生成方法
        output_ids = model.generate(
            **inputs,                               # 输入特征
            max_new_tokens=max_new_tokens,          # 最大生成标记数
            temperature=temperature,                # 温度参数
            top_p=top_p,                            # 核采样参数
            top_k=top_k,                            # Top-K采样参数
            repetition_penalty=repetition_penalty,  # 重复惩罚
            do_sample=do_sample,                    # 是否采样
        )
    
    # ===== 步骤2: 处理生成的标记ID =====
    if "input_ids" in inputs:
        # 如果有文本输入，只保留新生成的部分
        input_length = inputs["input_ids"].shape[1]  # 输入文本长度
        generated_ids = output_ids[:, input_length:]  # 切片，只保留新生成的部分
    else:
        # 如果没有文本输入，保留所有生成的标记
        generated_ids = output_ids
    
    # ===== 步骤3: 将标记ID解码为文本 =====
    # 取第一个样本(批次中的第一个)，并跳过特殊标记
    generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    
    return generated_text


def main():
    """
    主函数 - 推理流程的入口点
    
    这个函数实现了完整的推理流程:
    1. 初始化环境(参数解析、随机种子、日志、设备)
    2. 加载预训练模型和分词器
    3. 处理输入数据(文本/图像/视频)
    4. 使用模型生成响应
    5. 输出结果
    """
    # ===== 第1步: 初始化环境 =====
    
    # 解析命令行参数
    args = parse_args()
    
    # 设置随机种子以确保结果可复现
    set_seed(args.seed)
    
    # 设置日志记录器
    logger = setup_logger()
    
    # 检测并设置计算设备(GPU或CPU)
    # 如果指定了GPU但不可用，则回退到CPU
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    logger.info(f"使用设备: {device}")
    
    # ===== 第2步: 加载模型 =====
    
    # 从检查点加载模型和分词器
    logger.info(f"从 {args.checkpoint_path} 加载模型...")
    model, tokenizer = load_model(args.checkpoint_path, device)
    
    # ===== 第3步: 处理输入 =====
    
    # 检查是否提供了至少一种输入
    if args.text is None and args.image is None and args.video is None:
        logger.warning("未提供任何输入！请提供文本、图像或视频输入。")
        return
    
    # 处理所有提供的输入
    logger.info("处理输入...")
    if args.text:
        logger.info(f"文本输入: {args.text[:50]}..." if len(args.text) > 50 else f"文本输入: {args.text}")
    if args.image:
        logger.info(f"图像输入: {args.image}")
    if args.video:
        logger.info(f"视频输入: {args.video}")
    
    # 将输入转换为模型可处理的格式
    inputs = process_inputs(
        text=args.text,             # 文本输入
        image_path=args.image,      # 图像路径
        video_path=args.video,      # 视频路径
        tokenizer=tokenizer,        # 分词器
        device=device,              # 计算设备
    )
    
    # ===== 第4步: 生成响应 =====
    
    # 使用模型生成文本响应
    logger.info("生成响应...")
    response = generate_response(
        model=model,                          # 模型
        tokenizer=tokenizer,                  # 分词器
        inputs=inputs,                        # 处理后的输入
        max_new_tokens=args.max_new_tokens,   # 最大生成标记数
        temperature=args.temperature,         # 温度参数
        top_p=args.top_p,                     # 核采样参数
        top_k=args.top_k,                     # Top-K采样参数
        repetition_penalty=args.repetition_penalty,  # 重复惩罚
        do_sample=args.do_sample,             # 是否采样
    )
    
    # ===== 第5步: 输出结果 =====
    
    # 打印生成的响应
    print("\n生成的响应:")
    print("-" * 50)
    print(response)
    print("-" * 50)
    
    # 记录完成信息
    logger.info("推理完成！")


if __name__ == "__main__":
    """
    脚本入口点
    
    当脚本直接运行(而不是被导入)时，执行main()函数
    这是Python脚本的标准模式
    """
    main()