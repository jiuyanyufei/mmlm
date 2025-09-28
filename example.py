"""
多模态大模型使用示例

这个脚本展示了如何使用我们的多模态大模型进行训练和推理。
包含以下示例:
1. 加载和准备数据
2. 训练多模态模型
3. 使用模型进行推理
4. 生成不同模态的输出

这个示例脚本可以帮助用户快速上手使用多模态大模型。
"""

import os
import torch
import argparse
from transformers import GPT2Tokenizer

# 导入自定义模块
from config import ModelConfig, TrainingConfig, DataConfig
from data.processors import MultiModalProcessor
from models.multimodal_gpt import MultiModalGPT
from utils.logger import setup_logger, logger
from utils.utils import set_seed


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="多模态大模型示例")
    parser.add_argument("--mode", type=str, default="inference", choices=["train", "inference"],
                        help="运行模式: train或inference")
    parser.add_argument("--model_path", type=str, default="./outputs/checkpoint-final",
                        help="模型路径，用于推理模式")
    parser.add_argument("--output_dir", type=str, default="./outputs",
                        help="输出目录")
    parser.add_argument("--seed", type=int, default=42,
                        help="随机种子")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="计算设备")
    return parser.parse_args()


def train_example():
    """训练示例"""
    print("\n" + "="*50)
    print("多模态大模型训练示例")
    print("="*50)
    
    # 这里只是示例，实际训练应该使用train.py脚本
    print("\n1. 准备训练数据")
    print("   - 数据应包含文本、图像、视频和音频")
    print("   - 数据格式应为JSON或CSV")
    print("   - 数据应包含输入和输出模态信息")
    
    print("\n2. 训练命令示例:")
    print("""
    python train.py \\
        --train_file data/train.json \\
        --validation_file data/val.json \\
        --output_dir ./outputs \\
        --model_name mmlm \\
        --pretrained_text_model gpt2-medium \\
        --pretrained_vision_model google/vit-base-patch16-224 \\
        --pretrained_audio_model facebook/wav2vec2-base-960h \\
        --per_device_train_batch_size 8 \\
        --per_device_eval_batch_size 8 \\
        --gradient_accumulation_steps 4 \\
        --learning_rate 5e-5 \\
        --num_train_epochs 3 \\
        --enable_multimodal_output \\
        --fp16
    """)
    
    print("\n3. 训练过程:")
    print("   - 模型将学习处理多模态输入")
    print("   - 模型将学习生成多模态输出")
    print("   - 模型将学习决定输出哪种模态")
    
    print("\n4. 训练结果:")
    print("   - 模型检查点将保存在输出目录")
    print("   - 训练日志将记录损失和评估结果")
    print("   - 最终模型可用于推理")


def inference_example(args):
    """推理示例"""
    print("\n" + "="*50)
    print("多模态大模型推理示例")
    print("="*50)
    
    # 设置随机种子
    set_seed(args.seed)
    
    # 设置设备
    device = torch.device(args.device)
    print(f"\n使用设备: {device}")
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 设置日志
    log_file = os.path.join(args.output_dir, "example.log")
    logger = setup_logger(log_file=log_file)
    
    print("\n1. 加载模型")
    # 如果没有训练好的模型，这里只是演示流程
    if not os.path.exists(args.model_path):
        print(f"   - 模型路径 {args.model_path} 不存在，使用示例模型")
        
        # 加载分词器
        print("   - 加载分词器")
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2-medium")
        special_tokens = {
            "pad_token": "<pad>",
            "bos_token": "<bos>",
            "eos_token": "<eos>",
        }
        tokenizer.add_special_tokens(special_tokens)
        
        # 创建模型配置
        print("   - 创建模型配置")
        model_config = ModelConfig(
            pretrained_text_model="gpt2-medium",
            pretrained_vision_model="google/vit-base-patch16-224",
            pretrained_audio_model="facebook/wav2vec2-base-960h",
            enable_multimodal_output=True,
        )
        
        # 创建模型
        print("   - 创建模型")
        model = MultiModalGPT(model_config)
        model.text_model.resize_token_embeddings(len(tokenizer))
        
        # 创建处理器
        print("   - 创建多模态处理器")
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
    else:
        print(f"   - 加载模型: {args.model_path}")
        # 这里应该使用实际的模型加载代码
        # 为了示例，我们只是打印步骤
        print("   - 加载分词器")
        print("   - 加载模型权重")
        print("   - 创建多模态处理器")
    
    print("\n2. 准备输入")
    print("   - 模型支持多种输入模态:")
    print("     * 文本: 直接输入文本字符串")
    print("     * 图像: 提供图像文件路径")
    print("     * 视频: 提供视频文件路径")
    print("     * 音频: 提供音频文件路径")
    
    # 示例输入
    text_input = "这是一个多模态模型，可以处理文本、图像、视频和音频。"
    print(f"\n   示例文本输入: \"{text_input}\"")
    
    print("\n3. 生成输出")
    print("   - 模型可以生成多种输出模态:")
    print("     * 文本: 生成文本回复")
    print("     * 图像: 生成图像")
    print("     * 视频: 生成视频")
    print("     * 音频: 生成音频")
    
    print("\n   - 模型会自动决定输出哪种模态")
    print("   - 也可以强制指定输出模态类型")
    
    # 示例输出
    print("\n   示例文本输出: \"这是模型生成的文本回复，展示了多模态模型的能力。\"")
    
    print("\n4. 推理命令示例:")
    print("""
    python inference.py \\
        --model_path ./outputs/checkpoint-final \\
        --text "这是一个多模态输入示例" \\
        --image ./examples/image.jpg \\
        --output_dir ./results \\
        --force_output_modality 0  # 0:文本, 1:图像, 2:视频, 3:音频
    """)


def main():
    """主函数"""
    # 解析命令行参数
    args = parse_args()
    
    # 打印标题
    print("\n" + "*"*70)
    print("*" + " "*24 + "多模态大模型示例" + " "*24 + "*")
    print("*" + " "*70 + "*")
    print("* 这个脚本展示了如何使用多模态大模型进行训练和推理" + " "*18 + "*")
    print("* 模型支持任意形态的输入和输出，可以自己学习决定输出哪种模态" + " "*8 + "*")
    print("*"*70)
    
    # 根据模式运行不同的示例
    if args.mode == "train":
        train_example()
    else:
        inference_example(args)
    
    # 打印结束信息
    print("\n" + "*"*70)
    print("* 示例结束，更多详细信息请参考文档" + " "*33 + "*")
    print("*"*70 + "\n")


if __name__ == "__main__":
    main()