"""
多模态大模型训练脚本

这个脚本用于训练一个能够处理文本、图像、视频和音频的多模态大模型。
主要功能包括:
1. 加载和预处理多模态数据
2. 构建多模态GPT模型
3. 训练模型并支持任意形态的输入和输出
4. 记录训练过程和评估结果

模型可以自己学习决定输出哪种模态，支持文本、图像、视频和音频的生成。
"""
import os                       # 操作系统相关功能
import argparse                 # 命令行参数解析
import torch                    # PyTorch深度学习框架
from transformers import GPT2Tokenizer  # Hugging Face的GPT2分词器

# 导入自定义模块
from config import ModelConfig, TrainingConfig, DataConfig  # 配置类
from utils.logger import setup_logger, logger               # 日志工具
from utils.utils import set_seed, save_config_to_yaml       # 通用工具函数
from data.processors import MultiModalProcessor             # 多模态数据处理器
from data.dataset import create_data_loaders                # 数据加载器创建函数
from models.multimodal_gpt import MultiModalGPT             # 多模态GPT模型
from trainer import Trainer                                 # 训练器


def parse_args():
    """
    解析命令行参数
    
    这个函数定义了训练脚本需要的所有命令行参数，包括数据路径、模型配置和训练超参数等。
    使用argparse库来处理命令行输入，使脚本更加灵活和可配置。
    
    Returns:
        argparse.Namespace: 包含所有解析后参数的命名空间对象
    """
    # 创建参数解析器
    parser = argparse.ArgumentParser(description="多模态大模型训练")
    
    # ===== 数据参数 =====
    # 这些参数定义了训练和验证数据的位置和格式
    parser.add_argument(
        "--train_file", 
        type=str, 
        required=True,                # 必须提供训练文件
        help="训练数据文件路径 (JSON或CSV格式)"
    )
    parser.add_argument(
        "--validation_file", 
        type=str, 
        help="验证数据文件路径 (JSON或CSV格式，可选)"
    )
    parser.add_argument(
        "--data_dir", 
        type=str, 
        default="./data", 
        help="数据目录，用于解析相对路径"
    )
    parser.add_argument(
        "--enable_multimodal_output",
        action="store_true",
        help="是否启用多模态输出（模型自己学习决定输出哪种模态）"
    )
    
    # ===== 模型参数 =====
    # 这些参数定义了模型的基本配置
    parser.add_argument(
        "--model_name", 
        type=str, 
        default="mmlm", 
        help="模型名称，用于保存和日志"
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
    
    # ===== 训练参数 =====
    # 这些参数控制训练过程的各个方面
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="./outputs", 
        help="输出目录，用于保存模型检查点和日志"
    )
    parser.add_argument(
        "--per_device_train_batch_size", 
        type=int, 
        default=8, 
        help="每个设备(GPU)的训练批次大小"
    )
    parser.add_argument(
        "--per_device_eval_batch_size", 
        type=int, 
        default=8, 
        help="每个设备(GPU)的评估批次大小"
    )
    parser.add_argument(
        "--gradient_accumulation_steps", 
        type=int, 
        default=4, 
        help="梯度累积步数，用于模拟更大的批次大小"
    )
    parser.add_argument(
        "--learning_rate", 
        type=float, 
        default=5e-5, 
        help="初始学习率"
    )
    parser.add_argument(
        "--weight_decay", 
        type=float, 
        default=0.01, 
        help="权重衰减(L2正则化)"
    )
    parser.add_argument(
        "--num_train_epochs", 
        type=int, 
        default=3, 
        help="训练轮数(完整数据集迭代次数)"
    )
    parser.add_argument(
        "--warmup_ratio", 
        type=float, 
        default=0.1, 
        help="学习率预热比例(总训练步数的百分比)"
    )
    parser.add_argument(
        "--logging_steps", 
        type=int, 
        default=100, 
        help="日志记录间隔步数"
    )
    parser.add_argument(
        "--save_steps", 
        type=int, 
        default=1000, 
        help="模型保存间隔步数"
    )
    parser.add_argument(
        "--eval_steps", 
        type=int, 
        default=1000, 
        help="评估间隔步数"
    )
    
    # ===== 其他参数 =====
    # 这些参数控制训练的其他方面
    parser.add_argument(
        "--seed", 
        type=int, 
        default=42, 
        help="随机种子，用于结果复现"
    )
    parser.add_argument(
        "--fp16", 
        action="store_true",      # 布尔标志，存在则为True
        help="是否使用混合精度训练(FP16)以加速训练"
    )
    parser.add_argument(
        "--local_rank", 
        type=int, 
        default=-1, 
        help="分布式训练的本地排名，由分布式启动器设置"
    )
    parser.add_argument(
        "--wandb", 
        action="store_true",      # 布尔标志，存在则为True
        help="是否使用Weights & Biases记录实验"
    )
    parser.add_argument(
        "--wandb_project", 
        type=str, 
        default="mmlm", 
        help="Weights & Biases项目名称"
    )
    
    # 解析参数并返回
    return parser.parse_args()


def main():
    """
    主函数 - 训练流程的入口点
    
    这个函数实现了完整的训练流程:
    1. 初始化环境(随机种子、日志、设备)
    2. 准备数据(分词器、处理器、数据加载器)
    3. 构建模型
    4. 训练模型
    5. 保存结果
    """
    # ===== 第1步: 初始化环境 =====
    
    # 解析命令行参数
    args = parse_args()
    
    # 设置随机种子以确保结果可复现
    # 这会设置Python、NumPy和PyTorch的随机种子
    set_seed(args.seed)
    
    # 创建输出目录并设置日志
    os.makedirs(args.output_dir, exist_ok=True)  # 创建输出目录(如果不存在)
    log_file = os.path.join(args.output_dir, "train.log")  # 日志文件路径
    logger = setup_logger(log_file=log_file)  # 设置日志记录器
    
    # 检测并设置计算设备(GPU或CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"使用设备: {device}")  # 记录使用的设备信息
    
    # ===== 第2步: 准备数据 =====
    
    # 加载预训练的GPT2分词器
    logger.info(f"加载分词器: {args.pretrained_text_model}")
    tokenizer = GPT2Tokenizer.from_pretrained(args.pretrained_text_model)
    
    # 添加特殊标记到分词器
    # 这些标记对于序列处理很重要
    special_tokens = {
        "pad_token": "<pad>",  # 填充标记
        "bos_token": "<bos>",  # 序列开始标记
        "eos_token": "<eos>",  # 序列结束标记
    }
    tokenizer.add_special_tokens(special_tokens)
    logger.info(f"分词器词汇表大小: {len(tokenizer)}")
    
    # 创建多模态处理器
    # 这个处理器负责处理文本、图像、视频和音频输入，以及多模态输出
    logger.info("创建多模态处理器")
    processor = MultiModalProcessor(
        tokenizer=tokenizer,       # 文本分词器
        image_size=224,            # 图像大小
        num_frames=8,              # 视频帧数
        frame_size=224,            # 视频帧大小
        video_sample_rate=4,       # 视频采样率
        max_text_length=512,       # 最大文本长度
        audio_sample_rate=16000,   # 音频采样率
        max_audio_length=10,       # 最大音频长度(秒)
    )
    
    # 创建训练和评估数据加载器
    logger.info(f"创建数据加载器，训练文件: {args.train_file}")
    train_dataloader, eval_dataloader = create_data_loaders(
        train_file=args.train_file,                              # 训练数据文件
        processor=processor,                                     # 数据处理器
        validation_file=args.validation_file,                    # 验证数据文件(可选)
        data_dir=args.data_dir,                                  # 数据目录
        train_batch_size=args.per_device_train_batch_size,       # 训练批次大小
        eval_batch_size=args.per_device_eval_batch_size,         # 评估批次大小
        num_workers=4,                                           # 数据加载线程数
        is_instruction=True,                                     # 是否为指令格式数据
    )
    
    # ===== 第3步: 构建模型 =====
    
    # 创建多模态GPT模型
    logger.info(f"创建多模态GPT模型，基于: {args.pretrained_text_model}、{args.pretrained_vision_model} 和 {args.pretrained_audio_model}")
    
    # 创建模型配置
    model_config = ModelConfig(
        pretrained_text_model=args.pretrained_text_model,          # 预训练文本模型
        pretrained_vision_model=args.pretrained_vision_model,      # 预训练视觉模型
        pretrained_audio_model=args.pretrained_audio_model,        # 预训练音频模型
        enable_multimodal_output=args.enable_multimodal_output,    # 是否启用多模态输出
    )
    
    # 创建多模态GPT模型
    model = MultiModalGPT(model_config)
    
    # 调整模型词汇表大小以匹配分词器
    # 这是必要的，因为我们添加了特殊标记
    model.text_model.resize_token_embeddings(len(tokenizer))
    logger.info(f"模型参数总数: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # ===== 第4步: 训练模型 =====
    
    # 创建训练器
    # 训练器封装了训练循环、优化器、学习率调度等
    logger.info("创建训练器")
    trainer = Trainer(
        model=model,                  # 模型
        tokenizer=tokenizer,          # 分词器
        train_dataloader=train_dataloader,  # 训练数据加载器
        eval_dataloader=eval_dataloader,    # 评估数据加载器
        args=args,                    # 训练参数
        device=device,                # 计算设备
    )
    
    # 开始训练
    logger.info("开始训练")
    trainer.train()
    
    # ===== 第5步: 保存结果 =====
    
    # 保存最终模型
    final_checkpoint_path = os.path.join(args.output_dir, "checkpoint-final")
    logger.info(f"保存最终模型到: {final_checkpoint_path}")
    trainer.save_model(final_checkpoint_path)
    
    # 训练完成
    logger.info("训练完成！")


if __name__ == "__main__":
    """
    脚本入口点
    
    当脚本直接运行(而不是被导入)时，执行main()函数
    这是Python脚本的标准模式
    """
    main()