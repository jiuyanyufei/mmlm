"""
多模态大模型训练脚本
"""
import os
import argparse
import torch
from transformers import GPT2Tokenizer

from config import ModelConfig, TrainingConfig, DataConfig
from utils.logger import setup_logger, logger
from utils.utils import set_seed, save_config_to_yaml
from data.processors import MultiModalProcessor
from data.dataset import create_data_loaders
from models.multimodal_gpt import MultiModalGPT
from trainer import Trainer


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="多模态大模型训练")
    
    # 数据参数
    parser.add_argument("--train_file", type=str, required=True, help="训练数据文件路径")
    parser.add_argument("--validation_file", type=str, help="验证数据文件路径")
    parser.add_argument("--data_dir", type=str, default="./data", help="数据目录")
    
    # 模型参数
    parser.add_argument("--model_name", type=str, default="mmlm", help="模型名称")
    parser.add_argument("--pretrained_text_model", type=str, default="gpt2-medium", help="预训练文本模型名称")
    parser.add_argument("--pretrained_vision_model", type=str, default="google/vit-base-patch16-224", help="预训练视觉模型名称")
    
    # 训练参数
    parser.add_argument("--output_dir", type=str, default="./outputs", help="输出目录")
    parser.add_argument("--per_device_train_batch_size", type=int, default=8, help="每个设备的训练批次大小")
    parser.add_argument("--per_device_eval_batch_size", type=int, default=8, help="每个设备的评估批次大小")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4, help="梯度累积步数")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="学习率")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="权重衰减")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="训练轮数")
    parser.add_argument("--warmup_ratio", type=float, default=0.1, help="预热比例")
    parser.add_argument("--logging_steps", type=int, default=100, help="日志记录步数")
    parser.add_argument("--save_steps", type=int, default=1000, help="保存步数")
    parser.add_argument("--eval_steps", type=int, default=1000, help="评估步数")
    
    # 其他参数
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--fp16", action="store_true", help="是否使用混合精度训练")
    parser.add_argument("--local_rank", type=int, default=-1, help="分布式训练的本地排名")
    parser.add_argument("--wandb", action="store_true", help="是否使用wandb记录实验")
    parser.add_argument("--wandb_project", type=str, default="mmlm", help="wandb项目名称")
    
    return parser.parse_args()


def main():
    """主函数"""
    # 解析参数
    args = parse_args()
    
    # 设置随机种子
    set_seed(args.seed)
    
    # 设置日志
    log_file = os.path.join(args.output_dir, "train.log")
    os.makedirs(args.output_dir, exist_ok=True)
    logger = setup_logger(log_file=log_file)
    
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"使用设备: {device}")
    
    # 加载分词器
    tokenizer = GPT2Tokenizer.from_pretrained(args.pretrained_text_model)
    # 确保有特殊标记
    special_tokens = {
        "pad_token": "<pad>",
        "bos_token": "<bos>",
        "eos_token": "<eos>",
    }
    tokenizer.add_special_tokens(special_tokens)
    
    # 创建多模态处理器
    processor = MultiModalProcessor(
        tokenizer=tokenizer,
        image_size=224,
        num_frames=8,
        frame_size=224,
        video_sample_rate=4,
        max_text_length=512,
    )
    
    # 创建数据加载器
    train_dataloader, eval_dataloader = create_data_loaders(
        train_file=args.train_file,
        processor=processor,
        validation_file=args.validation_file,
        data_dir=args.data_dir,
        train_batch_size=args.per_device_train_batch_size,
        eval_batch_size=args.per_device_eval_batch_size,
        num_workers=4,
        is_instruction=True,
    )
    
    # 创建模型
    model = MultiModalGPT(
        pretrained_text_model=args.pretrained_text_model,
        pretrained_vision_model=args.pretrained_vision_model,
    )
    
    # 调整模型词汇表大小以匹配分词器
    model.text_model.resize_token_embeddings(len(tokenizer))
    
    # 创建训练器
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
        args=args,
        device=device,
    )
    
    # 开始训练
    trainer.train()
    
    # 保存最终模型
    trainer.save_model(os.path.join(args.output_dir, "checkpoint-final"))
    
    logger.info("训练完成！")


if __name__ == "__main__":
    main()