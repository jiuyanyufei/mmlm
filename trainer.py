"""
训练器模块
"""
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm
import time
import wandb
from typing import Dict, List, Optional, Union, Any, Tuple

from utils.logger import logger
from utils.utils import count_parameters, format_time


class Trainer:
    """训练器"""
    
    def __init__(
        self,
        model: nn.Module,
        tokenizer,
        train_dataloader: DataLoader,
        eval_dataloader: Optional[DataLoader] = None,
        args = None,
        device = None,
    ):
        """
        初始化训练器
        
        Args:
            model: 模型
            tokenizer: 分词器
            train_dataloader: 训练数据加载器
            eval_dataloader: 评估数据加载器
            args: 训练参数
            device: 设备
        """
        self.model = model
        self.tokenizer = tokenizer
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.args = args
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 将模型移动到设备
        self.model.to(self.device)
        
        # 打印模型参数数量
        num_params = count_parameters(self.model)
        logger.info(f"模型参数数量: {num_params:,}")
        
        # 设置优化器
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
        )
        
        # 计算总训练步数
        self.num_update_steps_per_epoch = len(self.train_dataloader) // args.gradient_accumulation_steps
        self.max_train_steps = args.num_train_epochs * self.num_update_steps_per_epoch
        
        # 设置学习率调度器
        self.lr_scheduler = get_linear_schedule_with_warmup(
            optimizer=self.optimizer,
            num_warmup_steps=int(self.max_train_steps * args.warmup_ratio),
            num_training_steps=self.max_train_steps,
        )
        
        # 设置混合精度训练
        self.scaler = torch.cuda.amp.GradScaler() if args.fp16 else None
        
        # 初始化wandb
        if args.wandb:
            wandb.init(project=args.wandb_project, name=args.model_name)
            wandb.config.update(args)
    
    def train(self):
        """训练模型"""
        logger.info("开始训练...")
        global_step = 0
        total_loss = 0.0
        best_eval_loss = float("inf")
        
        for epoch in range(self.args.num_train_epochs):
            self.model.train()
            epoch_start_time = time.time()
            
            progress_bar = tqdm(self.train_dataloader, desc=f"Epoch {epoch+1}/{self.args.num_train_epochs}")
            
            for step, batch in enumerate(progress_bar):
                # 将批次移动到设备
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                
                # 混合精度训练
                if self.args.fp16:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(**batch)
                        loss = outputs["loss"]
                        loss = loss / self.args.gradient_accumulation_steps
                    
                    # 缩放损失并反向传播
                    self.scaler.scale(loss).backward()
                    
                    # 梯度累积
                    if (step + 1) % self.args.gradient_accumulation_steps == 0:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                        self.lr_scheduler.step()
                        self.optimizer.zero_grad()
                        global_step += 1
                else:
                    # 正常训练
                    outputs = self.model(**batch)
                    loss = outputs["loss"]
                    loss = loss / self.args.gradient_accumulation_steps
                    
                    # 反向传播
                    loss.backward()
                    
                    # 梯度累积
                    if (step + 1) % self.args.gradient_accumulation_steps == 0:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                        self.optimizer.step()
                        self.lr_scheduler.step()
                        self.optimizer.zero_grad()
                        global_step += 1
                
                # 更新总损失
                total_loss += loss.item()
                
                # 更新进度条
                progress_bar.set_postfix({"loss": loss.item()})
                
                # 记录日志
                if global_step > 0 and self.args.logging_steps > 0 and global_step % self.args.logging_steps == 0:
                    avg_loss = total_loss / self.args.logging_steps
                    logger.info(f"步骤 {global_step}/{self.max_train_steps} - 平均损失: {avg_loss:.4f}")
                    
                    if self.args.wandb:
                        wandb.log({"train/loss": avg_loss, "train/lr": self.lr_scheduler.get_last_lr()[0]})
                    
                    total_loss = 0.0
                
                # 评估
                if self.eval_dataloader is not None and global_step > 0 and self.args.eval_steps > 0 and global_step % self.args.eval_steps == 0:
                    eval_loss = self.evaluate()
                    logger.info(f"步骤 {global_step}/{self.max_train_steps} - 评估损失: {eval_loss:.4f}")
                    
                    if self.args.wandb:
                        wandb.log({"eval/loss": eval_loss})
                    
                    # 保存最佳模型
                    if eval_loss < best_eval_loss:
                        best_eval_loss = eval_loss
                        self.save_model(os.path.join(self.args.output_dir, "checkpoint-best"))
                        logger.info(f"保存最佳模型，评估损失: {best_eval_loss:.4f}")
                
                # 保存检查点
                if global_step > 0 and self.args.save_steps > 0 and global_step % self.args.save_steps == 0:
                    self.save_model(os.path.join(self.args.output_dir, f"checkpoint-{global_step}"))
            
            # 计算每个epoch的时间
            epoch_time = time.time() - epoch_start_time
            logger.info(f"Epoch {epoch+1}/{self.args.num_train_epochs} 完成，用时: {format_time(epoch_time)}")
    
    def evaluate(self):
        """评估模型"""
        self.model.eval()
        total_loss = 0.0
        
        with torch.no_grad():
            for batch in tqdm(self.eval_dataloader, desc="Evaluating"):
                # 将批次移动到设备
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                
                # 前向传播
                outputs = self.model(**batch)
                loss = outputs["loss"]
                
                # 更新总损失
                total_loss += loss.item()
        
        # 计算平均损失
        avg_loss = total_loss / len(self.eval_dataloader)
        
        # 切换回训练模式
        self.model.train()
        
        return avg_loss
    
    def save_model(self, output_dir: str):
        """
        保存模型
        
        Args:
            output_dir: 输出目录
        """
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 保存模型
        self.model.save_pretrained(output_dir)
        
        # 保存分词器
        self.tokenizer.save_pretrained(output_dir)
        
        logger.info(f"模型保存到 {output_dir}")