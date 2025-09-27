# 多模态大模型 (MMLM)

这是一个基于GPT架构的多模态大模型实现，支持处理文本、图像和视频输入，并生成文本输出。

## 功能特点

- 支持多模态输入：文本、图像、视频
- 基于GPT架构的生成式模型
- 支持指令微调
- 灵活的训练和推理配置

## 项目结构

```
.
├── config.py           # 配置文件
├── data/               # 数据处理模块
│   ├── __init__.py
│   ├── dataset.py      # 数据集定义
│   └── processors.py   # 数据处理器
├── models/             # 模型定义
│   ├── __init__.py
│   ├── generation.py   # 生成模块
│   ├── multimodal_gpt.py # 多模态GPT模型
│   └── vision_encoder.py # 视觉编码器
├── utils/              # 工具函数
│   ├── __init__.py
│   ├── logger.py       # 日志工具
│   └── utils.py        # 通用工具
├── train.py            # 训练脚本
├── trainer.py          # 训练器
├── inference.py        # 推理脚本
└── requirements.txt    # 项目依赖
```

## 安装

1. 克隆仓库：

```bash
git clone https://github.com/yourusername/mmlm.git
cd mmlm
```

2. 安装依赖：

```bash
pip install -r requirements.txt
```

## 数据准备

准备训练数据，格式为JSON或CSV，包含以下字段：

- `text`: 文本输入
- `image`: 图像路径（可选）
- `video`: 视频路径（可选）
- `instruction`: 指令（用于指令微调）
- `answer`: 期望的回答（用于指令微调）

## 训练

使用以下命令训练模型：

```bash
python train.py \
    --train_file path/to/train.json \
    --validation_file path/to/validation.json \
    --data_dir path/to/data \
    --output_dir path/to/output \
    --pretrained_text_model gpt2-medium \
    --pretrained_vision_model google/vit-base-patch16-224 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --gradient_accumulation_steps 4 \
    --learning_rate 5e-5 \
    --num_train_epochs 3 \
    --fp16
```

## 推理

使用以下命令进行推理：

```bash
python inference.py \
    --checkpoint_path path/to/checkpoint \
    --text "这是一个测试文本" \
    --image path/to/image.jpg \
    --video path/to/video.mp4 \
    --max_new_tokens 100 \
    --temperature 0.7 \
    --do_sample
```

## 示例

### 文本+图像输入

```python
from transformers import GPT2Tokenizer
from models.multimodal_gpt import MultiModalGPT
from data.processors import ImageProcessor

# 加载模型和分词器
tokenizer = GPT2Tokenizer.from_pretrained("path/to/checkpoint")
model = MultiModalGPT.from_pretrained("path/to/checkpoint")

# 处理输入
text = "请描述这张图片"
image_processor = ImageProcessor(image_size=224)
image = image_processor("path/to/image.jpg").unsqueeze(0)
inputs = tokenizer(text, return_tensors="pt")

# 生成文本
outputs = model.generate(
    input_ids=inputs["input_ids"],
    attention_mask=inputs["attention_mask"],
    image=image,
    max_new_tokens=100,
)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)
```

## 许可证

MIT

## 致谢

- [Transformers](https://github.com/huggingface/transformers)
- [PyTorch](https://pytorch.org/)