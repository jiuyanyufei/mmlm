"""
多模态大模型使用示例
"""
import torch
from transformers import GPT2Tokenizer

from data.processors import ImageProcessor, VideoProcessor
from models.multimodal_gpt import MultiModalGPT


def text_only_example(model_path: str):
    """纯文本示例"""
    # 加载分词器和模型
    tokenizer = GPT2Tokenizer.from_pretrained(model_path)
    model = MultiModalGPT.from_pretrained(model_path)
    model.eval()
    
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # 输入文本
    text = "请介绍一下人工智能的发展历程"
    
    # 处理输入
    inputs = tokenizer(text, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)
    
    # 生成文本
    with torch.no_grad():
        output_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=100,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
        )
    
    # 解码生成的文本
    generated_text = tokenizer.decode(output_ids[0][input_ids.shape[1]:], skip_special_tokens=True)
    
    print("输入:", text)
    print("输出:", generated_text)


def image_text_example(model_path: str, image_path: str):
    """图像+文本示例"""
    # 加载分词器和模型
    tokenizer = GPT2Tokenizer.from_pretrained(model_path)
    model = MultiModalGPT.from_pretrained(model_path)
    model.eval()
    
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # 输入文本
    text = "请描述这张图片"
    
    # 处理文本输入
    inputs = tokenizer(text, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)
    
    # 处理图像输入
    image_processor = ImageProcessor(image_size=224)
    image = image_processor(image_path).unsqueeze(0).to(device)
    
    # 生成文本
    with torch.no_grad():
        output_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            image=image,
            max_new_tokens=100,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
        )
    
    # 解码生成的文本
    generated_text = tokenizer.decode(output_ids[0][input_ids.shape[1]:], skip_special_tokens=True)
    
    print("输入:", text)
    print("输出:", generated_text)


def video_text_example(model_path: str, video_path: str):
    """视频+文本示例"""
    # 加载分词器和模型
    tokenizer = GPT2Tokenizer.from_pretrained(model_path)
    model = MultiModalGPT.from_pretrained(model_path)
    model.eval()
    
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # 输入文本
    text = "请描述这个视频"
    
    # 处理文本输入
    inputs = tokenizer(text, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)
    
    # 处理视频输入
    video_processor = VideoProcessor(num_frames=8, frame_size=224)
    video = video_processor(video_path).unsqueeze(0).to(device)
    
    # 生成文本
    with torch.no_grad():
        output_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            video=video,
            max_new_tokens=100,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
        )
    
    # 解码生成的文本
    generated_text = tokenizer.decode(output_ids[0][input_ids.shape[1]:], skip_special_tokens=True)
    
    print("输入:", text)
    print("输出:", generated_text)


if __name__ == "__main__":
    # 示例用法
    model_path = "./outputs/checkpoint-best"
    
    # 纯文本示例
    print("===== 纯文本示例 =====")
    text_only_example(model_path)
    
    # 图像+文本示例
    print("\n===== 图像+文本示例 =====")
    image_text_example(model_path, "./data/examples/image.jpg")
    
    # 视频+文本示例
    print("\n===== 视频+文本示例 =====")
    video_text_example(model_path, "./data/examples/video.mp4")