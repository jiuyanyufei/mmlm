"""
多模态大模型使用示例

这个文件展示了如何使用我们的多模态大模型进行不同类型的推理：
1. 纯文本推理 - 只使用文本输入
2. 图像+文本推理 - 结合图像和文本输入
3. 视频+文本推理 - 结合视频和文本输入

每个示例都展示了完整的处理流程，从加载模型到生成最终输出。
"""

# 导入PyTorch，这是我们用于深度学习的主要库
import torch
# 导入GPT2分词器，用于将文本转换为模型可以理解的标记(tokens)
from transformers import GPT2Tokenizer

# 导入我们自定义的处理器，用于处理图像和视频输入
from data.processors import ImageProcessor, VideoProcessor
# 导入我们的多模态GPT模型
from models.multimodal_gpt import MultiModalGPT


def text_only_example(model_path: str):
    """
    纯文本示例 - 演示如何使用模型处理纯文本输入并生成回答
    
    参数:
        model_path: 模型检查点的路径，例如 "./outputs/checkpoint-best"
    """
    # 第1步: 加载分词器和模型
    # 分词器将文本转换为数字ID序列，这些ID是模型可以理解的
    tokenizer = GPT2Tokenizer.from_pretrained(model_path)
    # 从保存的检查点加载我们的多模态GPT模型
    model = MultiModalGPT.from_pretrained(model_path)
    # 将模型设置为评估模式，这会禁用dropout等训练特定功能
    model.eval()
    
    # 第2步: 设置计算设备
    # 如果有GPU可用就使用GPU，否则使用CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 将模型移动到选定的设备上
    model.to(device)
    
    # 第3步: 准备输入文本
    text = "请介绍一下人工智能的发展历程"
    
    # 第4步: 处理输入文本
    # tokenizer将文本转换为标记ID，并返回一个包含input_ids和attention_mask的字典
    # return_tensors="pt"表示返回PyTorch张量
    inputs = tokenizer(text, return_tensors="pt")
    # 提取输入ID并移动到设备上
    input_ids = inputs["input_ids"].to(device)
    # 提取注意力掩码并移动到设备上（掩码用于处理不同长度的序列）
    attention_mask = inputs["attention_mask"].to(device)
    
    # 第5步: 生成文本
    # torch.no_grad()禁用梯度计算，这在推理时可以节省内存并加速计算
    with torch.no_grad():
        # 调用模型的generate方法生成文本
        output_ids = model.generate(
            input_ids=input_ids,                # 输入ID
            attention_mask=attention_mask,      # 注意力掩码
            max_new_tokens=100,                 # 最多生成100个新标记
            temperature=0.7,                    # 温度参数控制随机性，较低的值使输出更确定
            top_p=0.9,                          # 只考虑概率累积达到90%的标记
            do_sample=True,                     # 使用采样而不是贪婪解码
        )
    
    # 第6步: 解码生成的文本
    # 我们只需要新生成的部分，所以从input_ids.shape[1]开始
    # skip_special_tokens=True会移除特殊标记如<eos>
    generated_text = tokenizer.decode(output_ids[0][input_ids.shape[1]:], skip_special_tokens=True)
    
    # 第7步: 打印结果
    print("输入:", text)
    print("输出:", generated_text)


def image_text_example(model_path: str, image_path: str):
    """
    图像+文本示例 - 演示如何结合图像和文本输入进行多模态推理
    
    参数:
        model_path: 模型检查点的路径
        image_path: 图像文件的路径
    """
    # 第1步: 加载分词器和模型（与纯文本示例相同）
    tokenizer = GPT2Tokenizer.from_pretrained(model_path)
    model = MultiModalGPT.from_pretrained(model_path)
    model.eval()
    
    # 第2步: 设置计算设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # 第3步: 准备输入文本
    text = "请描述这张图片"
    
    # 第4步: 处理文本输入（与纯文本示例相同）
    inputs = tokenizer(text, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)
    
    # 第5步: 处理图像输入 - 这是多模态处理的关键部分
    # 创建图像处理器，设置图像大小为224x224像素（ViT模型的标准输入大小）
    image_processor = ImageProcessor(image_size=224)
    # 处理图像：读取、调整大小、标准化，然后添加批次维度并移动到设备上
    # unsqueeze(0)添加批次维度，因为模型期望输入形状为[batch_size, channels, height, width]
    image = image_processor(image_path).unsqueeze(0).to(device)
    
    # 第6步: 生成文本 - 注意这里我们额外传入了image参数
    with torch.no_grad():
        output_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            image=image,                        # 传入处理后的图像
            max_new_tokens=100,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
        )
    
    # 第7步: 解码生成的文本
    generated_text = tokenizer.decode(output_ids[0][input_ids.shape[1]:], skip_special_tokens=True)
    
    # 第8步: 打印结果
    print("输入:", text)
    print("输出:", generated_text)


def video_text_example(model_path: str, video_path: str):
    """
    视频+文本示例 - 演示如何结合视频和文本输入进行多模态推理
    
    参数:
        model_path: 模型检查点的路径
        video_path: 视频文件的路径
    """
    # 第1步: 加载分词器和模型（与前面的示例相同）
    tokenizer = GPT2Tokenizer.from_pretrained(model_path)
    model = MultiModalGPT.from_pretrained(model_path)
    model.eval()
    
    # 第2步: 设置计算设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # 第3步: 准备输入文本
    text = "请描述这个视频"
    
    # 第4步: 处理文本输入（与前面的示例相同）
    inputs = tokenizer(text, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)
    
    # 第5步: 处理视频输入 - 这是处理视频的关键部分
    # 创建视频处理器，设置要采样的帧数和每帧的大小
    # num_frames=8表示我们从视频中采样8帧
    # frame_size=224设置每帧的大小为224x224像素
    video_processor = VideoProcessor(num_frames=8, frame_size=224)
    # 处理视频：读取、采样帧、调整大小、标准化，然后添加批次维度并移动到设备上
    # 输出形状为[batch_size, num_frames, channels, height, width]
    video = video_processor(video_path).unsqueeze(0).to(device)
    
    # 第6步: 生成文本 - 注意这里我们传入了video参数而不是image
    with torch.no_grad():
        output_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            video=video,                        # 传入处理后的视频
            max_new_tokens=100,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
        )
    
    # 第7步: 解码生成的文本
    generated_text = tokenizer.decode(output_ids[0][input_ids.shape[1]:], skip_special_tokens=True)
    
    # 第8步: 打印结果
    print("输入:", text)
    print("输出:", generated_text)


if __name__ == "__main__":
    """
    主程序入口点 - 当直接运行此脚本时执行
    
    这里我们展示了三种不同类型的推理示例：
    1. 纯文本推理
    2. 图像+文本推理
    3. 视频+文本推理
    
    要运行此示例，您需要:
    1. 已训练好的模型检查点
    2. 示例图像和视频文件
    """
    # 设置模型路径 - 这应该指向您训练好的模型检查点
    model_path = "./outputs/checkpoint-best"
    
    # 运行纯文本示例
    print("===== 纯文本示例 =====")
    print("这个示例展示了模型如何处理纯文本查询")
    text_only_example(model_path)
    
    # 运行图像+文本示例
    print("\n===== 图像+文本示例 =====")
    print("这个示例展示了模型如何结合图像和文本输入")
    # 确保图像文件存在于指定路径
    image_text_example(model_path, "./data/examples/image.jpg")
    
    # 运行视频+文本示例
    print("\n===== 视频+文本示例 =====")
    print("这个示例展示了模型如何结合视频和文本输入")
    # 确保视频文件存在于指定路径
    video_text_example(model_path, "./data/examples/video.mp4")
    
    print("\n所有示例运行完成！")