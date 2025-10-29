# 导入命令行参数解析模块
import argparse
# 导入PIL图像处理库
from PIL import Image
# 导入进度条显示库
from tqdm import tqdm
# 导入操作系统接口模块
import os

# 导入PyTorch深度学习框架
import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
# 导入PyTorch的图像变换模块
import torchvision.transforms as transforms

# 尝试从torchvision导入插值模式（新版本）
try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    # 如果导入失败（旧版本），使用PIL的双三次插值
    BICUBIC = Image.BICUBIC

# 导入数据增强器和数据集构建函数
from data.datautils import Augmenter, build_dataset
# 导入随机种子设置工具
from utils.tools import set_random_seed
# 导入类别名称相关的所有内容
from data.cls_to_names import *

# 导入CLIP模型
from clip import clip
# 加载CLIP模型到CPU
# 参数:
#   arch: 模型架构名称（如'ViT-B/16'）
# 返回:
#   model: 加载好的CLIP模型
def load_clip_to_cpu(arch):
    # 获取模型的下载URL
    url = clip._MODELS[arch]
    # 下载模型文件
    model_path = clip._download(url)
    try:
        # 尝试加载JIT编译的模型
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None
    except RuntimeError:
        # 如果JIT加载失败，则加载状态字典
        state_dict = torch.load(model_path, map_location="cpu")
    # 构建CLIP模型
    model = clip.build_model(state_dict or model.state_dict())
    return model

# 预提取多模态特征函数（Multimodal Enhanced Classification版本）
# 这个函数将两组图像（retrieved_images和test_images）及其对应文本描述的多模态特征提前提取并保存
# 实现：待测试[图-文] 与 检索到的[图-文] 进行匹配
# 参数:
#   retrieved_loader: 检索图像数据加载器（包含检索到的图片）
#   test_loader: 测试图像数据加载器（包含待测试的图片）
#   clip_model: CLIP模型
#   args: 命令行参数
@torch.no_grad()
def pre_extract_multimodal_feature(retrieved_loader, test_loader, clip_model, args):

    # 创建保存目录
    save_dir = f"./pre_extracted_feat/{args.arch.replace('/', '')}/seed{args.seed}"
    os.makedirs(save_dir, exist_ok=True)

    # 加载文本描述
    retrieved_desc_file = f"./descriptions/{args.test_set}_retrieved_descriptions.json"
    test_desc_file = f"./descriptions/{args.test_set}_test_descriptions.json"
    
    if not os.path.exists(retrieved_desc_file):
        raise FileNotFoundError(f"检索描述文件不存在: {retrieved_desc_file}")
    if not os.path.exists(test_desc_file):
        raise FileNotFoundError(f"测试描述文件不存在: {test_desc_file}")
    
    retrieved_descriptions = load_json(retrieved_desc_file)
    test_descriptions = load_json(test_desc_file)
    
    print(f"加载检索描述: {len(retrieved_descriptions)} 条")
    print(f"加载测试描述: {len(test_descriptions)} 条")

    # 存储所有提取的多模态特征
    all_retrieved_data = []  # 检索到的[图-文]特征
    all_test_data = []       # 待测试的[图-文]特征
    
    # 处理检索图像及其描述
    print("Processing retrieved [image-text] pairs...")
    for i, (images, target) in enumerate(tqdm(retrieved_loader)):
        # 确保images是列表（包含多个增强视图）
        assert isinstance(images, list)
        # 将所有视图移到GPU
        for k in range(len(images)):
            images[k] = images[k].cuda(non_blocking=True)
        # 拼接所有视图
        images = torch.cat(images, dim=0)
        # 将标签移到GPU
        target = target.cuda(non_blocking=True)

        # 使用混合精度提取多模态特征
        with torch.cuda.amp.autocast():
            # 使用CLIP编码图像（多个增强视图）
            image_features = clip_model.encode_image(images)
            # L2归一化特征
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            
            # 编码对应的文本描述
            text_description = retrieved_descriptions[str(i)]  # 按索引获取描述
            text_tokens = clip.tokenize([text_description]).cuda()
            text_features = clip_model.encode_text(text_tokens)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            
            # 拼接图文特征 [Ti, Ii] - 检索到的[图-文]特征
            # image_features shape: (n_views, d), text_features shape: (1, d)
            # 将text_features扩展到与image_features相同的视图数
            text_features_expanded = text_features.expand(image_features.size(0), -1)
            multimodal_features = torch.cat([text_features_expanded, image_features], dim=-1)  # shape: (n_views, 2*d)

        # 保存检索到的[图-文]特征和标签
        all_retrieved_data.append((multimodal_features, target))

    # 处理待测试图像及其描述
    print("Processing test [image-text] pairs...")
    for i, (images, target) in enumerate(tqdm(test_loader)):
        # 确保images是列表（包含多个增强视图）
        assert isinstance(images, list)
        # 将所有视图移到GPU
        for k in range(len(images)):
            images[k] = images[k].cuda(non_blocking=True)
        # 拼接所有视图
        images = torch.cat(images, dim=0)
        # 将标签移到GPU
        target = target.cuda(non_blocking=True)

        # 使用混合精度提取多模态特征
        with torch.cuda.amp.autocast():
            # 使用CLIP编码图像（多个增强视图）
            image_features = clip_model.encode_image(images)
            # L2归一化特征
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            
            # 编码对应的文本描述
            text_description = test_descriptions[str(i)]  # 按索引获取描述
            text_tokens = clip.tokenize([text_description]).cuda()
            text_features = clip_model.encode_text(text_tokens)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            
            # 拼接图文特征 [T'j, I'j] - 待测试的[图-文]特征
            # image_features shape: (n_views, d), text_features shape: (1, d)
            # 将text_features扩展到与image_features相同的视图数
            text_features_expanded = text_features.expand(image_features.size(0), -1)
            multimodal_features = torch.cat([text_features_expanded, image_features], dim=-1)  # shape: (n_views, 2*d)

        # 保存待测试的[图-文]特征和标签
        all_test_data.append((multimodal_features, target))

    # 保存到文件
    retrieved_save_path = os.path.join(save_dir, f"{args.test_set}_retrieved.pth")
    test_save_path = os.path.join(save_dir, f"{args.test_set}_test.pth")
    
    torch.save(all_retrieved_data, retrieved_save_path)
    torch.save(all_test_data, test_save_path)
    
    print(f"Successfully save retrieved [image-text] features to [{retrieved_save_path}]")
    print(f"Successfully save test [image-text] features to [{test_save_path}]")


# 加载JSON文件的辅助函数
def load_json(file_path):
    import json
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


# 主工作函数（Multimodal Enhanced Classification版本）
# 参数:
#   args: 命令行参数
def main_worker(args):
    # 加载CLIP模型
    print("=> Model created: visual backbone {}".format(args.arch))
    clip_model = load_clip_to_cpu(args.arch)
    # 将模型移到GPU
    clip_model = clip_model.cuda()
    # 使用float32精度
    clip_model.float()
    # 设置为评估模式
    clip_model.eval()

    # 冻结所有参数（不需要梯度）
    for _, param in clip_model.named_parameters():
        param.requires_grad_(False)

    # CLIP的归一化统计参数（从clip.load()获取）
    normalize = transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                     std=[0.26862954, 0.26130258, 0.27577711])

    # 基础图像变换：调整大小和中心裁剪
    base_transform = transforms.Compose([
        transforms.Resize(args.resolution, interpolation=BICUBIC),
        transforms.CenterCrop(args.resolution)])
    # 预处理：转换为张量并归一化
    preprocess = transforms.Compose([transforms.ToTensor(), normalize])
    # 创建数据增强器，生成多个增强视图
    data_transform = Augmenter(base_transform, preprocess, n_views=args.batch_size)

    # 打印正在处理的数据集
    print("Extracting features for: {}".format(args.test_set))

    # 构建两个数据集：检索图像和测试图像
    retrieved_dataset = build_dataset(f"{args.test_set}_retrieved", data_transform, args.data, mode='test')
    test_dataset = build_dataset(f"{args.test_set}_test", data_transform, args.data, mode='test')
    
    print("number of retrieved samples: {}".format(len(retrieved_dataset)))
    print("number of test samples: {}".format(len(test_dataset)))
    
    # 创建数据加载器
    retrieved_loader = torch.utils.data.DataLoader(
                retrieved_dataset,
                batch_size=1, shuffle=False,  # 保持顺序以匹配描述文件
                num_workers=args.workers, pin_memory=True)
    
    test_loader = torch.utils.data.DataLoader(
                test_dataset,
                batch_size=1, shuffle=False,  # 保持顺序以匹配描述文件
                num_workers=args.workers, pin_memory=True)
    
    # 开始提取多模态特征
    pre_extract_multimodal_feature(retrieved_loader, test_loader, clip_model, args)


# 主程序入口
if __name__ == '__main__':
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description='Pre-extracting image features')
    parser.add_argument('data', metavar='DIR', help='path to dataset root')  # 数据集根目录
    parser.add_argument('--test_set', type=str, help='dataset name')  # 数据集名称
    parser.add_argument('-a', '--arch', metavar='ARCH', default='ViT-B/16')  # 模型架构
    parser.add_argument('--resolution', default=224, type=int, help='CLIP image resolution')  # 图像分辨率
    parser.add_argument('-j', '--workers', default=16, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')  # 数据加载线程数
    parser.add_argument('-b', '--batch-size', default=50, type=int, metavar='N')  # 增强视图数量
    parser.add_argument('--seed', type=int, default=0)  # 随机种子

    args = parser.parse_args()
    # 设置随机种子以保证可重复性
    set_random_seed(args.seed)
    # 启动主工作函数
    main_worker(args)