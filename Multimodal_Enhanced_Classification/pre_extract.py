# 导入命令行参数解析模块
import argparse
# 导入PIL图像处理库
from PIL import Image
# 导入进度条显示库
from tqdm import tqdm
# 导入操作系统接口模块
import os
import json
import sys
# 导入YAML配置文件处理模块
import yaml

# 导入文本描述生成器
from text_description_generator import TextDescriptionGenerator

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

# 导入CLIP模型
from clip import clip

# 读取配置文件
def load_config(config_path="./config.yaml"):
    """
    读取YAML配置文件
    """
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        print(f"⚠️  读取配置文件失败: {e}")
        # 返回默认配置
        return {
            "k_shot_image_processing": "average",
            "similarity_processing": "image_text_pair"
        }
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

# 简化的特征提取函数，专为discovering.py集成设计
@torch.no_grad()
def pre_extract_multimodal_feature(retrieved_loader, test_loader, clip_model, args, text_generator=None):
    """
    简化的多模态特征提取函数
    专为与discovering.py快慢思考系统集成而设计
    
    Args:
        text_generator: 文本描述生成器，用于为子视图生成描述
    
    修改支持：
    - 每个类别处理k张图像（从category_image_paths.json获取）
    - 构建所有k张图像的增强子图和描述
    - 支持批量特征提取和加权相似度计算
    """
    # 创建保存目录
    save_dir = f"./pre_extracted_feat/{args.arch.replace('/', '')}/seed{args.seed}"
    os.makedirs(save_dir, exist_ok=True)

    # 构建描述文件路径
    descriptions_dir = "./descriptions"
    retrieved_desc_file = os.path.join(descriptions_dir, f"{args.test_set}_retrieved_descriptions.json")
    test_desc_file = os.path.join(descriptions_dir, f"{args.test_set}_test_descriptions.json")
    
    print(f"检查描述文件:")
    print(f"  检索描述: {retrieved_desc_file}")
    print(f"  测试描述: {test_desc_file}")
    
    # 检查描述文件存在性
    if not os.path.exists(retrieved_desc_file):
        print(f"❌ 检索描述文件不存在: {retrieved_desc_file}")
        return False
    if not os.path.exists(test_desc_file):
        print(f"❌ 测试描述文件不存在: {test_desc_file}")
        return False
    
    # 加载描述文件
    try:
        with open(retrieved_desc_file, 'r', encoding='utf-8') as f:
            retrieved_descriptions_raw = json.load(f)
        with open(test_desc_file, 'r', encoding='utf-8') as f:
            test_descriptions_raw = json.load(f)
        
        # 处理可能的列表格式（由dump_json产生）
        if isinstance(retrieved_descriptions_raw, list) and len(retrieved_descriptions_raw) > 0:
            retrieved_descriptions = retrieved_descriptions_raw[0]
        else:
            retrieved_descriptions = retrieved_descriptions_raw
            
        if isinstance(test_descriptions_raw, list) and len(test_descriptions_raw) > 0:
            test_descriptions = test_descriptions_raw[0]
        else:
            test_descriptions = test_descriptions_raw
            
    except Exception as e:
        print(f"❌ 加载描述文件失败: {e}")
        return False
    
    print(f"✅ 成功加载描述文件:")
    print(f"  检索描述: {len(retrieved_descriptions)} 条")
    print(f"  测试描述: {len(test_descriptions)} 条")

    # 生成子视图文本描述（如果启用）
    generated_descriptions = {}
    if text_generator and text_generator.is_generate:
        print("🔄 为子视图生成文本描述...")
        try:
            # 为当前数据集生成描述
            generated_descriptions = text_generator.generate_dataset_descriptions(
                data_dir=args.data,
                dataset_name=args.test_set,
                cache_dir="./description_cache"
            )
            print(f"✅ 生成了 {len(generated_descriptions)} 个图像描述")
        except Exception as e:
            print(f"⚠️  文本描述生成失败: {e}")
            generated_descriptions = {}
    else:
        print("⚠️  文本描述生成功能已禁用")

    # 存储所有提取的多模态特征
    all_retrieved_data = []  # 检索到的[图-文]特征
    all_test_data = []       # 待测试的[图-文]特征
    
    # 处理检索图像及其描述 - 支持每个类别k张图像
    print("🔄 处理检索图像-文本对（支持每类别k张图像）...")
    try:
        # 按类别分组处理检索数据
        category_features = {}  # 存储每个类别的所有图像特征
        
        for i, batch_data in enumerate(tqdm(retrieved_loader, desc="Processing retrieved")):
            try:
                # 安全解包批次数据
                if isinstance(batch_data, (list, tuple)) and len(batch_data) >= 2:
                    images, target = batch_data[0], batch_data[1]
                else:
                    print(f"❌ 批次数据格式错误: {type(batch_data)}, 跳过")
                    continue
                
                print(f"Debug: batch {i}, images type={type(images)}, target type={type(target)}")
                
                # 安全处理图像数据
                if isinstance(images, list):
                    # 处理图像列表
                    valid_images = []
                    for k, img in enumerate(images):
                        if hasattr(img, 'cuda'):
                            valid_images.append(img.cuda(non_blocking=True))
                        else:
                            print(f"❌ 图像 {k} 不是tensor: {type(img)}")
                    
                    if valid_images:
                        images = torch.cat(valid_images, dim=0)
                    else:
                        print(f"❌ 没有有效图像，跳过批次 {i}")
                        continue
                        
                elif hasattr(images, 'cuda'):
                    # 单张图像tensor
                    images = images.cuda(non_blocking=True)
                else:
                    print(f"❌ 图像数据不是tensor: {type(images)}, 跳过")
                    continue
                
                # 安全处理标签
                if hasattr(target, 'cuda'):
                    target = target.cuda(non_blocking=True)
                    target_item = target.item()
                else:
                    print(f"❌ 标签不是tensor: {type(target)}, 跳过")
                    continue
                    
            except Exception as e:
                print(f"❌ 处理批次 {i} 失败: {e}")
                continue

            # 使用混合精度提取多模态特征
            with torch.cuda.amp.autocast():
                # 使用CLIP编码图像
                image_features = clip_model.encode_image(images)
                # L2归一化特征
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                
                # 获取对应的文本描述
                try:
                    # 首先尝试从生成的描述中获取
                    text_description = None
                    
                    # 尝试根据图像路径获取生成的描述
                    # 这里需要获取当前图像的实际路径
                    # 由于我们无法直接获取路径，先使用原有逻辑，后续可以改进
                    
                    desc_key = str(i) if str(i) in retrieved_descriptions else list(retrieved_descriptions.keys())[i % len(retrieved_descriptions)]
                    original_description = retrieved_descriptions[desc_key]
                    
                    # 如果有生成的描述且原描述是特征向量，则使用生成的描述
                    if generated_descriptions and isinstance(original_description, list) and len(original_description) > 0 and isinstance(original_description[0], (int, float)):
                        # 原描述是特征向量，尝试使用生成的描述
                        # 这里需要更好的映射机制，暂时使用默认描述
                        text_description = "a detailed photo"
                        print(f"⚠️  使用默认描述替代特征向量")
                    else:
                        text_description = original_description
                    
                    print(f"Debug: desc_key={desc_key}, text_description type={type(text_description)}")
                    
                    # 确保text_description是字符串
                    if isinstance(text_description, list):
                        # 检查是否是特征向量（数值列表）
                        if len(text_description) > 0 and isinstance(text_description[0], (int, float)):
                            # 这是特征向量，使用默认描述
                            text_description = "a photo"
                            print(f"⚠️  检测到特征向量而非文本描述，使用默认描述")
                        else:
                            # 这是文本列表，取第一个
                            text_description = text_description[0] if len(text_description) > 0 else "a photo"
                    elif isinstance(text_description, (int, float)):
                        text_description = "a photo"
                        print(f"⚠️  检测到数值而非文本描述，使用默认描述")
                    elif text_description is None:
                        text_description = "a photo"
                    elif not isinstance(text_description, str):
                        text_description = str(text_description)
                        
                    print(f"Debug: final text_description={text_description}")
                except Exception as e:
                    print(f"Debug: Error in text description processing: {e}")
                    text_description = "a photo"
                
                # 编码文本描述
                text_tokens = clip.tokenize([text_description], truncate=True).cuda()
                text_features = clip_model.encode_text(text_tokens)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                
                # 拼接图文特征 [Ti, Ii]
                if len(image_features.shape) == 1:
                    image_features = image_features.unsqueeze(0)
                text_features_expanded = text_features.expand(image_features.size(0), -1)
                multimodal_features = torch.cat([text_features_expanded, image_features], dim=-1)

            # 按类别存储特征（支持每个类别多张图像）
            if target_item not in category_features:
                category_features[target_item] = []
            category_features[target_item].append((multimodal_features, target))
            
            # 每100个样本打印一次进度
            if (i + 1) % 100 == 0:
                print(f"  已处理检索样本: {i + 1}")
        
        # 将按类别分组的特征转换为最终格式
        # 每个类别的所有图像作为一个批次处理
        for category, features_list in category_features.items():
            if len(features_list) > 1:
                # 多张图像：拼接所有图像的特征
                all_features = []
                targets = []
                for feat, tgt in features_list:
                    all_features.append(feat)
                    targets.append(tgt)
                # 拼接成一个大的特征张量 shape: (k*n_views, feature_dim)
                combined_features = torch.cat(all_features, dim=0)
                # 使用第一个目标作为代表（所有图像都是同一类别）
                representative_target = targets[0]
                all_retrieved_data.append((combined_features, representative_target))
            else:
                # 单张图像：直接使用
                all_retrieved_data.append(features_list[0])
                
    except Exception as e:
        print(f"❌ 处理检索图像失败: {e}")
        return False

    # 处理测试图像及其描述
    print("🔄 处理测试图像-文本对...")
    try:
        for i, batch_data in enumerate(tqdm(test_loader, desc="Processing test")):
            try:
                # 安全解包批次数据
                if isinstance(batch_data, (list, tuple)) and len(batch_data) >= 2:
                    images, target = batch_data[0], batch_data[1]
                else:
                    print(f"❌ 测试批次数据格式错误: {type(batch_data)}, 跳过")
                    continue
                
                # 安全处理图像数据
                if isinstance(images, list):
                    # 处理图像列表
                    valid_images = []
                    for k, img in enumerate(images):
                        if hasattr(img, 'cuda'):
                            valid_images.append(img.cuda(non_blocking=True))
                        else:
                            print(f"❌ 测试图像 {k} 不是tensor: {type(img)}")
                    
                    if valid_images:
                        images = torch.cat(valid_images, dim=0)
                    else:
                        print(f"❌ 没有有效测试图像，跳过批次 {i}")
                        continue
                        
                elif hasattr(images, 'cuda'):
                    # 单张图像tensor
                    images = images.cuda(non_blocking=True)
                else:
                    print(f"❌ 测试图像数据不是tensor: {type(images)}, 跳过")
                    continue
                
                # 安全处理标签
                if hasattr(target, 'cuda'):
                    target = target.cuda(non_blocking=True)
                    target_item = target.item()
                else:
                    print(f"❌ 测试标签不是tensor: {type(target)}, 跳过")
                    continue
                    
            except Exception as e:
                print(f"❌ 处理测试批次 {i} 失败: {e}")
                continue

            # 使用混合精度提取多模态特征
            with torch.cuda.amp.autocast():
                # 使用CLIP编码图像
                image_features = clip_model.encode_image(images)
                # L2归一化特征
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                
                # 获取对应的文本描述
                desc_key = str(i) if str(i) in test_descriptions else list(test_descriptions.keys())[i % len(test_descriptions)]
                text_description = test_descriptions[desc_key]
                
                # 确保text_description是字符串
                if isinstance(text_description, list):
                    # 检查是否是特征向量（数值列表）
                    if len(text_description) > 0 and isinstance(text_description[0], (int, float)):
                        # 这是特征向量，使用默认描述
                        text_description = "a photo"
                        print(f"⚠️  测试数据检测到特征向量而非文本描述，使用默认描述")
                    else:
                        # 这是文本列表，取第一个
                        text_description = text_description[0] if len(text_description) > 0 else "a photo"
                elif isinstance(text_description, (int, float)):
                    text_description = "a photo"
                    print(f"⚠️  测试数据检测到数值而非文本描述，使用默认描述")
                elif text_description is None:
                    text_description = "a photo"
                elif not isinstance(text_description, str):
                    text_description = str(text_description)
                
                # 编码文本描述
                text_tokens = clip.tokenize([text_description], truncate=True).cuda()
                text_features = clip_model.encode_text(text_tokens)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                
                # 拼接图文特征 [T'j, I'j]
                if len(image_features.shape) == 1:
                    image_features = image_features.unsqueeze(0)
                text_features_expanded = text_features.expand(image_features.size(0), -1)
                multimodal_features = torch.cat([text_features_expanded, image_features], dim=-1)

            # 保存特征和标签
            all_test_data.append((multimodal_features, target))
            
            # 每100个样本打印一次进度
            if (i + 1) % 100 == 0:
                print(f"  已处理测试样本: {i + 1}")
                
    except Exception as e:
        print(f"❌ 处理测试图像失败: {e}")
        return False

    # 保存到文件
    try:
        retrieved_save_path = os.path.join(save_dir, f"{args.test_set}_retrieved.pth")
        test_save_path = os.path.join(save_dir, f"{args.test_set}_test.pth")
        
        torch.save(all_retrieved_data, retrieved_save_path)
        torch.save(all_test_data, test_save_path)
        
        print(f"✅ 成功保存检索特征到: {retrieved_save_path}")
        print(f"✅ 成功保存测试特征到: {test_save_path}")
        print(f"📊 检索样本: {len(all_retrieved_data)} 个")
        print(f"📊 测试样本: {len(all_test_data)} 个")
        
        return True
        
    except Exception as e:
        print(f"❌ 保存特征文件失败: {e}")
        return False


# 加载JSON文件的辅助函数
def load_json(file_path):
    import json
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


# 简化的主工作函数，专为discovering.py集成设计
def main_worker(args):
    """
    简化的主工作函数
    专为与discovering.py快慢思考系统集成而设计
    """
    print(f"🚀 开始MEC特征预提取: {args.test_set}")
    print(f"📐 模型架构: {args.arch}")
    print(f"🎲 随机种子: {args.seed}")
    
    # 初始化文本描述生成器
    print("🔄 初始化文本描述生成器...")
    try:
        text_generator = TextDescriptionGenerator(device='cuda', device_id=0)
        print("✅ 文本描述生成器初始化完成")
    except Exception as e:
        print(f"⚠️  文本描述生成器初始化失败: {e}")
        text_generator = None
    
    try:
        # 加载CLIP模型
        print("🔄 加载CLIP模型...")
        clip_model = load_clip_to_cpu(args.arch)
        clip_model = clip_model.cuda()
        clip_model.float()
        clip_model.eval()

        # 冻结所有参数
        for _, param in clip_model.named_parameters():
            param.requires_grad_(False)
        
        print("✅ CLIP模型加载成功")

        # CLIP归一化参数
        normalize = transforms.Normalize(
            mean=[0.48145466, 0.4578275, 0.40821073],
            std=[0.26862954, 0.26130258, 0.27577711]
        )

        # 简化的图像变换（不使用过多增强，提高稳定性）
        base_transform = transforms.Compose([
            transforms.Resize(args.resolution, interpolation=BICUBIC),
            transforms.CenterCrop(args.resolution)
        ])
        
        preprocess = transforms.Compose([
            transforms.ToTensor(), 
            normalize
        ])
        
        # 创建数据增强器
        data_transform = Augmenter(base_transform, preprocess, n_views=args.batch_size)

        print(f"🔄 构建数据集: {args.test_set}")
        
        # 构建数据集 - 增加错误处理
        try:
            retrieved_dataset = build_dataset(f"{args.test_set}_retrieved", data_transform, args.data, mode='test')
            test_dataset = build_dataset(f"{args.test_set}_test", data_transform, args.data, mode='test')
        except Exception as e:
            print(f"❌ 构建数据集失败: {e}")
            print("💡 请检查数据目录结构和描述文件是否正确")
            return False
        
        print(f"📊 检索样本数量: {len(retrieved_dataset)}")
        print(f"📊 测试样本数量: {len(test_dataset)}")
        
        if len(retrieved_dataset) == 0 or len(test_dataset) == 0:
            print("❌ 数据集为空，请检查数据目录")
            return False
        
        # 创建数据加载器 - 减少并发避免问题
        print("🔄 创建数据加载器...")
        retrieved_loader = torch.utils.data.DataLoader(
            retrieved_dataset,
            batch_size=1, 
            shuffle=False,  
            num_workers=min(args.workers, 2),  # 减少worker数量
            pin_memory=True,
            timeout=60  # 增加超时
        )
        
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=1, 
            shuffle=False,  
            num_workers=min(args.workers, 2),  # 减少worker数量
            pin_memory=True,
            timeout=60  # 增加超时
        )
        
        print("✅ 数据加载器创建成功")
        
        # 开始提取多模态特征
        print("🚀 开始提取多模态特征...")
        success = pre_extract_multimodal_feature(retrieved_loader, test_loader, clip_model, args, text_generator)
        
        if success:
            print("🎉 MEC特征预提取完成!")
            return True
        else:
            print("❌ MEC特征预提取失败!")
            return False
            
    except Exception as e:
        print(f"❌ MEC特征预提取发生异常: {e}")
        import traceback
        traceback.print_exc()
        return False


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
    success = main_worker(args)
    
    # 根据执行结果设置退出码
    if success:
        print("✅ 特征预提取成功完成")
        exit(0)
    else:
        print("❌ 特征预提取失败")
        exit(1)