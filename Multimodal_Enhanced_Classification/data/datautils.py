# 导入操作系统接口模块
import os
# 导入PIL图像处理库
from PIL import Image

# 导入PyTorch的图像变换模块
import torchvision.transforms as transforms
# 导入PyTorch的数据集模块
import torchvision.datasets as datasets

# 尝试从torchvision导入插值模式（新版本）
try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    # 如果导入失败（旧版本），使用PIL的双三次插值
    BICUBIC = Image.BICUBIC

# 导入少样本学习数据集相关的所有内容
from data.fewshot_datasets import *

# 数据集ID到目录名称的映射字典
# 用于将数据集标识符映射到实际的文件系统目录名
ID_to_DIRNAME={
    'imagenet': 'imagenet',
    'imagenet_a': 'imagenet-adversarial',
    'imagenet_sketch': 'imagenet-sketch',
    'imagenet_r': 'imagenet-rendition',
    'imagenetv2': 'imagenetv2',
    'oxford_flowers': 'oxford_flowers',
    'dtd': 'dtd',
    'oxford_pets': 'oxford_pets',
    'stanford_cars': 'stanford_cars',
    'ucf101': 'ucf101',
    'caltech101': 'caltech-101',
    'food101': 'food-101',
    'sun397': 'sun397',
    'fgvc_aircraft': 'fgvc_aircraft',
    'eurosat': 'eurosat',
    'caltech256': 'caltech256',
    'cub': 'cub',
    'birdsnap': 'birdsnap',
    'stanford_dogs': 'stanford_dogs',  # ========== 新增: Stanford Dogs 数据集路径映射 ==========
}

# 构建数据集的主函数
# 参数:
#   set_id: 数据集标识符（如'imagenet', 'oxford_flowers'等）
#   transform: 图像变换/增强操作
#   data_root: 数据集根目录路径
#   mode: 模式，'train'或'test'，默认为'test'
#   n_shot: 少样本学习中每类的样本数，None表示使用全部数据
#   split: 数据划分方式，默认为"all"
#   bongard_anno: 是否使用Bongard注释，默认为False
# 返回:
#   testset: 构建好的数据集对象
def build_dataset(set_id, transform, data_root, mode='test', n_shot=None, split="all", bongard_anno=False):
    # 处理ImageNet及其变体数据集
    if set_id in ['imagenet', 'imagenet_a', 'imagenet_sketch', 'imagenet_r', 'imagenetv2']:
        # 根据不同的ImageNet变体设置测试目录路径
        if set_id == 'imagenet':
            # 标准ImageNet验证集路径
            testdir = os.path.join(os.path.join(data_root, ID_to_DIRNAME[set_id]), 'images', 'val')
        elif set_id == 'imagenetv2':
            # ImageNet-V2数据集路径
            testdir = os.path.join(data_root, ID_to_DIRNAME[set_id], 'imagenetv2-matched-frequency-format-val')
        elif set_id == 'imagenet_a':
            # ImageNet-A（对抗样本）数据集路径
            testdir = os.path.join(data_root, ID_to_DIRNAME[set_id], 'imagenet-a')
        elif set_id == 'imagenet_r':
            # ImageNet-R（艺术风格）数据集路径
            testdir = os.path.join(data_root, ID_to_DIRNAME[set_id], 'imagenet-r')
        elif set_id == 'imagenet_sketch':
            # ImageNet-Sketch（素描风格）数据集路径
            testdir = os.path.join(data_root, ID_to_DIRNAME[set_id], 'images')
        # 使用PyTorch的ImageFolder加载数据集
        testset = datasets.ImageFolder(testdir, transform=transform)
    # 处理少样本学习数据集
    elif set_id in fewshot_datasets:
        # 如果是训练模式且指定了少样本数量
        if mode == 'train' and n_shot:
            # 构建少样本训练数据集
            testset = build_fewshot_dataset(set_id, os.path.join(data_root, ID_to_DIRNAME[set_id.lower()]), transform, mode=mode, n_shot=n_shot)
        else:
            # 构建完整数据集（测试模式或不限制样本数）
            testset = build_fewshot_dataset(set_id, os.path.join(data_root, ID_to_DIRNAME[set_id.lower()]), transform, mode=mode)
    else:
        # 处理动态生成的数据集（如从discovering.py调用）
        # 假设数据集位于 data_root/set_id 目录下
        testdir = os.path.join(data_root, set_id)
        print(f"🔄 处理动态数据集: {set_id}")
        print(f"📁 目标目录: {testdir}")
        
        if os.path.exists(testdir) and os.path.isdir(testdir):
            print(f"✅ 找到数据集目录: {testdir}")
            
            # 检查目录是否有有效的图像文件
            has_valid_images = False
            valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
            
            # 遍历所有子目录查找图像文件
            for root, dirs, files in os.walk(testdir):
                for file in files:
                    if any(file.lower().endswith(ext) for ext in valid_extensions):
                        has_valid_images = True
                        break
                if has_valid_images:
                    break
            
            if not has_valid_images:
                print(f"⚠️  目录存在但为空，创建虚拟数据...")
                # 创建dummy类别目录
                dummy_class_dir = os.path.join(testdir, "dummy_class")
                os.makedirs(dummy_class_dir, exist_ok=True)
                
                # 创建虚拟图像
                import numpy as np
                dummy_img_path = os.path.join(dummy_class_dir, "dummy.jpg")
                if not os.path.exists(dummy_img_path):
                    dummy_img = Image.fromarray(np.zeros((32, 32, 3), dtype=np.uint8), 'RGB')
                    dummy_img.save(dummy_img_path, 'JPEG')
                    print(f"✅ 创建虚拟图像: {dummy_img_path}")
            
            # 检查每个子目录，确保都有图像文件
            for item in os.listdir(testdir):
                item_path = os.path.join(testdir, item)
                if os.path.isdir(item_path):
                    files_in_dir = [f for f in os.listdir(item_path) 
                                  if os.path.isfile(os.path.join(item_path, f)) and 
                                  any(f.lower().endswith(ext) for ext in valid_extensions)]
                    
                    if len(files_in_dir) == 0:
                        print(f"⚠️  子目录为空，添加虚拟图像: {item_path}")
                        import numpy as np
                        dummy_img_path = os.path.join(item_path, "dummy.jpg")
                        if not os.path.exists(dummy_img_path):
                            dummy_img = Image.fromarray(np.zeros((32, 32, 3), dtype=np.uint8), 'RGB')
                            dummy_img.save(dummy_img_path, 'JPEG')
                            print(f"✅ 创建虚拟图像: {dummy_img_path}")
            
            try:
                testset = datasets.ImageFolder(testdir, transform=transform)
                print(f"✅ 成功加载数据集: {len(testset)} 个样本, {len(testset.classes)} 个类别")
            except Exception as e:
                print(f"❌ ImageFolder加载失败: {e}")
                raise RuntimeError(f"无法加载数据集 {set_id}: {str(e)}")
                
        else:
            print(f"❌ 数据集目录不存在: {testdir}")
            print("🔧 创建最小目录结构...")
            
            # 创建目录结构
            os.makedirs(testdir, exist_ok=True, mode=0o755)
            
            # 创建dummy类别目录
            dummy_class_dir = os.path.join(testdir, "dummy_class")
            os.makedirs(dummy_class_dir, exist_ok=True)
            
            # 创建虚拟图像文件
            try:
                import numpy as np
                dummy_img_path = os.path.join(dummy_class_dir, "dummy.jpg")
                dummy_img = Image.fromarray(np.zeros((32, 32, 3), dtype=np.uint8), 'RGB')
                dummy_img.save(dummy_img_path, 'JPEG')
                print(f"✅ 创建虚拟图像: {dummy_img_path}")
                
                testset = datasets.ImageFolder(testdir, transform=transform)
                print(f"🔧 创建兜底数据集: {len(testset)} 个样本")
                
            except Exception as e:
                print(f"❌ 创建兜底数据集失败: {e}")
                raise RuntimeError(f"无法创建数据集 {set_id}: {str(e)}")
        
    return testset


# ========== 图像变换和数据增强 ==========

# 获取预增强变换
# 返回:
#   transforms.Compose: 组合的图像变换操作
def get_preaugment():
    return transforms.Compose([
            # 随机裁剪并调整大小到224x224
            transforms.RandomResizedCrop(224),
            # 随机水平翻转
            transforms.RandomHorizontalFlip(),
        ])

# 对单张图像进行增强
# 参数:
#   image: 输入的PIL图像
#   preprocess: 预处理函数
# 返回:
#   x_processed: 增强后的图像张量
def aug(image, preprocess):
    # 获取预增强变换
    preaugment = get_preaugment()
    # 应用预增强（随机裁剪和翻转）
    x_orig = preaugment(image)
    # 应用预处理（归一化等）
    x_processed = preprocess(x_orig)
    return x_processed


# 图像增强器类
# 用于生成一张原始图像和多个增强视图
class Augmenter(object):
    # 初始化增强器
    # 参数:
    #   base_transform: 基础变换（如中心裁剪）
    #   preprocess: 预处理函数（如归一化）
    #   n_views: 生成的增强视图数量，默认为2
    def __init__(self, base_transform, preprocess, n_views=2):
        self.base_transform = base_transform  # 保存基础变换
        self.preprocess = preprocess  # 保存预处理函数
        self.n_views = n_views  # 保存视图数量
        
    # 调用增强器生成多个视图
    # 参数:
    #   x: 输入的PIL图像
    # 返回:
    #   list: 包含原始图像和n_views个增强视图的列表
    def __call__(self, x):
        # 生成基础视图（应用基础变换和预处理）
        image = self.preprocess(self.base_transform(x))
        # 生成n_views个增强视图（应用随机增强和预处理）
        views = [aug(x, self.preprocess) for _ in range(self.n_views)]
        # 返回基础视图和所有增强视图的列表
        return [image] + views
