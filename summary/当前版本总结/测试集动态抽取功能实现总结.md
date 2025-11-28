# 测试集动态抽取功能实现完成

## 功能概述

成功实现了测试集的动态抽取功能，支持从JSON文件加载测试集数据，并根据`test_data_true_random`配置进行真随机或伪随机抽取，结果保存到不同的目录结构中。

## 核心特性

### 1. JSON文件加载
- **数据源**: 从`./experiments/<dataset>/images_split/images_test.json`加载测试集
- **格式支持**: 标准JSON格式`[class_name, class_id, [image_paths]]`
- **路径处理**: 自动处理相对路径转换

### 2. 分离式保存结构
- **真随机模式**: 保存到`./experiments/<dataset>/result/test_data_true_randomness/`
- **伪随机模式**: 保存到`./experiments/<dataset>/result/test_data_true_pseudorandom/`
- **文件命名**: `test_<percentage>.json`（如`test_1.json`, `test_5.json`）

### 3. 真随机支持
- **真随机模式**: 使用`/dev/random`硬件随机数生成器
- **伪随机模式**: 使用固定种子的伪随机数生成器
- **自动回退**: 真随机不可用时自动回退到伪随机

## 文件结构

### 重命名和重构
```
data/test_data.py → data/extract_from_testsets.py
```

### 输出目录结构
```
experiments/
├── bird200/
│   ├── result/
│   │   ├── test_data_true_randomness/      # 真随机抽取结果
│   │   │   └── test_1.json                 # 1%采样
│   │   └── test_data_true_pseudorandom/     # 伪随机抽取结果
│   │       ├── test_3.json                 # 3%采样
│   │       └── test_5.json                 # 5%采样
│   └── images_split/
│       └── images_test.json                 # 源测试集数据
└── dog120/
    ├── result/
    │   ├── test_data_true_randomness/
    │   │   └── test_2.json
    │   └── test_data_true_pseudorandom/
    └── images_split/
        └── images_test.json
```

## 使用方法

### 1. 命令行使用

```bash
# 真随机模式抽取1%
python data/extract_from_testsets.py --dataset bird200 --percentage 1 --true_random

# 伪随机模式抽取5%（固定种子42）
python data/extract_from_testsets.py --dataset bird200 --percentage 5

# 指定种子的伪随机模式
python data/extract_from_testsets.py --dataset dog120 --percentage 2 --seed 123
```

### 2. 集成使用

通过`discovering.py`自动调用：
- `test_data_true_random=True`: 使用真随机模式
- `test_data_true_random=False`: 使用伪随机模式

## 技术实现

### 1. JSON文件加载
```python
def load_test_set_from_json(json_path: str) -> List[Tuple[str, str]]:
    """从JSON文件加载测试集数据"""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    test_images = []
    for class_entry in data:
        class_name, class_id, image_paths = class_entry[0], class_entry[1], class_entry[2]
        for img_path in image_paths:
            test_images.append((img_path, class_name))
    
    return test_images
```

### 2. 真随机数生成
```python
from utils.true_random import get_true_random_generator

true_random = get_true_random_generator(use_blocking=True, fallback_to_pseudo=True)
if true_random.is_available():
    sampled = true_random.sample(image_files, num_samples)
else:
    sampled = random.sample(image_files, num_samples)
```

### 3. 按百分比采样
```python
# 计算每类采样数量（向上取整，至少1张）
total_images = len(image_files)
num_samples = max(1, math.ceil(total_images * test_percentage / 100.0))
num_samples = min(num_samples, total_images)
```

## 测试验证

### ✅ 已测试数据集
- **bird200**: 200类，2380张图像
  - 真随机1%: 200张图像
  - 伪随机3%: 200张图像
  - 伪随机5%: 200张图像

- **dog120**: 120类，4162张图像  
  - 真随机2%: 121张图像

### ✅ 功能验证
- JSON文件加载正常
- 真随机数生成器工作正常
- 伪随机数生成器工作正常
- 分离式目录保存正常
- JSON文件验证通过
- 路径导入问题已修复

## 配置集成

### discovering.py中的关键配置
```python
# 第40行
test_data_true_random = True  # 测试集采样是否实现真随机

# 自动调用测试集抽取
sampled_images = get_test_images_by_percentage(
    dataset_key, 
    args.test_percentage,
    seed=cfg.get('seed', 42),
    use_true_random=test_data_true_random
)
```

## 关键修复

### 1. 导入路径问题
- 创建了`utils/__init__.py`文件
- 修复了Python路径设置逻辑
- 确保项目根目录在sys.path中

### 2. 真随机数生成器
- 修复了模块导入问题
- 添加了异常处理和自动回退
- 确保真随机数生成器正常工作

### 3. JSON文件处理
- 从目录结构改为JSON文件加载
- 保持与发现集相同的JSON格式
- 添加了文件验证机制

## 示例输出

### 真随机模式
```
🔄 开始抽取测试集: bird200, 采样比例: 1.0%
🎲 真随机模式: 每次运行结果不同
从JSON文件加载测试集: ./experiments/bird200/images_split/images_test.json
✓ 加载完成: 200 个类别，共 2380 张图像
✓ 真随机数生成器已初始化: 使用 /dev/random
✓ 使用真随机数生成器进行采样
✓ 测试集已保存到: ./experiments/bird200/result/test_data_true_randomness/test_1.json
✓ JSON文件验证通过: 200 个类别
```

### 伪随机模式
```
🔄 开始抽取测试集: bird200, 采样比例: 3.0%
🔒 固定种子模式: seed=42
从JSON文件加载测试集: ./experiments/bird200/images_split/images_test.json
✓ 加载完成: 200 个类别，共 2380 张图像
✓ 测试集已保存到: ./experiments/bird200/result/test_data_true_pseudorandom/test_3.json
✓ JSON文件验证通过: 200 个类别
```

## 总结

✅ **成功完成测试集动态抽取功能的实现！**

### 主要成就：
1. **文件重构**: 将`test_data.py`改名为`extract_from_testsets.py`
2. **JSON加载**: 从目录结构改为JSON文件加载测试集数据
3. **分离保存**: 真随机和伪随机结果保存到不同目录
4. **真随机支持**: 修复真随机数生成器导入和路径问题
5. **完整验证**: 所有功能经过测试验证

### 技术亮点：
- 支持硬件真随机数生成（/dev/random）
- 自动回退机制确保系统稳定性
- JSON格式统一，便于管理和验证
- 路径处理兼容不同数据集结构

测试集抽取功能现已完全集成到系统中，支持根据`test_data_true_random`配置自动选择真随机或伪随机模式。
