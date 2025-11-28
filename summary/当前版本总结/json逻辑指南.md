# JSON数据集加载逻辑指南

## 概述

本指南说明如何使用JSON文件替代图像复制来优化数据集加载，节省内存空间并提高数据访问效率。

## 新增：JSON-based数据加载流程

### 问题背景
原有系统依赖目录结构加载测试数据，需要创建大量的`images_discovery_all_k`目录，占用大量存储空间且管理复杂。

### 解决方案
实现基于JSON文件的数据加载机制，通过以下改进：
1. **discovering.py**：新增`load_test_data_from_json`函数，支持从JSON文件加载测试数据
2. **脚本适配**：修改所有shell脚本，使用JSON文件路径替代目录路径
3. **智能检测**：自动检测文件类型（JSON vs 目录），选择合适的加载方式
4. **批量复制**：使用`copy_datasets_json.py`脚本批量处理所有数据集

### 核心实现

#### 1. JSON数据加载函数
```python
def load_test_data_from_json(json_file_path: str, dataset_name: str = None) -> dict:
    """从JSON文件加载测试数据"""
    print(f"从JSON文件加载测试数据: {json_file_path}")
    
    if not os.path.exists(json_file_path):
        raise FileNotFoundError(f"测试数据JSON文件不存在: {json_file_path}")
    
    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    test_samples = defaultdict(list)
    
    # 获取数据集名称用于类别名标准化
    from data.class_name_mapper import standardize_test_class_name
    
    for class_entry in data:
        if len(class_entry) >= 3:
            class_name, class_id, image_paths = class_entry[0], class_entry[1], class_entry[2]
            
            # 标准化类别名称
            if dataset_name:
                standardized_class_name = standardize_test_class_name(class_name, dataset_name)
            else:
                standardized_class_name = class_name
            
            # 添加所有图像路径
            for img_path in image_paths:
                test_samples[standardized_class_name].append(img_path)
    
    total_images = sum(len(paths) for paths in test_samples.values())
    print(f"✓ 从JSON文件加载 {len(test_samples)} 个类别，共 {total_images} 张图像")
    
    return dict(test_samples)
```

#### 2. 脚本适配示例
```bash
# 原有方式（目录）
TEST_DATA_DIR="./experiments/car196/images_discovery_all_1"

# 新方式（JSON文件）
TEST_DATA_JSON="./experiments/car196/images_split/images_discovery_all_1.json"

# 脚本中的条件判断
if [ "${USE_TEST_DATA}" = "true" ]; then
    # 使用测试集
    CMD="python discovering.py --use_test_data --test_percentage=${TEST_PERCENTAGE} ..."
else
    # 使用discovery集（JSON文件）
    CMD="python discovering.py --test_data_dir=${TEST_DATA_JSON} ..."
fi
```

## 支持的数据集

### 已适配数据集列表
截至当前版本，以下数据集已完成JSON化适配：

| 数据集键名 | 数据目录 | 实验目录 | 状态 |
|-----------|----------|----------|------|
| car | car_196 | car196 | ✅ 完成 |
| pet | pet_37 | pet37 | ✅ 完成 |
| bird | CUB_200_2011 | bird200 | ✅ 完成 |
| dog | dogs_120 | dog120 | ✅ 完成 |
| flower | flowers_102 | flower102 | ✅ 完成 |
| food | food_101 | food101 | ✅ 完成 |
| dtd | dtd | dtd47 | ✅ 完成 |
| eurosat | eurosat | eurosat10 | ✅ 完成 |
| aircraft | fgvc_aircraft | aircraft100 | ✅ 完成 |
| caltech101 | caltech101 | caltech101 | ✅ 完成 |
| caltech256 | caltech256 | caltech256 | ✅ 完成 |
| imagenet_a | ImageNet_A | imagenet_a200 | ✅ 完成 |
| imagenet_r | ImageNet_R | imagenet_r200 | ✅ 完成 |

### 特殊路径处理
- **CUB_200_2011**：双层嵌套目录结构，自动处理`CUB_200_2011/CUB_200_2011/images_split`
- **SUN397**：使用传统目录结构，暂未适配JSON格式

## JSON复制脚本使用指南

### 脚本功能
`./data/copy_datasets_json.py`脚本提供以下功能：
1. 批量复制所有数据集的JSON文件
2. 单独处理指定数据集
3. 自动路径修正（相对路径转换）
4. 特殊目录结构处理

### 使用方法

#### 1. 复制所有数据集
```bash
python data/copy_datasets_json.py
```

#### 2. 复制指定数据集
```bash
python data/copy_datasets_json.py --dataset pet
python data/copy_datasets_json.py --dataset caltech101
```

#### 3. 查看帮助
```bash
python data/copy_datasets_json.py --help
```

### 脚本参数说明
- `--dataset`：指定要处理的数据集（可选，默认处理所有）
- `--config`：配置文件路径（默认：`configs/datasets_list.yml`）

## 目录结构

### 源数据集结构
```
datasets/[dataset_name]/images_split/
├── images.json                          # 所有图片的类别和路径
├── images_test.json                     # 测试集图片
├── images_train.json                    # 训练集图片
├── images_val.json                      # 验证集图片（如果存在）
├── split_[dataset_name]_images.json     # 训练测试划分
├── images_discovery_all.json            # 默认发现集（每类3张）
├── images_discovery_all_1.json          # 每类1张发现集
├── images_discovery_all_2.json          # 每类2张发现集
├── ...
├── images_discovery_all_10.json         # 每类10张发现集
└── images_discovery_random.json         # 随机数量发现集
```

### 实验目录结构
```
experiments/[experiment_dir]/images_split/
├── images.json                          # 修改后的路径（相对于项目根目录）
├── images_test.json
├── images_train.json
├── images_val.json
├── split_[dataset_name]_images.json
├── images_discovery_all.json
├── images_discovery_all_1.json
├── images_discovery_all_2.json
├── ...
├── images_discovery_all_10.json
└── images_discovery_random.json
```

## JSON文件格式

### 标准格式
```json
[
  [
    "类别名称",
    类别ID,
    [
      "./datasets/dataset_name/path/to/image1.jpg",
      "./datasets/dataset_name/path/to/image2.jpg"
    ]
  ],
  [
    "另一个类别",
    类别ID,
    [
      "./datasets/dataset_name/path/to/image3.jpg"
    ]
  ]
]
```

### 路径转换规则
- **原始路径**：`images/category/image.jpg`（相对于数据集目录）
- **转换后路径**：`./datasets/dataset_name/images/category/image.jpg`（相对于项目根目录）

## 配置文件映射

### datasets_list.yml配置
```yaml
dataset_mapping:
  pet:
    full_name: "pet37"
    num_classes: 37
    config_file: "pet37_all.yml"
    data_dir: "pet_37"              # 数据集目录名
    experiment_dir: "pet37"         # 实验目录名
    stats_file: "pet37/knowledge_base/stats.json"
  
  bird:
    full_name: "bird200"
    num_classes: 200
    config_file: "bird200_all.yml"
    data_dir: "CUB_200_2011"
    experiment_dir: "bird200"
    special_subdir: "CUB_200_2011"  # 特殊子目录
    stats_file: "bird200/knowledge_base/stats.json"
```

## 运行脚本适配

### 修改的脚本列表
1. `run_fast_slow.sh`：快慢思考系统脚本
2. `run_pipeline.sh`：流水线脚本
3. `run_discovery.sh`：发现集脚本
4. `run_build_knowledge_base.sh`：知识库构建脚本

### 关键修改点
- 变量名从`TEST_DATA_DIR`改为`TEST_DATA_JSON`
- 文件存在性检查改为JSON文件
- 命令行参数传递JSON文件路径

## 错误处理和调试

### 常见错误及解决方案

#### 1. FileNotFoundError
**错误**：`测试数据JSON文件不存在`
**解决**：
- 检查JSON文件是否已复制到experiments目录
- 运行`python data/copy_datasets_json.py --dataset [dataset_name]`

#### 2. 路径不匹配
**错误**：图像文件路径错误
**解决**：
- 检查JSON文件中的路径格式
- 确认路径以`./datasets/`开头
- 验证实际文件是否存在

#### 3. 特殊数据集路径
**错误**：CUB_200_2011路径错误
**解决**：
- 确认配置文件中有`special_subdir`字段
- 检查双层目录结构

### 调试技巧

#### 1. 检查JSON文件内容
```bash
head -10 experiments/pet37/images_split/images_discovery_all_1.json
```

#### 2. 验证文件存在性
```bash
ls -la ./datasets/pet_37/images/Abyssinian_168.jpg
```

#### 3. 测试单个数据集
```bash
python data/copy_datasets_json.py --dataset pet
python discovering.py --test_data_dir=./experiments/pet37/images_split/images_discovery_all_1.json --mode=fast_slow ...
```

## 性能优化

### 内存使用
- **JSON方式**：仅存储路径信息，内存占用极小
- **目录方式**：需要创建大量目录和软链接，占用较多存储空间

### 加载速度
- **JSON方式**：直接读取路径，无需文件系统遍历
- **目录方式**：需要遍历目录结构，速度较慢

## 扩展新数据集

### 添加新数据集步骤

#### 1. 准备JSON文件
在数据集目录下创建`images_split`目录和相应的JSON文件：
```bash
mkdir -p datasets/new_dataset/images_split
# 创建各种JSON文件...
```

#### 2. 更新配置文件
在`configs/datasets_list.yml`中添加新数据集：
```yaml
new_dataset:
  full_name: "new_dataset100"
  num_classes: 100
  config_file: "new_dataset_all.yml"
  data_dir: "new_dataset"
  experiment_dir: "new_dataset"
  stats_file: "new_dataset/knowledge_base/stats.json"
```

#### 3. 复制JSON文件
```bash
python data/copy_datasets_json.py --dataset new_dataset
```

#### 4. 测试运行
```bash
bash scripts/run_fast_slow.sh new_dataset
```

### 自动化脚本
可以创建批量处理脚本：
```bash
#!/bin/bash
datasets=("pet" "car" "bird" "dog" "flower")
for dataset in "${datasets[@]}"; do
    echo "Processing $dataset..."
    python data/copy_datasets_json.py --dataset $dataset
done
```

## 最佳实践

### 1. 路径一致性
- 确保所有JSON文件使用统一的路径格式
- 使用相对路径而非绝对路径
- 定期验证路径的有效性

### 2. 配置管理
- 集中管理数据集配置信息
- 使用YAML文件存储映射关系
- 定期更新配置文件

### 3. 错误处理
- 添加完善的错误检查机制
- 提供详细的错误信息
- 实现自动重试机制

### 4. 文档维护
- 及时更新本指南
- 记录新增数据集的适配过程
- 维护配置文件的版本历史

## 版本历史

- **v1.0**：初始版本，支持car数据集
- **v2.0**：添加pet数据集支持
- **v3.0**：批量适配13个数据集，支持特殊路径处理
- **v3.1**：完善文档，添加错误处理指南

---

**注意**：本指南会随着系统更新持续维护，请定期查看最新版本。
