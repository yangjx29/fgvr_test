# JSON数据集加载逻辑指南

## 概述

本指南说明如何使用JSON文件替代图像复制来优化数据集加载，节省内存空间并提高数据访问效率。

## 新增：JSON-based数据加载流程

### 问题背景
原有系统依赖目录结构加载测试数据，需要创建大量的`images_discovery_all_k`目录，占用大量存储空间且管理复杂。

### 解决方案
实现基于JSON文件的数据加载机制，通过以下改进：
1. ** discovering.py**：新增`load_test_data_from_json`函数，支持从JSON文件加载测试数据
2. **脚本适配**：修改所有shell脚本，使用JSON文件路径替代目录路径
3. **智能检测**：自动检测文件类型（JSON vs 目录），选择合适的加载方式

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

#### 2. 智能检测逻辑
```python
def prepare_test_samples(cfg, args):
    # 检查是JSON文件还是目录
    if args.test_data_dir.endswith('.json'):
        # 从JSON文件加载
        print("检测到JSON文件，使用JSON格式加载测试数据")
        test_samples = load_test_data_from_json(args.test_data_dir, cfg.get('dataset_name'))
    else:
        # 从目录加载（原有逻辑）
        print("检测到目录，使用目录格式加载测试数据")
        # ... 原有目录加载逻辑
```

#### 3. 脚本修改示例（run_fast_slow.sh）
```bash
# 原有配置
TEST_DATA_DIR="./datasets/${DATASET_DIR}/images_discovery_all_${TEST_DATA_SUFFIX}"

# 新配置
TEST_DATA_JSON="./experiments/${EXPERIMENT_DIR}/images_split/images_discovery_all_${TEST_DATA_SUFFIX}.json"

# 检查逻辑
if [ ! -f "${TEST_DATA_JSON}" ]; then
    print_error "测试数据JSON文件不存在: ${TEST_DATA_JSON}"
    print_info "请确保JSON文件已复制到experiments目录"
    exit 1
fi

# 命令行参数
--test_data_dir=${TEST_DATA_JSON}
```

### 完整工作流程

#### 步骤1：数据集JSON构建
```bash
# 1. 删除现有的images_split目录
rm -rf datasets/car_196/images_split
rm -rf experiments/car196/images_split

# 2. 重新构建JSON文件
cd datasets/car_196
python build_car196_json.py

# 3. 验证JSON文件
# 自动验证：文件完整性、数据一致性、互斥性、发现集来源
```

#### 步骤2：JSON文件复制
```bash
# 运行复制脚本
python data/copy_datasets_json.py

# 自动完成：
# - 复制16个JSON文件到experiments目录
# - 修改路径前缀为./datasets/car_196/
# - 验证复制完整性
```

#### 步骤3：脚本运行
```bash
# 现在可以正常运行，系统会自动检测并使用JSON文件
bash scripts/run_fast_slow.sh car
```

### 兼容性说明

#### 向后兼容
- 系统仍支持原有的目录加载方式
- 通过文件扩展名自动检测（.json vs 目录）
- 无需修改现有配置文件

#### 配置参数
- `scripts/config.yaml`中的`use_test_data`控制使用测试集还是发现集
- `test_data_suffix`控制使用哪个发现集文件（1-10, random）
- 无需新增配置参数

#### 错误处理
```bash
[ERROR] 测试数据JSON文件不存在: ./experiments/car196/images_split/images_discovery_all_1.json
[INFO] 请确保JSON文件已复制到experiments目录
```

### 性能优势

#### 1. 存储空间
- **目录方式**：需要复制实际图像文件，占用大量磁盘空间
- **JSON方式**：仅存储路径信息，空间占用极小

#### 2. 加载速度
- **目录方式**：需要遍历目录结构，I/O开销大
- **JSON方式**：一次性加载，解析速度快

#### 3. 维护性
- **目录方式**：文件管理复杂，容易出错
- **JSON方式**：集中管理，易于验证和调试

### 扩展指南

#### 新数据集JSON化
1. 参考car196的`build_car196_json.py`脚本
2. 根据数据集特点调整构建逻辑
3. 运行验证脚本确保正确性

#### 功能扩展
- 支持更多的发现集采样策略
- 添加JSON文件的压缩存储
- 实现增量更新机制

---

**实现状态**: ✅ 已完成car196数据集  
**测试状态**: ✅ 正在运行测试  
**文档更新**: 2025-11-27

## 目录结构

### 原始数据集结构
```
./datasets/[dataset_name]/images_split/
├── images.json                          # 所有图片的 类别，相对路径 json文件
├── images_test.json           # 所有测试集图片的 类别，相对路径 json文件
├── images_val.json           # (如果原本存在验证集划分文件就创建，否则不需要)所有验证集图片的 类别，相对路径 json文件
├── images_train.json          # 所有训练集图片的 类别，相对路径 json文件
├── split_[dataset_name]_196_images.json      # 训练集和测试集的划分json文件
├── images_discovery_all.json             # 默认发现集的类别，与images_discovery_all_3.json相同，相对路径 json文件（每类3张）
├── images_discovery_all_1.json           # 每类从训练集中随机抽取的1张图像的 类别，相对路径 json文件
├── images_discovery_all_2.json           # 每类从训练集中随机抽取的2张图像的 类别，相对路径 json文件
├── images_discovery_all_3.json           # 每类从训练集中随机抽取的3张图像的 类别，相对路径 json文件
├── ...
├── images_discovery_all_10.json          # 每类从训练集中随机抽取的10张图像的 类别，相对路径 json文件
└── images_discovery_random.json          # 每类包含从训练集中随机抽取的c张图像(c服从长尾分布)的 类别，相对路径 json文件
```

### 实验目录结构
```
./experiments/[dataset_name]/images_split/
├── images.json                          # 所有图片的 类别，相对路径 json文件
├── images_test.json           # 所有测试集图片的 类别，相对路径 json文件
├── images_val.json           # (如果原本存在验证集划分文件就创建，否则不需要)所有验证集图片的 类别，相对路径 json文件
├── images_train.json          # 所有训练集图片的 类别，相对路径 json文件
├── split_[dataset_name]_196_images.json      # 训练集和测试集的划分json文件
├── images_discovery_all.json             # 默认发现集的类别，与images_discovery_all_3.json相同，相对路径 json文件（每类3张）
├── images_discovery_all_1.json           # 每类从训练集中随机抽取的1张图像的 类别，相对路径 json文件
├── images_discovery_all_2.json           # 每类从训练集中随机抽取的2张图像的 类别，相对路径 json文件
├── images_discovery_all_3.json           # 每类从训练集中随机抽取的3张图像的 类别，相对路径 json文件
├── ...
├── images_discovery_all_10.json          # 每类从训练集中随机抽取的10张图像的 类别，相对路径 json文件
└── images_discovery_random.json          # 每类包含从训练集中随机抽取的c张图像(c服从长尾分布)的 类别，相对路径 json文件
```

## 数据集配置映射

### datasets_list.yml配置
```yaml
dataset_mapping:
  car:
    full_name: "car196"
    num_classes: 196
    config_file: "car196_all.yml"
    data_dir: "car_196"
    experiment_dir: "car196"  # experiments下的目录名
    stats_file: "car196/knowledge_base/stats.json"
```

## 实施步骤

### 1. 复制JSON文件脚本

使用 `./data/copy_datasets_json.py` 脚本：

```python
# 脚本功能：
# 1. 从datasets/[dataset_name]/images_split/读取原始JSON文件
# 2. 修改图像路径为相对于当前工作目录的路径
# 3. 复制到experiments/[experiment_dir]/images_split/
```

运行方式：
```bash
cd /home/hdl/project/fgvr_test_new
python data/copy_datasets_json.py
```

### 2. 路径修改规则

**原始路径格式**：
```
"images/pitted/00001.jpg"
```

**修改后路径格式**：
```
"./datasets/car_196/images/pitted/00001.jpg"
```

### 3. JSON文件格式

#### images.json格式
```json
[
  [
    "类别名称",
    类别ID,
    [
      "图像路径1",
      "图像路径2",
      ...
    ]
  ],
  ...
]
```

#### split_[dataset_name]_images.json格式
```json
{
  "train": [
    ["类别名称", 类别ID, "图像路径"],
    ...
  ],
  "test": [
    ["类别名称", 类别ID, "图像路径"],
    ...
  ],
  "val": [
    ["类别名称", 类别ID, "图像路径"],
    ...
  ]
}
```

## 数据加载逻辑修改

### 1. Discovery数据集加载

**原始方式**：
```python
class CarDiscovery196:
    def __init__(self, root, folder_suffix=''):
        img_root = os.path.join(root, f'images_discovery_all{folder_suffix}')
        # 从目录读取图像文件
```

**JSON方式**：
```python
class CarDiscovery196:
    def __init__(self, cfg, folder_suffix=''):
        # 从配置文件读取路径
        dataset_config = get_dataset_config('car')
        experiment_dir = dataset_config['experiment_dir']
        experiments_root = load_dataset_config().get('experiments_root', './experiments')
        
        # 加载JSON文件
        json_path = f'{experiments_root}/{experiment_dir}/images_split/images_discovery_all{folder_suffix}.json'
        with open(json_path, 'r', encoding='utf-8') as f:
            self.json_data = json.load(f)
        
        # 解析JSON数据
        for class_info in self.json_data:
            class_name, class_id, image_paths = class_info[0], class_info[1], class_info[2]
            # 处理数据...
```

### 2. Test数据集加载

**原始方式**：
```python
class CarDataset(Dataset):
    def __init__(self, root, train=True, transform=None, limit=0):
        # 从MAT文件读取标注
        # 从目录读取图像文件
```

**JSON方式**：
```python
class CarTestDataset(Dataset):
    def __init__(self, cfg, transform=None, limit=0):
        # 从JSON文件读取数据和标注
        json_path = f'{experiments_root}/{experiment_dir}/images_split/images_test.json'
        with open(json_path, 'r', encoding='utf-8') as f:
            self.json_data = json.load(f)
```

## 具体加载逻辑

根据数据集加载逻辑文档，具体加载路径如下：

- **测试集图片**: `./experiments/[dataset_name]/images_split/images_test.json`
- **验证集图片**: `./experiments/[dataset_name]/images_split/images_val.json` (如果存在)
- **训练集图片**: `./experiments/[dataset_name]/images_split/images_train.json`
- **发现集**: `./experiments/[dataset_name]/images_split/images_discovery_all.json`
- **发现集变体**: `./experiments/[dataset_name]/images_split/images_discovery_all_k.json` (k=1..10)
- **随机发现集**: `./experiments/[dataset_name]/images_split/images_discovery_random.json`

## 扩展到其他数据集

要为其他数据集实现相同的JSON加载逻辑：

### 1. 更新copy_datasets_json.py脚本

在脚本的main函数中添加其他数据集：

```python
# 添加其他数据集配置
for dataset_key, dataset_config in config['dataset_mapping'].items():
    if dataset_key in ['dog', 'bird', 'flower']:  # 选择要处理的数据集
        datasets.append({
            'name': dataset_config['data_dir'],
            'source_dir': f'/home/hdl/project/fgvr_test_new/datasets/{dataset_config["data_dir"]}/images_split',
            'target_dir': f'/home/hdl/project/fgvr_test_new/{experiments_root}/{dataset_config["experiment_dir"]}/images_split'
        })
```

### 2. 修改对应数据集文件

对每个数据集的数据文件（如`data/dog120.py`, `data/bird200.py`等）进行类似修改：

1. 添加配置加载函数
2. 修改Discovery类使用JSON加载
3. 修改Test类使用JSON加载
4. 更新build函数

### 3. 配置文件映射

确保`configs/datasets_list.yml`中包含所有数据集的正确映射信息。

## 优势

1. **节省内存**: 不需要复制图像文件，只存储路径信息
2. **统一管理**: 所有数据集使用相同的JSON格式
3. **灵活配置**: 通过YAML配置文件管理数据集路径
4. **易于扩展**: 可以轻松添加新的数据集支持

## 注意事项

1. 确保JSON文件路径正确指向原始图像位置
2. 保持JSON文件格式一致性
3. 测试数据加载功能确保路径正确
4. 定期验证JSON文件与原始数据的一致性

## 测试验证

使用以下代码测试JSON加载：

```python
from data.car196 import build_car196_discovery, build_car196_test

cfg = {
    'experiment_dir': 'car196',
    'image_size': 224,
    'batch_size': 32,
    'num_workers': 4
}

# 测试发现集加载
discovery = build_car196_discovery(cfg, folder_suffix='_3')
print(f'Discovery dataset: {len(discovery)} samples')

# 测试测试集加载
test_loader = build_car196_test(cfg)
print(f'Test dataset: {len(test_loader.dataset)} samples')
```
