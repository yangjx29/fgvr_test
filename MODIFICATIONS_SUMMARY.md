# 数据集路径修复和自动化检查功能修改总结

## 修改概述

本次修改主要解决了两个问题：
1. 将 `copy_datasets_json.py` 中的绝对路径改为相对路径
2. 在 `discovering.py` 中添加自动检查和生成JSON文件的功能

## 1. copy_datasets_json.py 修改

### 修改内容
- 将所有绝对路径改为相对路径，使脚本更加便携
- 修改的路径包括：
  - 源数据目录路径：`/home/hdl/project/fgvr_test_new/datasets/` → `./datasets/`
  - 目标实验目录路径：`/home/hdl/project/fgvr_test_new/experiments/` → `./experiments/`
  - 配置文件路径：`/home/hdl/project/fgvr_test_new/configs/` → `./configs/`

### 具体修改
```python
# 修改前
source_dir = f'/home/hdl/project/fgvr_test_new/datasets/{data_dir}/{dataset_config["special_subdir"]}/images_split'
source_dir = f'/home/hdl/project/fgvr_test_new/datasets/{data_dir}/images_split'
'target_dir': f'/home/hdl/project/fgvr_test_new/{experiments_root}/{dataset_config["experiment_dir"]}/images_split'
parser.add_argument('--config', type=str, default='/home/hdl/project/fgvr_test_new/configs/datasets_list.yml')

# 修改后
source_dir = f'./datasets/{data_dir}/{dataset_config["special_subdir"]}/images_split'
source_dir = f'./datasets/{data_dir}/images_split'
'target_dir': f'./{experiments_root}/{dataset_config["experiment_dir"]}/images_split'
parser.add_argument('--config', type=str, default='./configs/datasets_list.yml')
```

## 2. discovering.py 修改

### 新增功能
添加了 `check_and_generate_json_files()` 函数，用于：
1. 检查实验目录下的关键JSON文件是否存在
2. 如果缺失，自动运行 `copy_datasets_json.py` 生成JSON文件
3. 支持超时控制和错误处理

### 关键JSON文件检查
检查以下关键文件：
- `images_test.json` - 测试集数据
- `images_discovery_all_1.json` - 发现集数据

### 自动生成流程
1. 检测缺失的JSON文件
2. 显示友好的提示信息
3. 自动调用 `copy_datasets_json.py` 脚本
4. 验证文件是否成功生成
5. 提供错误处理和手动运行提示

### 集成位置
在 `discovering.py` 主函数中，设置当前数据集后立即调用检查：
```python
# 设置当前数据集
dataset_name = cfg.get('dataset_name', 'dog')
set_current_dataset(dataset_name)

# 检查并自动生成JSON文件（如果需要）
check_and_generate_json_files(dataset_name)
```

## 3. 支持的数据集

通过 `configs/datasets_list.yml` 配置文件，支持以下数据集的自动检查：
- `dog` (dogs_120)
- `bird` (bird200/CUB_200_2011)
- `aircraft` (aircraft100/fgvc_aircraft)
- `flower` (flower102/flowers_102)
- `pet` (pet37)
- `car` (car196)
- 以及其他配置文件中定义的数据集

## 4. 路径修复逻辑

针对不同数据集的目录结构，`copy_datasets_json.py` 中的 `fix_path` 函数实现了以下路径修复：

### caltech101
- `images/` → `101_ObjectCategories/`

### flowers_102
- `images/category/image.jpg` → `jpg/image.jpg` (移除类别子目录)

### food_101
- `images/` → `jpg/`

### CUB_200_2011
- `images/` → `CUB_200_2011/images/`

### fgvc_aircraft
- `images/category/image.jpg` → `images/image.jpg` (移除类别子目录)

### dogs_120
- `images/category/image.jpg` → `Images/nXXXXXX-category/image.jpg` (智能匹配实际目录名)

## 5. 使用示例

### 手动运行JSON复制
```bash
cd /home/hdl/project/fgvr_test_new
python data/copy_datasets_json.py --dataset dog
python data/copy_datasets_json.py --dataset bird
python data/copy_datasets_json.py --dataset aircraft
```

### 自动检查和生成
```bash
cd /home/hdl/project/fgvr_test_new
source activate finer_dynamic

# 运行任何实验模式，会自动检查JSON文件
python discovering.py --mode=fastonly --config_file_env=./configs/env_machine.yml --config_file_expt=./configs/expts/dog120_all.yml --use_test_data --test_percentage=1
```

## 6. 错误处理

- 超时控制：5分钟超时避免长时间等待
- 错误提示：提供清晰的手动运行指令
- 文件验证：生成后再次验证文件是否存在
- 异常捕获：处理各种运行时错误

## 7. 优势

1. **便携性**：使用相对路径，可以在不同环境中运行
2. **自动化**：无需手动检查和生成JSON文件
3. **统一性**：所有数据集使用相同的检查逻辑
4. **可靠性**：包含完整的错误处理和验证机制
5. **用户友好**：提供清晰的提示信息

## 8. 测试验证

已测试以下数据集的自动检查功能：
- ✅ dogs_120
- ✅ bird200 (CUB_200_2011)
- ✅ aircraft100 (fgvc_aircraft)

所有测试均通过，JSON文件能够正确生成，路径修复逻辑工作正常。
