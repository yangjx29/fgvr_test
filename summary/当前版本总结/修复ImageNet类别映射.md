# ImageNet数据集准确率过低问题修复方案

## 问题根本原因

**核心问题**: 类别映射不匹配导致知识库和测试集使用不同的类别标识符

### 具体表现
1. **知识库**: 使用ImageNet数字ID (如 `n01498041`)
2. **测试集**: 使用英文类别名 (如 `stingray`) 
3. **结果**: 系统无法正确匹配，导致准确率极低(6.34%)

### 问题来源
- JSON构建时将ImageNet数字ID重新编号为0,1,2...
- 知识库构建时使用原始数字ID
- 测试集使用重新编号后的类别名
- 缺少正确的ID到类别名映射

## 修复方案

### 方案1: 重建JSON文件 (推荐)
重新构建JSON文件，保持ImageNet原始数字ID不变

**步骤**:
1. 修改`build_imagenet_json.py`脚本
2. 保持类别ID为原始ImageNet数字ID (如`n01498041`)
3. 重新生成所有JSON文件
4. 重新构建知识库

### 方案2: 创建映射表
创建ImageNet ID到类别名的映射表，在推理时使用

**步骤**:
1. 提取所有ImageNet ID到类别名的映射关系
2. 在推理时进行ID到类别名的转换
3. 修改分类逻辑以支持映射转换

### 方案3: 统一为类别名
将知识库中的类别键改为类别名

**步骤**:
1. 修改知识库构建逻辑
2. 使用类别名作为知识库的键
3. 重新构建知识库

## 推荐实施方案

**选择方案1**，因为：
- 最彻底的解决方案
- 保持与ImageNet标准一致
- 避免后续映射问题
- 与其他数据集处理方式统一

## 实施细节

### 1. 提取ImageNet映射
```python
# 从目录结构提取映射
import os
imagenet_dir = "/datasets/ImageNet_A/images/"
class_mapping = {}
for class_id in sorted(os.listdir(imagenet_dir)):
    if os.path.isdir(os.path.join(imagenet_dir, class_id)):
        # 需要找到对应的类别名
        class_mapping[class_id] = get_class_name(class_id)
```

### 2. 修改JSON构建脚本
```python
# 保持原始ImageNet ID
for class_data in original_data:
    imagenet_id = class_data[0]  # 如 'n01498041'
    class_name = get_class_name(imagenet_id)  # 如 'stingray'
    image_list = class_data[2]
    
    result.append([class_name, imagenet_id, image_list])
```

### 3. 验证修复
- 重新构建JSON文件
- 重新构建知识库
- 运行测试验证准确率恢复

## 预期效果
修复后准确率应该从6.34%恢复到90%+的正常水平。
