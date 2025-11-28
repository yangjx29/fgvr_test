# classify_result.json 结构修改文档

## 修改内容

### 1. JSON结构变更
将原来的扁平化列表结构改为树状结构，按`is_correct`字段组织结果。

#### 修改前结构
```json
{
  "dataset": "pet37",
  "timestamp": "2025-11-28 01:50:22",
  "total_samples": 3,
  "metadata": {...},
  "results": [
    {"label": "A", "prediction": "B", "is_correct": false, ...},
    {"label": "C", "prediction": "C", "is_correct": true, ...},
    {"label": "D", "prediction": "E", "is_correct": false, ...}
  ]
}
```

#### 修改后结构
```json
{
  "dataset": "pet37",
  "timestamp": "2025-11-28 01:50:22",
  "total_samples": 3,
  "false_count": 2,
  "true_count": 1,
  "metadata": {...},
  "results": {
    "false": [
      {"label": "A", "prediction": "B", "is_correct": false, ...},
      {"label": "D", "prediction": "E", "is_correct": false, ...}
    ],
    "true": [
      {"label": "C", "prediction": "C", "is_correct": true, ...}
    ]
  }
}
```

### 2. 具体修改
- **新增字段**:
  - `false_count`: 错误分类的数量
  - `true_count`: 正确分类的数量
- **结构调整**:
  - `results`从数组变为对象
  - `results.false`: 包含所有`is_correct: false`的结果
  - `results.true`: 包含所有`is_correct: true`的结果
- **排序规则**: 先记录false结果，后记录true结果

### 3. 修改的文件
- `data/result_saver.py`: `save_classification_result`函数

### 4. 功能验证
- ✅ 错误分类结果正确归类到`false`数组
- ✅ 正确分类结果正确归类到`true`数组
- ✅ 统计信息准确显示
- ✅ 文件删除逻辑正常工作

### 5. 兼容性说明
此修改会影响所有使用`save_classification_result`函数的模式：
- `classify`模式
- `evaluate`模式  
- `fastonly`模式
- `slowonly`模式
- `fast_slow`模式

### 6. 优势
1. **更清晰的分析**: 可以快速查看错误分类和正确分类的分布
2. **便于统计**: 直接提供错误和正确的数量统计
3. **树状结构**: 符合用户要求的JSON树状结构
4. **便于扩展**: 可以在false/true分支下进一步细分

## 测试结果
测试样本: 3个（2个错误，1个正确）
- JSON结构正确生成
- 统计信息准确
- 文件保存路径正确
