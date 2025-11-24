# 数据泄漏检查报告

**生成时间**: /home/hdl/datasets/fgvr_fixed

---

## 📊 总体统计

| 数据集 | 训练集 | 测试集 | images_test | Discovery目录数 | 错误 |
|--------|--------|--------|-------------|-----------------|------|
| car_196 | 0 | 0 | N/A | 12 | 0 |
| CUB_200_2011 | 5994 | 5794 | 1991 | 12 | 0 |
| dogs_120 | 0 | 0 | N/A | 12 | 0 |
| dtd | 0 | 0 | N/A | 12 | 0 |
| eurosat | 0 | 0 | N/A | 12 | 0 |
| fgvc_aircraft | 3269 | 3269 | 3333 | 12 | 0 |
| flowers_102 | 0 | 0 | N/A | 12 | 0 |
| food_101 | 101 | 101 | 30300 | 12 | 0 |
| pet_37 | 0 | 0 | N/A | 12 | 0 |

---

## 🔍 car_196

**路径**: `/home/hdl/datasets/fgvr/car_196`

### 📈 数据集统计

- **训练集图像数**: 0
- **测试集图像数**: 0

### 📂 Discovery 目录统计

| 目录 | 图像数 |
|------|--------|
| images_discovery_all | 588 |
| images_discovery_all_1 | 196 |
| images_discovery_all_10 | 1960 |
| images_discovery_all_2 | 392 |
| images_discovery_all_3 | 588 |
| images_discovery_all_4 | 784 |
| images_discovery_all_5 | 980 |
| images_discovery_all_6 | 1176 |
| images_discovery_all_7 | 1372 |
| images_discovery_all_8 | 1568 |
| images_discovery_all_9 | 1764 |
| images_discovery_all_random | 203 |

### 🚨 数据泄漏检查 (Discovery → Test)

| Discovery 目录 | 总图像数 | 泄漏到测试集 | 泄漏比例 |
|----------------|----------|--------------|----------|
| ✅ images_discovery_all | 588 | 0 | 0.00% |
| ✅ images_discovery_all_1 | 196 | 0 | 0.00% |
| ✅ images_discovery_all_10 | 1960 | 0 | 0.00% |
| ✅ images_discovery_all_2 | 392 | 0 | 0.00% |
| ✅ images_discovery_all_3 | 588 | 0 | 0.00% |
| ✅ images_discovery_all_4 | 784 | 0 | 0.00% |
| ✅ images_discovery_all_5 | 980 | 0 | 0.00% |
| ✅ images_discovery_all_6 | 1176 | 0 | 0.00% |
| ✅ images_discovery_all_7 | 1372 | 0 | 0.00% |
| ✅ images_discovery_all_8 | 1568 | 0 | 0.00% |
| ✅ images_discovery_all_9 | 1764 | 0 | 0.00% |
| ✅ images_discovery_all_random | 203 | 0 | 0.00% |

**✅ 通过**: 所有 discovery 目录均不包含测试集图像。

### ✓ 训练集来源验证 (Discovery 来自 Train)

### 🔄 Discovery 目录之间的重叠

| 目录对 | 重叠图像数 | 占第一个目录比例 | 占第二个目录比例 |
|--------|------------|------------------|------------------|
| images_discovery_all_10 ↔ images_discovery_all_random | 45 | 2.3% | 22.2% |
| images_discovery_all_1 ↔ images_discovery_all_10 | 39 | 19.9% | 2.0% |
| images_discovery_all_1 ↔ images_discovery_all_2 | 10 | 5.1% | 2.6% |
| images_discovery_all_1 ↔ images_discovery_all_3 | 20 | 10.2% | 3.4% |
| images_discovery_all_1 ↔ images_discovery_all_4 | 22 | 11.2% | 2.8% |
| images_discovery_all_1 ↔ images_discovery_all_5 | 27 | 13.8% | 2.8% |
| images_discovery_all_1 ↔ images_discovery_all_6 | 26 | 13.3% | 2.2% |
| images_discovery_all_1 ↔ images_discovery_all_7 | 35 | 17.9% | 2.6% |
| images_discovery_all_1 ↔ images_discovery_all_8 | 31 | 15.8% | 2.0% |
| images_discovery_all_1 ↔ images_discovery_all_9 | 48 | 24.5% | 2.7% |
| images_discovery_all_1 ↔ images_discovery_all_random | 8 | 4.1% | 3.9% |
| images_discovery_all_2 ↔ images_discovery_all_10 | 110 | 28.1% | 5.6% |
| images_discovery_all_2 ↔ images_discovery_all_3 | 31 | 7.9% | 5.3% |
| images_discovery_all_2 ↔ images_discovery_all_4 | 28 | 7.1% | 3.6% |
| images_discovery_all_2 ↔ images_discovery_all_5 | 58 | 14.8% | 5.9% |
| images_discovery_all_2 ↔ images_discovery_all_6 | 60 | 15.3% | 5.1% |
| images_discovery_all_2 ↔ images_discovery_all_7 | 73 | 18.6% | 5.3% |
| images_discovery_all_2 ↔ images_discovery_all_8 | 78 | 19.9% | 5.0% |
| images_discovery_all_2 ↔ images_discovery_all_9 | 88 | 22.4% | 5.0% |
| images_discovery_all_2 ↔ images_discovery_all_random | 7 | 1.8% | 3.4% |
| images_discovery_all_3 ↔ images_discovery_all_10 | 129 | 21.9% | 6.6% |
| images_discovery_all_3 ↔ images_discovery_all_4 | 50 | 8.5% | 6.4% |
| images_discovery_all_3 ↔ images_discovery_all_5 | 73 | 12.4% | 7.4% |
| images_discovery_all_3 ↔ images_discovery_all_6 | 89 | 15.1% | 7.6% |
| images_discovery_all_3 ↔ images_discovery_all_7 | 100 | 17.0% | 7.3% |
| images_discovery_all_3 ↔ images_discovery_all_8 | 108 | 18.4% | 6.9% |
| images_discovery_all_3 ↔ images_discovery_all_9 | 125 | 21.3% | 7.1% |
| images_discovery_all_3 ↔ images_discovery_all_random | 14 | 2.4% | 6.9% |
| images_discovery_all_4 ↔ images_discovery_all_10 | 191 | 24.4% | 9.7% |
| images_discovery_all_4 ↔ images_discovery_all_5 | 105 | 13.4% | 10.7% |
| images_discovery_all_4 ↔ images_discovery_all_6 | 111 | 14.2% | 9.4% |
| images_discovery_all_4 ↔ images_discovery_all_7 | 126 | 16.1% | 9.2% |
| images_discovery_all_4 ↔ images_discovery_all_8 | 161 | 20.5% | 10.3% |
| images_discovery_all_4 ↔ images_discovery_all_9 | 178 | 22.7% | 10.1% |
| images_discovery_all_4 ↔ images_discovery_all_random | 22 | 2.8% | 10.8% |
| images_discovery_all_5 ↔ images_discovery_all_10 | 249 | 25.4% | 12.7% |
| images_discovery_all_5 ↔ images_discovery_all_6 | 163 | 16.6% | 13.9% |
| images_discovery_all_5 ↔ images_discovery_all_7 | 152 | 15.5% | 11.1% |
| images_discovery_all_5 ↔ images_discovery_all_8 | 199 | 20.3% | 12.7% |
| images_discovery_all_5 ↔ images_discovery_all_9 | 195 | 19.9% | 11.1% |
| images_discovery_all_5 ↔ images_discovery_all_random | 25 | 2.6% | 12.3% |
| images_discovery_all_6 ↔ images_discovery_all_10 | 306 | 26.0% | 15.6% |
| images_discovery_all_6 ↔ images_discovery_all_7 | 203 | 17.3% | 14.8% |
| images_discovery_all_6 ↔ images_discovery_all_8 | 242 | 20.6% | 15.4% |
| images_discovery_all_6 ↔ images_discovery_all_9 | 250 | 21.3% | 14.2% |
| images_discovery_all_6 ↔ images_discovery_all_random | 29 | 2.5% | 14.3% |
| images_discovery_all_7 ↔ images_discovery_all_10 | 314 | 22.9% | 16.0% |
| images_discovery_all_7 ↔ images_discovery_all_8 | 255 | 18.6% | 16.3% |
| images_discovery_all_7 ↔ images_discovery_all_9 | 274 | 20.0% | 15.5% |
| images_discovery_all_7 ↔ images_discovery_all_random | 32 | 2.3% | 15.8% |
| images_discovery_all_8 ↔ images_discovery_all_10 | 412 | 26.3% | 21.0% |
| images_discovery_all_8 ↔ images_discovery_all_9 | 358 | 22.8% | 20.3% |
| images_discovery_all_8 ↔ images_discovery_all_random | 40 | 2.6% | 19.7% |
| images_discovery_all_9 ↔ images_discovery_all_10 | 412 | 23.4% | 21.0% |
| images_discovery_all_9 ↔ images_discovery_all_random | 44 | 2.5% | 21.7% |
| images_discovery_all ↔ images_discovery_all_1 | 13 | 2.2% | 6.6% |
| images_discovery_all ↔ images_discovery_all_10 | 145 | 24.7% | 7.4% |
| images_discovery_all ↔ images_discovery_all_2 | 25 | 4.3% | 6.4% |
| images_discovery_all ↔ images_discovery_all_3 | 55 | 9.4% | 9.4% |
| images_discovery_all ↔ images_discovery_all_4 | 47 | 8.0% | 6.0% |
| images_discovery_all ↔ images_discovery_all_5 | 82 | 13.9% | 8.4% |
| images_discovery_all ↔ images_discovery_all_6 | 84 | 14.3% | 7.1% |
| images_discovery_all ↔ images_discovery_all_7 | 98 | 16.7% | 7.1% |
| images_discovery_all ↔ images_discovery_all_8 | 111 | 18.9% | 7.1% |
| images_discovery_all ↔ images_discovery_all_9 | 140 | 23.8% | 7.9% |
| images_discovery_all ↔ images_discovery_all_random | 15 | 2.6% | 7.4% |

*注: Discovery 目录之间允许有重叠，这是正常的。*


---

## 🔍 CUB_200_2011

**路径**: `/home/hdl/datasets/fgvr/CUB_200_2011/CUB_200_2011`

### 📈 数据集统计

- **训练集图像数**: 5994
- **测试集图像数**: 5794
- **images_test 目录**:
  - 图像数: 1991
  - 与官方测试集重叠: 1991 (34.4%)

### 📂 Discovery 目录统计

| 目录 | 图像数 |
|------|--------|
| images_discovery_all | 600 |
| images_discovery_all_1 | 200 |
| images_discovery_all_10 | 2000 |
| images_discovery_all_2 | 400 |
| images_discovery_all_3 | 600 |
| images_discovery_all_4 | 800 |
| images_discovery_all_5 | 1000 |
| images_discovery_all_6 | 1200 |
| images_discovery_all_7 | 1400 |
| images_discovery_all_8 | 1600 |
| images_discovery_all_9 | 1800 |
| images_discovery_all_random | 213 |

### 🚨 数据泄漏检查 (Discovery → Test)

| Discovery 目录 | 总图像数 | 泄漏到测试集 | 泄漏比例 |
|----------------|----------|--------------|----------|
| ✅ images_discovery_all | 600 | 0 | 0.00% |
| ✅ images_discovery_all_1 | 200 | 0 | 0.00% |
| ✅ images_discovery_all_10 | 2000 | 0 | 0.00% |
| ✅ images_discovery_all_2 | 400 | 0 | 0.00% |
| ✅ images_discovery_all_3 | 600 | 0 | 0.00% |
| ✅ images_discovery_all_4 | 800 | 0 | 0.00% |
| ✅ images_discovery_all_5 | 1000 | 0 | 0.00% |
| ✅ images_discovery_all_6 | 1200 | 0 | 0.00% |
| ✅ images_discovery_all_7 | 1400 | 0 | 0.00% |
| ✅ images_discovery_all_8 | 1600 | 0 | 0.00% |
| ✅ images_discovery_all_9 | 1800 | 0 | 0.00% |
| ✅ images_discovery_all_random | 213 | 0 | 0.00% |

**✅ 通过**: 所有 discovery 目录均不包含测试集图像。

### ✓ 训练集来源验证 (Discovery 来自 Train)

| Discovery 目录 | 总图像数 | 来自训练集 | 训练集覆盖率 | 不在训练集 |
|----------------|----------|------------|--------------|------------|
| ⚠️ images_discovery_all | 600 | 287 | 47.8% | 313 |
| ⚠️ images_discovery_all_1 | 200 | 109 | 54.5% | 91 |
| ⚠️ images_discovery_all_10 | 2000 | 1053 | 52.6% | 947 |
| ⚠️ images_discovery_all_2 | 400 | 190 | 47.5% | 210 |
| ⚠️ images_discovery_all_3 | 600 | 290 | 48.3% | 310 |
| ⚠️ images_discovery_all_4 | 800 | 434 | 54.2% | 366 |
| ⚠️ images_discovery_all_5 | 1000 | 524 | 52.4% | 476 |
| ⚠️ images_discovery_all_6 | 1200 | 589 | 49.1% | 611 |
| ⚠️ images_discovery_all_7 | 1400 | 692 | 49.4% | 708 |
| ⚠️ images_discovery_all_8 | 1600 | 824 | 51.5% | 776 |
| ⚠️ images_discovery_all_9 | 1800 | 929 | 51.6% | 871 |
| ⚠️ images_discovery_all_random | 213 | 103 | 48.4% | 110 |

### 🔄 Discovery 目录之间的重叠

| 目录对 | 重叠图像数 | 占第一个目录比例 | 占第二个目录比例 |
|--------|------------|------------------|------------------|
| images_discovery_all_10 ↔ images_discovery_all_random | 45 | 2.2% | 21.1% |
| images_discovery_all_1 ↔ images_discovery_all_10 | 20 | 10.0% | 1.0% |
| images_discovery_all_1 ↔ images_discovery_all_2 | 6 | 3.0% | 1.5% |
| images_discovery_all_1 ↔ images_discovery_all_3 | 7 | 3.5% | 1.2% |
| images_discovery_all_1 ↔ images_discovery_all_4 | 12 | 6.0% | 1.5% |
| images_discovery_all_1 ↔ images_discovery_all_5 | 16 | 8.0% | 1.6% |
| images_discovery_all_1 ↔ images_discovery_all_6 | 26 | 13.0% | 2.2% |
| images_discovery_all_1 ↔ images_discovery_all_7 | 15 | 7.5% | 1.1% |
| images_discovery_all_1 ↔ images_discovery_all_8 | 28 | 14.0% | 1.8% |
| images_discovery_all_1 ↔ images_discovery_all_9 | 29 | 14.5% | 1.6% |
| images_discovery_all_1 ↔ images_discovery_all_random | 5 | 2.5% | 2.3% |
| images_discovery_all_2 ↔ images_discovery_all_10 | 71 | 17.8% | 3.5% |
| images_discovery_all_2 ↔ images_discovery_all_3 | 17 | 4.2% | 2.8% |
| images_discovery_all_2 ↔ images_discovery_all_4 | 26 | 6.5% | 3.2% |
| images_discovery_all_2 ↔ images_discovery_all_5 | 39 | 9.8% | 3.9% |
| images_discovery_all_2 ↔ images_discovery_all_6 | 44 | 11.0% | 3.7% |
| images_discovery_all_2 ↔ images_discovery_all_7 | 57 | 14.2% | 4.1% |
| images_discovery_all_2 ↔ images_discovery_all_8 | 64 | 16.0% | 4.0% |
| images_discovery_all_2 ↔ images_discovery_all_9 | 61 | 15.2% | 3.4% |
| images_discovery_all_2 ↔ images_discovery_all_random | 5 | 1.2% | 2.3% |
| images_discovery_all_3 ↔ images_discovery_all_10 | 90 | 15.0% | 4.5% |
| images_discovery_all_3 ↔ images_discovery_all_4 | 39 | 6.5% | 4.9% |
| images_discovery_all_3 ↔ images_discovery_all_5 | 48 | 8.0% | 4.8% |
| images_discovery_all_3 ↔ images_discovery_all_6 | 76 | 12.7% | 6.3% |
| images_discovery_all_3 ↔ images_discovery_all_7 | 65 | 10.8% | 4.6% |
| images_discovery_all_3 ↔ images_discovery_all_8 | 81 | 13.5% | 5.1% |
| images_discovery_all_3 ↔ images_discovery_all_9 | 82 | 13.7% | 4.6% |
| images_discovery_all_3 ↔ images_discovery_all_random | 11 | 1.8% | 5.2% |
| images_discovery_all_4 ↔ images_discovery_all_10 | 134 | 16.8% | 6.7% |
| images_discovery_all_4 ↔ images_discovery_all_5 | 65 | 8.1% | 6.5% |
| images_discovery_all_4 ↔ images_discovery_all_6 | 78 | 9.8% | 6.5% |
| images_discovery_all_4 ↔ images_discovery_all_7 | 98 | 12.2% | 7.0% |
| images_discovery_all_4 ↔ images_discovery_all_8 | 91 | 11.4% | 5.7% |
| images_discovery_all_4 ↔ images_discovery_all_9 | 119 | 14.9% | 6.6% |
| images_discovery_all_4 ↔ images_discovery_all_random | 11 | 1.4% | 5.2% |
| images_discovery_all_5 ↔ images_discovery_all_10 | 180 | 18.0% | 9.0% |
| images_discovery_all_5 ↔ images_discovery_all_6 | 85 | 8.5% | 7.1% |
| images_discovery_all_5 ↔ images_discovery_all_7 | 110 | 11.0% | 7.9% |
| images_discovery_all_5 ↔ images_discovery_all_8 | 134 | 13.4% | 8.4% |
| images_discovery_all_5 ↔ images_discovery_all_9 | 145 | 14.5% | 8.1% |
| images_discovery_all_5 ↔ images_discovery_all_random | 20 | 2.0% | 9.4% |
| images_discovery_all_6 ↔ images_discovery_all_10 | 199 | 16.6% | 10.0% |
| images_discovery_all_6 ↔ images_discovery_all_7 | 145 | 12.1% | 10.4% |
| images_discovery_all_6 ↔ images_discovery_all_8 | 153 | 12.8% | 9.6% |
| images_discovery_all_6 ↔ images_discovery_all_9 | 201 | 16.8% | 11.2% |
| images_discovery_all_6 ↔ images_discovery_all_random | 21 | 1.8% | 9.9% |
| images_discovery_all_7 ↔ images_discovery_all_10 | 225 | 16.1% | 11.2% |
| images_discovery_all_7 ↔ images_discovery_all_8 | 200 | 14.3% | 12.5% |
| images_discovery_all_7 ↔ images_discovery_all_9 | 211 | 15.1% | 11.7% |
| images_discovery_all_7 ↔ images_discovery_all_random | 18 | 1.3% | 8.5% |
| images_discovery_all_8 ↔ images_discovery_all_10 | 272 | 17.0% | 13.6% |
| images_discovery_all_8 ↔ images_discovery_all_9 | 216 | 13.5% | 12.0% |
| images_discovery_all_8 ↔ images_discovery_all_random | 35 | 2.2% | 16.4% |
| images_discovery_all_9 ↔ images_discovery_all_10 | 321 | 17.8% | 16.1% |
| images_discovery_all_9 ↔ images_discovery_all_random | 31 | 1.7% | 14.6% |
| images_discovery_all ↔ images_discovery_all_1 | 11 | 1.8% | 5.5% |
| images_discovery_all ↔ images_discovery_all_10 | 91 | 15.2% | 4.5% |
| images_discovery_all ↔ images_discovery_all_2 | 18 | 3.0% | 4.5% |
| images_discovery_all ↔ images_discovery_all_3 | 26 | 4.3% | 4.3% |
| images_discovery_all ↔ images_discovery_all_4 | 54 | 9.0% | 6.8% |
| images_discovery_all ↔ images_discovery_all_5 | 55 | 9.2% | 5.5% |
| images_discovery_all ↔ images_discovery_all_6 | 60 | 10.0% | 5.0% |
| images_discovery_all ↔ images_discovery_all_7 | 58 | 9.7% | 4.1% |
| images_discovery_all ↔ images_discovery_all_8 | 87 | 14.5% | 5.4% |
| images_discovery_all ↔ images_discovery_all_9 | 90 | 15.0% | 5.0% |
| images_discovery_all ↔ images_discovery_all_random | 7 | 1.2% | 3.3% |

*注: Discovery 目录之间允许有重叠，这是正常的。*


---

## 🔍 dogs_120

**路径**: `/home/hdl/datasets/fgvr/dogs_120`

### 📈 数据集统计

- **训练集图像数**: 0
- **测试集图像数**: 0

### 📂 Discovery 目录统计

| 目录 | 图像数 |
|------|--------|
| images_discovery_all | 360 |
| images_discovery_all_1 | 120 |
| images_discovery_all_10 | 1200 |
| images_discovery_all_2 | 240 |
| images_discovery_all_3 | 360 |
| images_discovery_all_4 | 480 |
| images_discovery_all_5 | 600 |
| images_discovery_all_6 | 720 |
| images_discovery_all_7 | 840 |
| images_discovery_all_8 | 960 |
| images_discovery_all_9 | 1080 |
| images_discovery_all_random | 274 |

### 🚨 数据泄漏检查 (Discovery → Test)

| Discovery 目录 | 总图像数 | 泄漏到测试集 | 泄漏比例 |
|----------------|----------|--------------|----------|
| ✅ images_discovery_all | 360 | 0 | 0.00% |
| ✅ images_discovery_all_1 | 120 | 0 | 0.00% |
| ✅ images_discovery_all_10 | 1200 | 0 | 0.00% |
| ✅ images_discovery_all_2 | 240 | 0 | 0.00% |
| ✅ images_discovery_all_3 | 360 | 0 | 0.00% |
| ✅ images_discovery_all_4 | 480 | 0 | 0.00% |
| ✅ images_discovery_all_5 | 600 | 0 | 0.00% |
| ✅ images_discovery_all_6 | 720 | 0 | 0.00% |
| ✅ images_discovery_all_7 | 840 | 0 | 0.00% |
| ✅ images_discovery_all_8 | 960 | 0 | 0.00% |
| ✅ images_discovery_all_9 | 1080 | 0 | 0.00% |
| ✅ images_discovery_all_random | 274 | 0 | 0.00% |

**✅ 通过**: 所有 discovery 目录均不包含测试集图像。

### ✓ 训练集来源验证 (Discovery 来自 Train)

### 🔄 Discovery 目录之间的重叠

| 目录对 | 重叠图像数 | 占第一个目录比例 | 占第二个目录比例 |
|--------|------------|------------------|------------------|
| images_discovery_all_10 ↔ images_discovery_all_random | 58 | 4.8% | 21.2% |
| images_discovery_all_1 ↔ images_discovery_all_10 | 29 | 24.2% | 2.4% |
| images_discovery_all_1 ↔ images_discovery_all_2 | 7 | 5.8% | 2.9% |
| images_discovery_all_1 ↔ images_discovery_all_3 | 4 | 3.3% | 1.1% |
| images_discovery_all_1 ↔ images_discovery_all_4 | 11 | 9.2% | 2.3% |
| images_discovery_all_1 ↔ images_discovery_all_5 | 9 | 7.5% | 1.5% |
| images_discovery_all_1 ↔ images_discovery_all_6 | 23 | 19.2% | 3.2% |
| images_discovery_all_1 ↔ images_discovery_all_7 | 14 | 11.7% | 1.7% |
| images_discovery_all_1 ↔ images_discovery_all_8 | 16 | 13.3% | 1.7% |
| images_discovery_all_1 ↔ images_discovery_all_9 | 25 | 20.8% | 2.3% |
| images_discovery_all_1 ↔ images_discovery_all_random | 4 | 3.3% | 1.5% |
| images_discovery_all_2 ↔ images_discovery_all_10 | 47 | 19.6% | 3.9% |
| images_discovery_all_2 ↔ images_discovery_all_3 | 12 | 5.0% | 3.3% |
| images_discovery_all_2 ↔ images_discovery_all_4 | 11 | 4.6% | 2.3% |
| images_discovery_all_2 ↔ images_discovery_all_5 | 34 | 14.2% | 5.7% |
| images_discovery_all_2 ↔ images_discovery_all_6 | 28 | 11.7% | 3.9% |
| images_discovery_all_2 ↔ images_discovery_all_7 | 31 | 12.9% | 3.7% |
| images_discovery_all_2 ↔ images_discovery_all_8 | 51 | 21.2% | 5.3% |
| images_discovery_all_2 ↔ images_discovery_all_9 | 45 | 18.8% | 4.2% |
| images_discovery_all_2 ↔ images_discovery_all_random | 12 | 5.0% | 4.4% |
| images_discovery_all_3 ↔ images_discovery_all_10 | 84 | 23.3% | 7.0% |
| images_discovery_all_3 ↔ images_discovery_all_4 | 34 | 9.4% | 7.1% |
| images_discovery_all_3 ↔ images_discovery_all_5 | 40 | 11.1% | 6.7% |
| images_discovery_all_3 ↔ images_discovery_all_6 | 38 | 10.6% | 5.3% |
| images_discovery_all_3 ↔ images_discovery_all_7 | 42 | 11.7% | 5.0% |
| images_discovery_all_3 ↔ images_discovery_all_8 | 66 | 18.3% | 6.9% |
| images_discovery_all_3 ↔ images_discovery_all_9 | 59 | 16.4% | 5.5% |
| images_discovery_all_3 ↔ images_discovery_all_random | 21 | 5.8% | 7.7% |
| images_discovery_all_4 ↔ images_discovery_all_10 | 106 | 22.1% | 8.8% |
| images_discovery_all_4 ↔ images_discovery_all_5 | 41 | 8.5% | 6.8% |
| images_discovery_all_4 ↔ images_discovery_all_6 | 56 | 11.7% | 7.8% |
| images_discovery_all_4 ↔ images_discovery_all_7 | 71 | 14.8% | 8.5% |
| images_discovery_all_4 ↔ images_discovery_all_8 | 59 | 12.3% | 6.1% |
| images_discovery_all_4 ↔ images_discovery_all_9 | 95 | 19.8% | 8.8% |
| images_discovery_all_4 ↔ images_discovery_all_random | 24 | 5.0% | 8.8% |
| images_discovery_all_5 ↔ images_discovery_all_10 | 111 | 18.5% | 9.2% |
| images_discovery_all_5 ↔ images_discovery_all_6 | 75 | 12.5% | 10.4% |
| images_discovery_all_5 ↔ images_discovery_all_7 | 80 | 13.3% | 9.5% |
| images_discovery_all_5 ↔ images_discovery_all_8 | 102 | 17.0% | 10.6% |
| images_discovery_all_5 ↔ images_discovery_all_9 | 102 | 17.0% | 9.4% |
| images_discovery_all_5 ↔ images_discovery_all_random | 26 | 4.3% | 9.5% |
| images_discovery_all_6 ↔ images_discovery_all_10 | 155 | 21.5% | 12.9% |
| images_discovery_all_6 ↔ images_discovery_all_7 | 105 | 14.6% | 12.5% |
| images_discovery_all_6 ↔ images_discovery_all_8 | 111 | 15.4% | 11.6% |
| images_discovery_all_6 ↔ images_discovery_all_9 | 120 | 16.7% | 11.1% |
| images_discovery_all_6 ↔ images_discovery_all_random | 27 | 3.8% | 9.9% |
| images_discovery_all_7 ↔ images_discovery_all_10 | 182 | 21.7% | 15.2% |
| images_discovery_all_7 ↔ images_discovery_all_8 | 125 | 14.9% | 13.0% |
| images_discovery_all_7 ↔ images_discovery_all_9 | 151 | 18.0% | 14.0% |
| images_discovery_all_7 ↔ images_discovery_all_random | 39 | 4.6% | 14.2% |
| images_discovery_all_8 ↔ images_discovery_all_10 | 196 | 20.4% | 16.3% |
| images_discovery_all_8 ↔ images_discovery_all_9 | 164 | 17.1% | 15.2% |
| images_discovery_all_8 ↔ images_discovery_all_random | 39 | 4.1% | 14.2% |
| images_discovery_all_9 ↔ images_discovery_all_10 | 206 | 19.1% | 17.2% |
| images_discovery_all_9 ↔ images_discovery_all_random | 50 | 4.6% | 18.2% |
| images_discovery_all ↔ images_discovery_all_1 | 4 | 1.1% | 3.3% |
| images_discovery_all ↔ images_discovery_all_10 | 84 | 23.3% | 7.0% |
| images_discovery_all ↔ images_discovery_all_2 | 12 | 3.3% | 5.0% |
| images_discovery_all ↔ images_discovery_all_3 | 360 | 100.0% | 100.0% |
| images_discovery_all ↔ images_discovery_all_4 | 34 | 9.4% | 7.1% |
| images_discovery_all ↔ images_discovery_all_5 | 40 | 11.1% | 6.7% |
| images_discovery_all ↔ images_discovery_all_6 | 38 | 10.6% | 5.3% |
| images_discovery_all ↔ images_discovery_all_7 | 42 | 11.7% | 5.0% |
| images_discovery_all ↔ images_discovery_all_8 | 66 | 18.3% | 6.9% |
| images_discovery_all ↔ images_discovery_all_9 | 59 | 16.4% | 5.5% |
| images_discovery_all ↔ images_discovery_all_random | 21 | 5.8% | 7.7% |

*注: Discovery 目录之间允许有重叠，这是正常的。*


---

## 🔍 dtd

**路径**: `/home/hdl/datasets/fgvr/dtd`

### 📈 数据集统计

- **训练集图像数**: 0
- **测试集图像数**: 0

### 📂 Discovery 目录统计

| 目录 | 图像数 |
|------|--------|
| images_discovery_all | 141 |
| images_discovery_all_1 | 47 |
| images_discovery_all_10 | 470 |
| images_discovery_all_2 | 94 |
| images_discovery_all_3 | 141 |
| images_discovery_all_4 | 188 |
| images_discovery_all_5 | 235 |
| images_discovery_all_6 | 282 |
| images_discovery_all_7 | 329 |
| images_discovery_all_8 | 376 |
| images_discovery_all_9 | 423 |
| images_discovery_all_random | 109 |

### 🚨 数据泄漏检查 (Discovery → Test)

| Discovery 目录 | 总图像数 | 泄漏到测试集 | 泄漏比例 |
|----------------|----------|--------------|----------|
| ✅ images_discovery_all | 141 | 0 | 0.00% |
| ✅ images_discovery_all_1 | 47 | 0 | 0.00% |
| ✅ images_discovery_all_10 | 470 | 0 | 0.00% |
| ✅ images_discovery_all_2 | 94 | 0 | 0.00% |
| ✅ images_discovery_all_3 | 141 | 0 | 0.00% |
| ✅ images_discovery_all_4 | 188 | 0 | 0.00% |
| ✅ images_discovery_all_5 | 235 | 0 | 0.00% |
| ✅ images_discovery_all_6 | 282 | 0 | 0.00% |
| ✅ images_discovery_all_7 | 329 | 0 | 0.00% |
| ✅ images_discovery_all_8 | 376 | 0 | 0.00% |
| ✅ images_discovery_all_9 | 423 | 0 | 0.00% |
| ✅ images_discovery_all_random | 109 | 0 | 0.00% |

**✅ 通过**: 所有 discovery 目录均不包含测试集图像。

### ✓ 训练集来源验证 (Discovery 来自 Train)

### 🔄 Discovery 目录之间的重叠

| 目录对 | 重叠图像数 | 占第一个目录比例 | 占第二个目录比例 |
|--------|------------|------------------|------------------|
| images_discovery_all_10 ↔ images_discovery_all_random | 22 | 4.7% | 20.2% |
| images_discovery_all_1 ↔ images_discovery_all_10 | 2 | 4.3% | 0.4% |
| images_discovery_all_1 ↔ images_discovery_all_2 | 1 | 2.1% | 1.1% |
| images_discovery_all_1 ↔ images_discovery_all_4 | 1 | 2.1% | 0.5% |
| images_discovery_all_1 ↔ images_discovery_all_5 | 5 | 10.6% | 2.1% |
| images_discovery_all_1 ↔ images_discovery_all_6 | 3 | 6.4% | 1.1% |
| images_discovery_all_1 ↔ images_discovery_all_7 | 4 | 8.5% | 1.2% |
| images_discovery_all_1 ↔ images_discovery_all_8 | 3 | 6.4% | 0.8% |
| images_discovery_all_1 ↔ images_discovery_all_9 | 8 | 17.0% | 1.9% |
| images_discovery_all_1 ↔ images_discovery_all_random | 1 | 2.1% | 0.9% |
| images_discovery_all_2 ↔ images_discovery_all_10 | 11 | 11.7% | 2.3% |
| images_discovery_all_2 ↔ images_discovery_all_3 | 3 | 3.2% | 2.1% |
| images_discovery_all_2 ↔ images_discovery_all_4 | 6 | 6.4% | 3.2% |
| images_discovery_all_2 ↔ images_discovery_all_5 | 11 | 11.7% | 4.7% |
| images_discovery_all_2 ↔ images_discovery_all_6 | 7 | 7.4% | 2.5% |
| images_discovery_all_2 ↔ images_discovery_all_7 | 13 | 13.8% | 4.0% |
| images_discovery_all_2 ↔ images_discovery_all_8 | 13 | 13.8% | 3.5% |
| images_discovery_all_2 ↔ images_discovery_all_9 | 13 | 13.8% | 3.1% |
| images_discovery_all_2 ↔ images_discovery_all_random | 1 | 1.1% | 0.9% |
| images_discovery_all_3 ↔ images_discovery_all_10 | 24 | 17.0% | 5.1% |
| images_discovery_all_3 ↔ images_discovery_all_4 | 4 | 2.8% | 2.1% |
| images_discovery_all_3 ↔ images_discovery_all_5 | 13 | 9.2% | 5.5% |
| images_discovery_all_3 ↔ images_discovery_all_6 | 12 | 8.5% | 4.3% |
| images_discovery_all_3 ↔ images_discovery_all_7 | 19 | 13.5% | 5.8% |
| images_discovery_all_3 ↔ images_discovery_all_8 | 24 | 17.0% | 6.4% |
| images_discovery_all_3 ↔ images_discovery_all_9 | 26 | 18.4% | 6.1% |
| images_discovery_all_3 ↔ images_discovery_all_random | 3 | 2.1% | 2.8% |
| images_discovery_all_4 ↔ images_discovery_all_10 | 35 | 18.6% | 7.4% |
| images_discovery_all_4 ↔ images_discovery_all_5 | 15 | 8.0% | 6.4% |
| images_discovery_all_4 ↔ images_discovery_all_6 | 16 | 8.5% | 5.7% |
| images_discovery_all_4 ↔ images_discovery_all_7 | 20 | 10.6% | 6.1% |
| images_discovery_all_4 ↔ images_discovery_all_8 | 17 | 9.0% | 4.5% |
| images_discovery_all_4 ↔ images_discovery_all_9 | 29 | 15.4% | 6.9% |
| images_discovery_all_4 ↔ images_discovery_all_random | 8 | 4.3% | 7.3% |
| images_discovery_all_5 ↔ images_discovery_all_10 | 46 | 19.6% | 9.8% |
| images_discovery_all_5 ↔ images_discovery_all_6 | 31 | 13.2% | 11.0% |
| images_discovery_all_5 ↔ images_discovery_all_7 | 28 | 11.9% | 8.5% |
| images_discovery_all_5 ↔ images_discovery_all_8 | 33 | 14.0% | 8.8% |
| images_discovery_all_5 ↔ images_discovery_all_9 | 30 | 12.8% | 7.1% |
| images_discovery_all_5 ↔ images_discovery_all_random | 5 | 2.1% | 4.6% |
| images_discovery_all_6 ↔ images_discovery_all_10 | 40 | 14.2% | 8.5% |
| images_discovery_all_6 ↔ images_discovery_all_7 | 37 | 13.1% | 11.2% |
| images_discovery_all_6 ↔ images_discovery_all_8 | 36 | 12.8% | 9.6% |
| images_discovery_all_6 ↔ images_discovery_all_9 | 38 | 13.5% | 9.0% |
| images_discovery_all_6 ↔ images_discovery_all_random | 12 | 4.3% | 11.0% |
| images_discovery_all_7 ↔ images_discovery_all_10 | 53 | 16.1% | 11.3% |
| images_discovery_all_7 ↔ images_discovery_all_8 | 49 | 14.9% | 13.0% |
| images_discovery_all_7 ↔ images_discovery_all_9 | 50 | 15.2% | 11.8% |
| images_discovery_all_7 ↔ images_discovery_all_random | 14 | 4.3% | 12.8% |
| images_discovery_all_8 ↔ images_discovery_all_10 | 58 | 15.4% | 12.3% |
| images_discovery_all_8 ↔ images_discovery_all_9 | 53 | 14.1% | 12.5% |
| images_discovery_all_8 ↔ images_discovery_all_random | 21 | 5.6% | 19.3% |
| images_discovery_all_9 ↔ images_discovery_all_10 | 68 | 16.1% | 14.5% |
| images_discovery_all_9 ↔ images_discovery_all_random | 15 | 3.5% | 13.8% |
| images_discovery_all ↔ images_discovery_all_1 | 1 | 0.7% | 2.1% |
| images_discovery_all ↔ images_discovery_all_10 | 28 | 19.9% | 6.0% |
| images_discovery_all ↔ images_discovery_all_2 | 7 | 5.0% | 7.4% |
| images_discovery_all ↔ images_discovery_all_3 | 9 | 6.4% | 6.4% |
| images_discovery_all ↔ images_discovery_all_4 | 8 | 5.7% | 4.3% |
| images_discovery_all ↔ images_discovery_all_5 | 14 | 9.9% | 6.0% |
| images_discovery_all ↔ images_discovery_all_6 | 16 | 11.3% | 5.7% |
| images_discovery_all ↔ images_discovery_all_7 | 17 | 12.1% | 5.2% |
| images_discovery_all ↔ images_discovery_all_8 | 22 | 15.6% | 5.9% |
| images_discovery_all ↔ images_discovery_all_9 | 32 | 22.7% | 7.6% |
| images_discovery_all ↔ images_discovery_all_random | 5 | 3.5% | 4.6% |

*注: Discovery 目录之间允许有重叠，这是正常的。*


---

## 🔍 eurosat

**路径**: `/home/hdl/datasets/fgvr/eurosat`

### 📈 数据集统计

- **训练集图像数**: 0
- **测试集图像数**: 0

### 📂 Discovery 目录统计

| 目录 | 图像数 |
|------|--------|
| images_discovery_all | 30 |
| images_discovery_all_1 | 10 |
| images_discovery_all_10 | 100 |
| images_discovery_all_2 | 20 |
| images_discovery_all_3 | 30 |
| images_discovery_all_4 | 40 |
| images_discovery_all_5 | 50 |
| images_discovery_all_6 | 60 |
| images_discovery_all_7 | 70 |
| images_discovery_all_8 | 80 |
| images_discovery_all_9 | 90 |
| images_discovery_all_random | 27 |

### 🚨 数据泄漏检查 (Discovery → Test)

| Discovery 目录 | 总图像数 | 泄漏到测试集 | 泄漏比例 |
|----------------|----------|--------------|----------|
| ✅ images_discovery_all | 30 | 0 | 0.00% |
| ✅ images_discovery_all_1 | 10 | 0 | 0.00% |
| ✅ images_discovery_all_10 | 100 | 0 | 0.00% |
| ✅ images_discovery_all_2 | 20 | 0 | 0.00% |
| ✅ images_discovery_all_3 | 30 | 0 | 0.00% |
| ✅ images_discovery_all_4 | 40 | 0 | 0.00% |
| ✅ images_discovery_all_5 | 50 | 0 | 0.00% |
| ✅ images_discovery_all_6 | 60 | 0 | 0.00% |
| ✅ images_discovery_all_7 | 70 | 0 | 0.00% |
| ✅ images_discovery_all_8 | 80 | 0 | 0.00% |
| ✅ images_discovery_all_9 | 90 | 0 | 0.00% |
| ✅ images_discovery_all_random | 27 | 0 | 0.00% |

**✅ 通过**: 所有 discovery 目录均不包含测试集图像。

### ✓ 训练集来源验证 (Discovery 来自 Train)

### 🔄 Discovery 目录之间的重叠

| 目录对 | 重叠图像数 | 占第一个目录比例 | 占第二个目录比例 |
|--------|------------|------------------|------------------|
| images_discovery_all_3 ↔ images_discovery_all_10 | 1 | 3.3% | 1.0% |
| images_discovery_all_6 ↔ images_discovery_all_7 | 1 | 1.7% | 1.4% |
| images_discovery_all_6 ↔ images_discovery_all_9 | 1 | 1.7% | 1.1% |
| images_discovery_all_7 ↔ images_discovery_all_10 | 3 | 4.3% | 3.0% |
| images_discovery_all_7 ↔ images_discovery_all_9 | 1 | 1.4% | 1.1% |
| images_discovery_all_9 ↔ images_discovery_all_10 | 2 | 2.2% | 2.0% |

*注: Discovery 目录之间允许有重叠，这是正常的。*


---

## 🔍 fgvc_aircraft

**路径**: `/home/hdl/datasets/fgvr/fgvc_aircraft`

### 📈 数据集统计

- **训练集图像数**: 3269
- **测试集图像数**: 3269
- **images_test 目录**:
  - 图像数: 3333
  - 与官方测试集重叠: 0 (0.0%)
  - ⚠️ 不在官方测试集中: 3333

### 📂 Discovery 目录统计

| 目录 | 图像数 |
|------|--------|
| images_discovery_all | 300 |
| images_discovery_all_1 | 100 |
| images_discovery_all_10 | 1000 |
| images_discovery_all_2 | 200 |
| images_discovery_all_3 | 300 |
| images_discovery_all_4 | 400 |
| images_discovery_all_5 | 500 |
| images_discovery_all_6 | 600 |
| images_discovery_all_7 | 700 |
| images_discovery_all_8 | 800 |
| images_discovery_all_9 | 900 |
| images_discovery_all_random | 241 |

### 🚨 数据泄漏检查 (Discovery → Test)

| Discovery 目录 | 总图像数 | 泄漏到测试集 | 泄漏比例 |
|----------------|----------|--------------|----------|
| ✅ images_discovery_all | 300 | 0 | 0.00% |
| ✅ images_discovery_all_1 | 100 | 0 | 0.00% |
| ✅ images_discovery_all_10 | 1000 | 0 | 0.00% |
| ✅ images_discovery_all_2 | 200 | 0 | 0.00% |
| ✅ images_discovery_all_3 | 300 | 0 | 0.00% |
| ✅ images_discovery_all_4 | 400 | 0 | 0.00% |
| ✅ images_discovery_all_5 | 500 | 0 | 0.00% |
| ✅ images_discovery_all_6 | 600 | 0 | 0.00% |
| ✅ images_discovery_all_7 | 700 | 0 | 0.00% |
| ✅ images_discovery_all_8 | 800 | 0 | 0.00% |
| ✅ images_discovery_all_9 | 900 | 0 | 0.00% |
| ✅ images_discovery_all_random | 241 | 0 | 0.00% |

**✅ 通过**: 所有 discovery 目录均不包含测试集图像。

### ✓ 训练集来源验证 (Discovery 来自 Train)

| Discovery 目录 | 总图像数 | 来自训练集 | 训练集覆盖率 | 不在训练集 |
|----------------|----------|------------|--------------|------------|
| ⚠️ images_discovery_all | 300 | 0 | 0.0% | 300 |
| ⚠️ images_discovery_all_1 | 100 | 0 | 0.0% | 100 |
| ⚠️ images_discovery_all_10 | 1000 | 0 | 0.0% | 1000 |
| ⚠️ images_discovery_all_2 | 200 | 0 | 0.0% | 200 |
| ⚠️ images_discovery_all_3 | 300 | 0 | 0.0% | 300 |
| ⚠️ images_discovery_all_4 | 400 | 0 | 0.0% | 400 |
| ⚠️ images_discovery_all_5 | 500 | 0 | 0.0% | 500 |
| ⚠️ images_discovery_all_6 | 600 | 0 | 0.0% | 600 |
| ⚠️ images_discovery_all_7 | 700 | 0 | 0.0% | 700 |
| ⚠️ images_discovery_all_8 | 800 | 0 | 0.0% | 800 |
| ⚠️ images_discovery_all_9 | 900 | 0 | 0.0% | 900 |
| ⚠️ images_discovery_all_random | 241 | 0 | 0.0% | 241 |

### 🔄 Discovery 目录之间的重叠

| 目录对 | 重叠图像数 | 占第一个目录比例 | 占第二个目录比例 |
|--------|------------|------------------|------------------|
| images_discovery_all_10 ↔ images_discovery_all_random | 78 | 7.8% | 32.4% |
| images_discovery_all_1 ↔ images_discovery_all_10 | 35 | 35.0% | 3.5% |
| images_discovery_all_1 ↔ images_discovery_all_2 | 2 | 2.0% | 1.0% |
| images_discovery_all_1 ↔ images_discovery_all_3 | 8 | 8.0% | 2.7% |
| images_discovery_all_1 ↔ images_discovery_all_4 | 8 | 8.0% | 2.0% |
| images_discovery_all_1 ↔ images_discovery_all_5 | 16 | 16.0% | 3.2% |
| images_discovery_all_1 ↔ images_discovery_all_6 | 16 | 16.0% | 2.7% |
| images_discovery_all_1 ↔ images_discovery_all_7 | 19 | 19.0% | 2.7% |
| images_discovery_all_1 ↔ images_discovery_all_8 | 20 | 20.0% | 2.5% |
| images_discovery_all_1 ↔ images_discovery_all_9 | 33 | 33.0% | 3.7% |
| images_discovery_all_1 ↔ images_discovery_all_random | 11 | 11.0% | 4.6% |
| images_discovery_all_2 ↔ images_discovery_all_10 | 56 | 28.0% | 5.6% |
| images_discovery_all_2 ↔ images_discovery_all_3 | 23 | 11.5% | 7.7% |
| images_discovery_all_2 ↔ images_discovery_all_4 | 29 | 14.5% | 7.2% |
| images_discovery_all_2 ↔ images_discovery_all_5 | 33 | 16.5% | 6.6% |
| images_discovery_all_2 ↔ images_discovery_all_6 | 31 | 15.5% | 5.2% |
| images_discovery_all_2 ↔ images_discovery_all_7 | 46 | 23.0% | 6.6% |
| images_discovery_all_2 ↔ images_discovery_all_8 | 54 | 27.0% | 6.8% |
| images_discovery_all_2 ↔ images_discovery_all_9 | 49 | 24.5% | 5.4% |
| images_discovery_all_2 ↔ images_discovery_all_random | 16 | 8.0% | 6.6% |
| images_discovery_all_3 ↔ images_discovery_all_10 | 82 | 27.3% | 8.2% |
| images_discovery_all_3 ↔ images_discovery_all_4 | 40 | 13.3% | 10.0% |
| images_discovery_all_3 ↔ images_discovery_all_5 | 46 | 15.3% | 9.2% |
| images_discovery_all_3 ↔ images_discovery_all_6 | 62 | 20.7% | 10.3% |
| images_discovery_all_3 ↔ images_discovery_all_7 | 52 | 17.3% | 7.4% |
| images_discovery_all_3 ↔ images_discovery_all_8 | 74 | 24.7% | 9.2% |
| images_discovery_all_3 ↔ images_discovery_all_9 | 76 | 25.3% | 8.4% |
| images_discovery_all_3 ↔ images_discovery_all_random | 21 | 7.0% | 8.7% |
| images_discovery_all_4 ↔ images_discovery_all_10 | 118 | 29.5% | 11.8% |
| images_discovery_all_4 ↔ images_discovery_all_5 | 58 | 14.5% | 11.6% |
| images_discovery_all_4 ↔ images_discovery_all_6 | 69 | 17.2% | 11.5% |
| images_discovery_all_4 ↔ images_discovery_all_7 | 90 | 22.5% | 12.9% |
| images_discovery_all_4 ↔ images_discovery_all_8 | 99 | 24.8% | 12.4% |
| images_discovery_all_4 ↔ images_discovery_all_9 | 91 | 22.8% | 10.1% |
| images_discovery_all_4 ↔ images_discovery_all_random | 31 | 7.8% | 12.9% |
| images_discovery_all_5 ↔ images_discovery_all_10 | 145 | 29.0% | 14.5% |
| images_discovery_all_5 ↔ images_discovery_all_6 | 91 | 18.2% | 15.2% |
| images_discovery_all_5 ↔ images_discovery_all_7 | 107 | 21.4% | 15.3% |
| images_discovery_all_5 ↔ images_discovery_all_8 | 125 | 25.0% | 15.6% |
| images_discovery_all_5 ↔ images_discovery_all_9 | 137 | 27.4% | 15.2% |
| images_discovery_all_5 ↔ images_discovery_all_random | 46 | 9.2% | 19.1% |
| images_discovery_all_6 ↔ images_discovery_all_10 | 173 | 28.8% | 17.3% |
| images_discovery_all_6 ↔ images_discovery_all_7 | 126 | 21.0% | 18.0% |
| images_discovery_all_6 ↔ images_discovery_all_8 | 140 | 23.3% | 17.5% |
| images_discovery_all_6 ↔ images_discovery_all_9 | 163 | 27.2% | 18.1% |
| images_discovery_all_6 ↔ images_discovery_all_random | 42 | 7.0% | 17.4% |
| images_discovery_all_7 ↔ images_discovery_all_10 | 224 | 32.0% | 22.4% |
| images_discovery_all_7 ↔ images_discovery_all_8 | 179 | 25.6% | 22.4% |
| images_discovery_all_7 ↔ images_discovery_all_9 | 183 | 26.1% | 20.3% |
| images_discovery_all_7 ↔ images_discovery_all_random | 52 | 7.4% | 21.6% |
| images_discovery_all_8 ↔ images_discovery_all_10 | 241 | 30.1% | 24.1% |
| images_discovery_all_8 ↔ images_discovery_all_9 | 219 | 27.4% | 24.3% |
| images_discovery_all_8 ↔ images_discovery_all_random | 59 | 7.4% | 24.5% |
| images_discovery_all_9 ↔ images_discovery_all_10 | 279 | 31.0% | 27.9% |
| images_discovery_all_9 ↔ images_discovery_all_random | 71 | 7.9% | 29.5% |
| images_discovery_all ↔ images_discovery_all_1 | 7 | 2.3% | 7.0% |
| images_discovery_all ↔ images_discovery_all_10 | 97 | 32.3% | 9.7% |
| images_discovery_all ↔ images_discovery_all_2 | 18 | 6.0% | 9.0% |
| images_discovery_all ↔ images_discovery_all_3 | 21 | 7.0% | 7.0% |
| images_discovery_all ↔ images_discovery_all_4 | 35 | 11.7% | 8.8% |
| images_discovery_all ↔ images_discovery_all_5 | 39 | 13.0% | 7.8% |
| images_discovery_all ↔ images_discovery_all_6 | 55 | 18.3% | 9.2% |
| images_discovery_all ↔ images_discovery_all_7 | 63 | 21.0% | 9.0% |
| images_discovery_all ↔ images_discovery_all_8 | 69 | 23.0% | 8.6% |
| images_discovery_all ↔ images_discovery_all_9 | 83 | 27.7% | 9.2% |
| images_discovery_all ↔ images_discovery_all_random | 21 | 7.0% | 8.7% |

*注: Discovery 目录之间允许有重叠，这是正常的。*


---

## 🔍 flowers_102

**路径**: `/home/hdl/datasets/fgvr/flowers_102`

### 📈 数据集统计

- **训练集图像数**: 0
- **测试集图像数**: 0

### 📂 Discovery 目录统计

| 目录 | 图像数 |
|------|--------|
| images_discovery_all | 306 |
| images_discovery_all_1 | 102 |
| images_discovery_all_10 | 1020 |
| images_discovery_all_2 | 204 |
| images_discovery_all_3 | 306 |
| images_discovery_all_4 | 408 |
| images_discovery_all_5 | 510 |
| images_discovery_all_6 | 612 |
| images_discovery_all_7 | 714 |
| images_discovery_all_8 | 816 |
| images_discovery_all_9 | 918 |
| images_discovery_all_random | 229 |

### 🚨 数据泄漏检查 (Discovery → Test)

| Discovery 目录 | 总图像数 | 泄漏到测试集 | 泄漏比例 |
|----------------|----------|--------------|----------|
| ✅ images_discovery_all | 306 | 0 | 0.00% |
| ✅ images_discovery_all_1 | 102 | 0 | 0.00% |
| ✅ images_discovery_all_10 | 1020 | 0 | 0.00% |
| ✅ images_discovery_all_2 | 204 | 0 | 0.00% |
| ✅ images_discovery_all_3 | 306 | 0 | 0.00% |
| ✅ images_discovery_all_4 | 408 | 0 | 0.00% |
| ✅ images_discovery_all_5 | 510 | 0 | 0.00% |
| ✅ images_discovery_all_6 | 612 | 0 | 0.00% |
| ✅ images_discovery_all_7 | 714 | 0 | 0.00% |
| ✅ images_discovery_all_8 | 816 | 0 | 0.00% |
| ✅ images_discovery_all_9 | 918 | 0 | 0.00% |
| ✅ images_discovery_all_random | 229 | 0 | 0.00% |

**✅ 通过**: 所有 discovery 目录均不包含测试集图像。

### ✓ 训练集来源验证 (Discovery 来自 Train)

### 🔄 Discovery 目录之间的重叠

| 目录对 | 重叠图像数 | 占第一个目录比例 | 占第二个目录比例 |
|--------|------------|------------------|------------------|
| images_discovery_all_10 ↔ images_discovery_all_random | 117 | 11.5% | 51.1% |
| images_discovery_all_1 ↔ images_discovery_all_10 | 48 | 47.1% | 4.7% |
| images_discovery_all_1 ↔ images_discovery_all_2 | 10 | 9.8% | 4.9% |
| images_discovery_all_1 ↔ images_discovery_all_3 | 17 | 16.7% | 5.6% |
| images_discovery_all_1 ↔ images_discovery_all_4 | 19 | 18.6% | 4.7% |
| images_discovery_all_1 ↔ images_discovery_all_5 | 22 | 21.6% | 4.3% |
| images_discovery_all_1 ↔ images_discovery_all_6 | 24 | 23.5% | 3.9% |
| images_discovery_all_1 ↔ images_discovery_all_7 | 31 | 30.4% | 4.3% |
| images_discovery_all_1 ↔ images_discovery_all_8 | 47 | 46.1% | 5.8% |
| images_discovery_all_1 ↔ images_discovery_all_9 | 47 | 46.1% | 5.1% |
| images_discovery_all_1 ↔ images_discovery_all_random | 12 | 11.8% | 5.2% |
| images_discovery_all_2 ↔ images_discovery_all_10 | 102 | 50.0% | 10.0% |
| images_discovery_all_2 ↔ images_discovery_all_3 | 25 | 12.3% | 8.2% |
| images_discovery_all_2 ↔ images_discovery_all_4 | 42 | 20.6% | 10.3% |
| images_discovery_all_2 ↔ images_discovery_all_5 | 49 | 24.0% | 9.6% |
| images_discovery_all_2 ↔ images_discovery_all_6 | 71 | 34.8% | 11.6% |
| images_discovery_all_2 ↔ images_discovery_all_7 | 76 | 37.3% | 10.6% |
| images_discovery_all_2 ↔ images_discovery_all_8 | 81 | 39.7% | 9.9% |
| images_discovery_all_2 ↔ images_discovery_all_9 | 98 | 48.0% | 10.7% |
| images_discovery_all_2 ↔ images_discovery_all_random | 18 | 8.8% | 7.9% |
| images_discovery_all_3 ↔ images_discovery_all_10 | 138 | 45.1% | 13.5% |
| images_discovery_all_3 ↔ images_discovery_all_4 | 55 | 18.0% | 13.5% |
| images_discovery_all_3 ↔ images_discovery_all_5 | 69 | 22.5% | 13.5% |
| images_discovery_all_3 ↔ images_discovery_all_6 | 81 | 26.5% | 13.2% |
| images_discovery_all_3 ↔ images_discovery_all_7 | 117 | 38.2% | 16.4% |
| images_discovery_all_3 ↔ images_discovery_all_8 | 121 | 39.5% | 14.8% |
| images_discovery_all_3 ↔ images_discovery_all_9 | 153 | 50.0% | 16.7% |
| images_discovery_all_3 ↔ images_discovery_all_random | 31 | 10.1% | 13.5% |
| images_discovery_all_4 ↔ images_discovery_all_10 | 208 | 51.0% | 20.4% |
| images_discovery_all_4 ↔ images_discovery_all_5 | 110 | 27.0% | 21.6% |
| images_discovery_all_4 ↔ images_discovery_all_6 | 106 | 26.0% | 17.3% |
| images_discovery_all_4 ↔ images_discovery_all_7 | 154 | 37.7% | 21.6% |
| images_discovery_all_4 ↔ images_discovery_all_8 | 156 | 38.2% | 19.1% |
| images_discovery_all_4 ↔ images_discovery_all_9 | 186 | 45.6% | 20.3% |
| images_discovery_all_4 ↔ images_discovery_all_random | 44 | 10.8% | 19.2% |
| images_discovery_all_5 ↔ images_discovery_all_10 | 248 | 48.6% | 24.3% |
| images_discovery_all_5 ↔ images_discovery_all_6 | 158 | 31.0% | 25.8% |
| images_discovery_all_5 ↔ images_discovery_all_7 | 167 | 32.7% | 23.4% |
| images_discovery_all_5 ↔ images_discovery_all_8 | 205 | 40.2% | 25.1% |
| images_discovery_all_5 ↔ images_discovery_all_9 | 227 | 44.5% | 24.7% |
| images_discovery_all_5 ↔ images_discovery_all_random | 70 | 13.7% | 30.6% |
| images_discovery_all_6 ↔ images_discovery_all_10 | 304 | 49.7% | 29.8% |
| images_discovery_all_6 ↔ images_discovery_all_7 | 211 | 34.5% | 29.6% |
| images_discovery_all_6 ↔ images_discovery_all_8 | 260 | 42.5% | 31.9% |
| images_discovery_all_6 ↔ images_discovery_all_9 | 265 | 43.3% | 28.9% |
| images_discovery_all_6 ↔ images_discovery_all_random | 72 | 11.8% | 31.4% |
| images_discovery_all_7 ↔ images_discovery_all_10 | 359 | 50.3% | 35.2% |
| images_discovery_all_7 ↔ images_discovery_all_8 | 289 | 40.5% | 35.4% |
| images_discovery_all_7 ↔ images_discovery_all_9 | 319 | 44.7% | 34.7% |
| images_discovery_all_7 ↔ images_discovery_all_random | 89 | 12.5% | 38.9% |
| images_discovery_all_8 ↔ images_discovery_all_10 | 395 | 48.4% | 38.7% |
| images_discovery_all_8 ↔ images_discovery_all_9 | 350 | 42.9% | 38.1% |
| images_discovery_all_8 ↔ images_discovery_all_random | 81 | 9.9% | 35.4% |
| images_discovery_all_9 ↔ images_discovery_all_10 | 465 | 50.7% | 45.6% |
| images_discovery_all_9 ↔ images_discovery_all_random | 104 | 11.3% | 45.4% |
| images_discovery_all ↔ images_discovery_all_1 | 17 | 5.6% | 16.7% |
| images_discovery_all ↔ images_discovery_all_10 | 138 | 45.1% | 13.5% |
| images_discovery_all ↔ images_discovery_all_2 | 25 | 8.2% | 12.3% |
| images_discovery_all ↔ images_discovery_all_3 | 306 | 100.0% | 100.0% |
| images_discovery_all ↔ images_discovery_all_4 | 55 | 18.0% | 13.5% |
| images_discovery_all ↔ images_discovery_all_5 | 69 | 22.5% | 13.5% |
| images_discovery_all ↔ images_discovery_all_6 | 81 | 26.5% | 13.2% |
| images_discovery_all ↔ images_discovery_all_7 | 117 | 38.2% | 16.4% |
| images_discovery_all ↔ images_discovery_all_8 | 121 | 39.5% | 14.8% |
| images_discovery_all ↔ images_discovery_all_9 | 153 | 50.0% | 16.7% |
| images_discovery_all ↔ images_discovery_all_random | 31 | 10.1% | 13.5% |

*注: Discovery 目录之间允许有重叠，这是正常的。*


---

## 🔍 food_101

**路径**: `/home/hdl/datasets/fgvr/food_101`

### 📈 数据集统计

- **训练集图像数**: 101
- **测试集图像数**: 101
- **images_test 目录**:
  - 图像数: 30300
  - 与官方测试集重叠: 0 (0.0%)
  - ⚠️ 不在官方测试集中: 30300

### 📂 Discovery 目录统计

| 目录 | 图像数 |
|------|--------|
| images_discovery_all | 303 |
| images_discovery_all_1 | 101 |
| images_discovery_all_10 | 1010 |
| images_discovery_all_2 | 202 |
| images_discovery_all_3 | 303 |
| images_discovery_all_4 | 404 |
| images_discovery_all_5 | 505 |
| images_discovery_all_6 | 606 |
| images_discovery_all_7 | 707 |
| images_discovery_all_8 | 808 |
| images_discovery_all_9 | 909 |
| images_discovery_all_random | 242 |

### 🚨 数据泄漏检查 (Discovery → Test)

| Discovery 目录 | 总图像数 | 泄漏到测试集 | 泄漏比例 |
|----------------|----------|--------------|----------|
| ✅ images_discovery_all | 303 | 0 | 0.00% |
| ✅ images_discovery_all_1 | 101 | 0 | 0.00% |
| ✅ images_discovery_all_10 | 1010 | 0 | 0.00% |
| ✅ images_discovery_all_2 | 202 | 0 | 0.00% |
| ✅ images_discovery_all_3 | 303 | 0 | 0.00% |
| ✅ images_discovery_all_4 | 404 | 0 | 0.00% |
| ✅ images_discovery_all_5 | 505 | 0 | 0.00% |
| ✅ images_discovery_all_6 | 606 | 0 | 0.00% |
| ✅ images_discovery_all_7 | 707 | 0 | 0.00% |
| ✅ images_discovery_all_8 | 808 | 0 | 0.00% |
| ✅ images_discovery_all_9 | 909 | 0 | 0.00% |
| ✅ images_discovery_all_random | 242 | 0 | 0.00% |

**✅ 通过**: 所有 discovery 目录均不包含测试集图像。

### ✓ 训练集来源验证 (Discovery 来自 Train)

| Discovery 目录 | 总图像数 | 来自训练集 | 训练集覆盖率 | 不在训练集 |
|----------------|----------|------------|--------------|------------|
| ⚠️ images_discovery_all | 303 | 0 | 0.0% | 303 |
| ⚠️ images_discovery_all_1 | 101 | 0 | 0.0% | 101 |
| ⚠️ images_discovery_all_10 | 1010 | 0 | 0.0% | 1010 |
| ⚠️ images_discovery_all_2 | 202 | 0 | 0.0% | 202 |
| ⚠️ images_discovery_all_3 | 303 | 0 | 0.0% | 303 |
| ⚠️ images_discovery_all_4 | 404 | 0 | 0.0% | 404 |
| ⚠️ images_discovery_all_5 | 505 | 0 | 0.0% | 505 |
| ⚠️ images_discovery_all_6 | 606 | 0 | 0.0% | 606 |
| ⚠️ images_discovery_all_7 | 707 | 0 | 0.0% | 707 |
| ⚠️ images_discovery_all_8 | 808 | 0 | 0.0% | 808 |
| ⚠️ images_discovery_all_9 | 909 | 0 | 0.0% | 909 |
| ⚠️ images_discovery_all_random | 242 | 0 | 0.0% | 242 |

### 🔄 Discovery 目录之间的重叠

| 目录对 | 重叠图像数 | 占第一个目录比例 | 占第二个目录比例 |
|--------|------------|------------------|------------------|
| images_discovery_all_10 ↔ images_discovery_all_random | 7 | 0.7% | 2.9% |
| images_discovery_all_1 ↔ images_discovery_all_10 | 3 | 3.0% | 0.3% |
| images_discovery_all_1 ↔ images_discovery_all_3 | 2 | 2.0% | 0.7% |
| images_discovery_all_1 ↔ images_discovery_all_5 | 1 | 1.0% | 0.2% |
| images_discovery_all_1 ↔ images_discovery_all_8 | 2 | 2.0% | 0.2% |
| images_discovery_all_1 ↔ images_discovery_all_9 | 1 | 1.0% | 0.1% |
| images_discovery_all_2 ↔ images_discovery_all_10 | 7 | 3.5% | 0.7% |
| images_discovery_all_2 ↔ images_discovery_all_3 | 2 | 1.0% | 0.7% |
| images_discovery_all_2 ↔ images_discovery_all_5 | 2 | 1.0% | 0.4% |
| images_discovery_all_2 ↔ images_discovery_all_6 | 4 | 2.0% | 0.7% |
| images_discovery_all_2 ↔ images_discovery_all_8 | 2 | 1.0% | 0.2% |
| images_discovery_all_2 ↔ images_discovery_all_9 | 1 | 0.5% | 0.1% |
| images_discovery_all_3 ↔ images_discovery_all_10 | 5 | 1.7% | 0.5% |
| images_discovery_all_3 ↔ images_discovery_all_4 | 1 | 0.3% | 0.2% |
| images_discovery_all_3 ↔ images_discovery_all_5 | 2 | 0.7% | 0.4% |
| images_discovery_all_3 ↔ images_discovery_all_6 | 5 | 1.7% | 0.8% |
| images_discovery_all_3 ↔ images_discovery_all_7 | 4 | 1.3% | 0.6% |
| images_discovery_all_3 ↔ images_discovery_all_8 | 2 | 0.7% | 0.2% |
| images_discovery_all_3 ↔ images_discovery_all_9 | 5 | 1.7% | 0.6% |
| images_discovery_all_3 ↔ images_discovery_all_random | 2 | 0.7% | 0.8% |
| images_discovery_all_4 ↔ images_discovery_all_10 | 4 | 1.0% | 0.4% |
| images_discovery_all_4 ↔ images_discovery_all_5 | 3 | 0.7% | 0.6% |
| images_discovery_all_4 ↔ images_discovery_all_6 | 4 | 1.0% | 0.7% |
| images_discovery_all_4 ↔ images_discovery_all_7 | 5 | 1.2% | 0.7% |
| images_discovery_all_4 ↔ images_discovery_all_8 | 6 | 1.5% | 0.7% |
| images_discovery_all_4 ↔ images_discovery_all_9 | 8 | 2.0% | 0.9% |
| images_discovery_all_4 ↔ images_discovery_all_random | 5 | 1.2% | 2.1% |
| images_discovery_all_5 ↔ images_discovery_all_10 | 5 | 1.0% | 0.5% |
| images_discovery_all_5 ↔ images_discovery_all_6 | 5 | 1.0% | 0.8% |
| images_discovery_all_5 ↔ images_discovery_all_7 | 12 | 2.4% | 1.7% |
| images_discovery_all_5 ↔ images_discovery_all_8 | 9 | 1.8% | 1.1% |
| images_discovery_all_5 ↔ images_discovery_all_9 | 7 | 1.4% | 0.8% |
| images_discovery_all_5 ↔ images_discovery_all_random | 1 | 0.2% | 0.4% |
| images_discovery_all_6 ↔ images_discovery_all_10 | 12 | 2.0% | 1.2% |
| images_discovery_all_6 ↔ images_discovery_all_7 | 8 | 1.3% | 1.1% |
| images_discovery_all_6 ↔ images_discovery_all_8 | 13 | 2.1% | 1.6% |
| images_discovery_all_6 ↔ images_discovery_all_9 | 19 | 3.1% | 2.1% |
| images_discovery_all_6 ↔ images_discovery_all_random | 5 | 0.8% | 2.1% |
| images_discovery_all_7 ↔ images_discovery_all_10 | 12 | 1.7% | 1.2% |
| images_discovery_all_7 ↔ images_discovery_all_8 | 12 | 1.7% | 1.5% |
| images_discovery_all_7 ↔ images_discovery_all_9 | 15 | 2.1% | 1.7% |
| images_discovery_all_7 ↔ images_discovery_all_random | 2 | 0.3% | 0.8% |
| images_discovery_all_8 ↔ images_discovery_all_10 | 15 | 1.9% | 1.5% |
| images_discovery_all_8 ↔ images_discovery_all_9 | 14 | 1.7% | 1.5% |
| images_discovery_all_8 ↔ images_discovery_all_random | 5 | 0.6% | 2.1% |
| images_discovery_all_9 ↔ images_discovery_all_10 | 17 | 1.9% | 1.7% |
| images_discovery_all_9 ↔ images_discovery_all_random | 2 | 0.2% | 0.8% |
| images_discovery_all ↔ images_discovery_all_1 | 1 | 0.3% | 1.0% |
| images_discovery_all ↔ images_discovery_all_10 | 8 | 2.6% | 0.8% |
| images_discovery_all ↔ images_discovery_all_2 | 1 | 0.3% | 0.5% |
| images_discovery_all ↔ images_discovery_all_3 | 1 | 0.3% | 0.3% |
| images_discovery_all ↔ images_discovery_all_4 | 1 | 0.3% | 0.2% |
| images_discovery_all ↔ images_discovery_all_5 | 2 | 0.7% | 0.4% |
| images_discovery_all ↔ images_discovery_all_6 | 4 | 1.3% | 0.7% |
| images_discovery_all ↔ images_discovery_all_7 | 4 | 1.3% | 0.6% |
| images_discovery_all ↔ images_discovery_all_8 | 3 | 1.0% | 0.4% |
| images_discovery_all ↔ images_discovery_all_9 | 6 | 2.0% | 0.7% |
| images_discovery_all ↔ images_discovery_all_random | 1 | 0.3% | 0.4% |

*注: Discovery 目录之间允许有重叠，这是正常的。*


---

## 🔍 pet_37

**路径**: `/home/hdl/datasets/fgvr/pet_37`

### 📈 数据集统计

- **训练集图像数**: 0
- **测试集图像数**: 0

### 📂 Discovery 目录统计

| 目录 | 图像数 |
|------|--------|
| images_discovery_all | 111 |
| images_discovery_all_1 | 37 |
| images_discovery_all_10 | 370 |
| images_discovery_all_2 | 74 |
| images_discovery_all_3 | 111 |
| images_discovery_all_4 | 148 |
| images_discovery_all_5 | 185 |
| images_discovery_all_6 | 222 |
| images_discovery_all_7 | 259 |
| images_discovery_all_8 | 296 |
| images_discovery_all_9 | 333 |
| images_discovery_all_random | 115 |

### 🚨 数据泄漏检查 (Discovery → Test)

| Discovery 目录 | 总图像数 | 泄漏到测试集 | 泄漏比例 |
|----------------|----------|--------------|----------|
| ✅ images_discovery_all | 111 | 0 | 0.00% |
| ✅ images_discovery_all_1 | 37 | 0 | 0.00% |
| ✅ images_discovery_all_10 | 370 | 0 | 0.00% |
| ✅ images_discovery_all_2 | 74 | 0 | 0.00% |
| ✅ images_discovery_all_3 | 111 | 0 | 0.00% |
| ✅ images_discovery_all_4 | 148 | 0 | 0.00% |
| ✅ images_discovery_all_5 | 185 | 0 | 0.00% |
| ✅ images_discovery_all_6 | 222 | 0 | 0.00% |
| ✅ images_discovery_all_7 | 259 | 0 | 0.00% |
| ✅ images_discovery_all_8 | 296 | 0 | 0.00% |
| ✅ images_discovery_all_9 | 333 | 0 | 0.00% |
| ✅ images_discovery_all_random | 115 | 0 | 0.00% |

**✅ 通过**: 所有 discovery 目录均不包含测试集图像。

### ✓ 训练集来源验证 (Discovery 来自 Train)

### 🔄 Discovery 目录之间的重叠

| 目录对 | 重叠图像数 | 占第一个目录比例 | 占第二个目录比例 |
|--------|------------|------------------|------------------|
| images_discovery_all_10 ↔ images_discovery_all_random | 24 | 6.5% | 20.9% |
| images_discovery_all_1 ↔ images_discovery_all_10 | 8 | 21.6% | 2.2% |
| images_discovery_all_1 ↔ images_discovery_all_3 | 1 | 2.7% | 0.9% |
| images_discovery_all_1 ↔ images_discovery_all_4 | 4 | 10.8% | 2.7% |
| images_discovery_all_1 ↔ images_discovery_all_5 | 5 | 13.5% | 2.7% |
| images_discovery_all_1 ↔ images_discovery_all_6 | 5 | 13.5% | 2.3% |
| images_discovery_all_1 ↔ images_discovery_all_7 | 5 | 13.5% | 1.9% |
| images_discovery_all_1 ↔ images_discovery_all_8 | 3 | 8.1% | 1.0% |
| images_discovery_all_1 ↔ images_discovery_all_9 | 7 | 18.9% | 2.1% |
| images_discovery_all_1 ↔ images_discovery_all_random | 4 | 10.8% | 3.5% |
| images_discovery_all_2 ↔ images_discovery_all_10 | 13 | 17.6% | 3.5% |
| images_discovery_all_2 ↔ images_discovery_all_3 | 6 | 8.1% | 5.4% |
| images_discovery_all_2 ↔ images_discovery_all_4 | 4 | 5.4% | 2.7% |
| images_discovery_all_2 ↔ images_discovery_all_5 | 2 | 2.7% | 1.1% |
| images_discovery_all_2 ↔ images_discovery_all_6 | 10 | 13.5% | 4.5% |
| images_discovery_all_2 ↔ images_discovery_all_7 | 8 | 10.8% | 3.1% |
| images_discovery_all_2 ↔ images_discovery_all_8 | 11 | 14.9% | 3.7% |
| images_discovery_all_2 ↔ images_discovery_all_9 | 15 | 20.3% | 4.5% |
| images_discovery_all_2 ↔ images_discovery_all_random | 4 | 5.4% | 3.5% |
| images_discovery_all_3 ↔ images_discovery_all_10 | 18 | 16.2% | 4.9% |
| images_discovery_all_3 ↔ images_discovery_all_4 | 8 | 7.2% | 5.4% |
| images_discovery_all_3 ↔ images_discovery_all_5 | 9 | 8.1% | 4.9% |
| images_discovery_all_3 ↔ images_discovery_all_6 | 15 | 13.5% | 6.8% |
| images_discovery_all_3 ↔ images_discovery_all_7 | 16 | 14.4% | 6.2% |
| images_discovery_all_3 ↔ images_discovery_all_8 | 17 | 15.3% | 5.7% |
| images_discovery_all_3 ↔ images_discovery_all_9 | 22 | 19.8% | 6.6% |
| images_discovery_all_3 ↔ images_discovery_all_random | 8 | 7.2% | 7.0% |
| images_discovery_all_4 ↔ images_discovery_all_10 | 28 | 18.9% | 7.6% |
| images_discovery_all_4 ↔ images_discovery_all_5 | 16 | 10.8% | 8.6% |
| images_discovery_all_4 ↔ images_discovery_all_6 | 23 | 15.5% | 10.4% |
| images_discovery_all_4 ↔ images_discovery_all_7 | 16 | 10.8% | 6.2% |
| images_discovery_all_4 ↔ images_discovery_all_8 | 19 | 12.8% | 6.4% |
| images_discovery_all_4 ↔ images_discovery_all_9 | 27 | 18.2% | 8.1% |
| images_discovery_all_4 ↔ images_discovery_all_random | 8 | 5.4% | 7.0% |
| images_discovery_all_5 ↔ images_discovery_all_10 | 39 | 21.1% | 10.5% |
| images_discovery_all_5 ↔ images_discovery_all_6 | 27 | 14.6% | 12.2% |
| images_discovery_all_5 ↔ images_discovery_all_7 | 14 | 7.6% | 5.4% |
| images_discovery_all_5 ↔ images_discovery_all_8 | 27 | 14.6% | 9.1% |
| images_discovery_all_5 ↔ images_discovery_all_9 | 29 | 15.7% | 8.7% |
| images_discovery_all_5 ↔ images_discovery_all_random | 14 | 7.6% | 12.2% |
| images_discovery_all_6 ↔ images_discovery_all_10 | 42 | 18.9% | 11.4% |
| images_discovery_all_6 ↔ images_discovery_all_7 | 32 | 14.4% | 12.4% |
| images_discovery_all_6 ↔ images_discovery_all_8 | 32 | 14.4% | 10.8% |
| images_discovery_all_6 ↔ images_discovery_all_9 | 39 | 17.6% | 11.7% |
| images_discovery_all_6 ↔ images_discovery_all_random | 12 | 5.4% | 10.4% |
| images_discovery_all_7 ↔ images_discovery_all_10 | 54 | 20.8% | 14.6% |
| images_discovery_all_7 ↔ images_discovery_all_8 | 28 | 10.8% | 9.5% |
| images_discovery_all_7 ↔ images_discovery_all_9 | 44 | 17.0% | 13.2% |
| images_discovery_all_7 ↔ images_discovery_all_random | 20 | 7.7% | 17.4% |
| images_discovery_all_8 ↔ images_discovery_all_10 | 50 | 16.9% | 13.5% |
| images_discovery_all_8 ↔ images_discovery_all_9 | 57 | 19.3% | 17.1% |
| images_discovery_all_8 ↔ images_discovery_all_random | 24 | 8.1% | 20.9% |
| images_discovery_all_9 ↔ images_discovery_all_10 | 64 | 19.2% | 17.3% |
| images_discovery_all_9 ↔ images_discovery_all_random | 19 | 5.7% | 16.5% |
| images_discovery_all ↔ images_discovery_all_1 | 1 | 0.9% | 2.7% |
| images_discovery_all ↔ images_discovery_all_10 | 18 | 16.2% | 4.9% |
| images_discovery_all ↔ images_discovery_all_2 | 6 | 5.4% | 8.1% |
| images_discovery_all ↔ images_discovery_all_3 | 111 | 100.0% | 100.0% |
| images_discovery_all ↔ images_discovery_all_4 | 8 | 7.2% | 5.4% |
| images_discovery_all ↔ images_discovery_all_5 | 9 | 8.1% | 4.9% |
| images_discovery_all ↔ images_discovery_all_6 | 15 | 13.5% | 6.8% |
| images_discovery_all ↔ images_discovery_all_7 | 16 | 14.4% | 6.2% |
| images_discovery_all ↔ images_discovery_all_8 | 17 | 15.3% | 5.7% |
| images_discovery_all ↔ images_discovery_all_9 | 22 | 19.8% | 6.6% |
| images_discovery_all ↔ images_discovery_all_random | 8 | 7.2% | 7.0% |

*注: Discovery 目录之间允许有重叠，这是正常的。*


---

## 📝 总结

### ✅ 无数据泄漏

所有数据集的 discovery 目录均不包含测试集图像。

### 建议

1. **数据泄漏**: 如果发现 discovery 目录与 test 目录有重叠，需要重新生成 discovery 目录，确保只使用训练集图像。
2. **Discovery重叠**: Discovery 目录之间的重叠是允许的，因为它们都是从训练集随机采样的。
3. **训练集来源**: 确保所有 discovery 图像都来自训练集（100%覆盖率）。
4. **测试集验证**: 确保 images_test 目录与官方测试集划分一致。
