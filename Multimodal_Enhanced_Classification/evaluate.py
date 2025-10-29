# 导入命令行参数解析模块
import argparse
# 导入时间模块
import time
# 导入操作系统接口模块
import os
# 导入JSON处理模块
import json

# 导入PyTorch深度学习框架
import torch
# 导入PyTorch的函数式接口
import torch.nn.functional as F

# 导入工具函数：统计摘要、平均值计算器、进度显示器、准确率计算、随机种子设置
from utils.tools import Summary, AverageMeter, ProgressMeter, accuracy, set_random_seed
# 导入类别名称获取函数和自定义模板
from data.cls_to_names import get_classnames, CUSTOM_TEMPLATES

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

# 计算批次的熵值
# 参数:
#   logits: 模型输出的logits张量
# 返回:
#   entropy: 熵值张量
def calculate_batch_entropy(logits):
    # 使用熵的公式: H = -Σ(p * log(p))
    return -(logits.softmax(-1) * logits.log_softmax(-1)).sum(-1)

# 基于熵计算图像和文本的权重
# 这是AWT方法的核心：使用熵来衡量不确定性，熵越低表示越确定，权重越高
# 参数:
#   output: 模型输出 shape: (n_views, n_prompts, n_classes)
#   img_t: 图像温度参数，用于控制权重分布的平滑度，默认0.5
#   text_t: 文本温度参数，用于控制权重分布的平滑度，默认0.5
# 返回:
#   image_weights: 图像权重 shape: (n_views,)
#   text_weights: 文本权重 shape: (n_classes, n_descriptors)
@torch.no_grad()
def get_entropy_weight(output, img_t=0.5, text_t=0.5):
    with torch.cuda.amp.autocast():
        # 计算图像的权重
        # 对每个视图在所有描述符上取平均，然后计算熵
        image_entropy = calculate_batch_entropy(output.mean(1))
        # 使用负熵通过softmax得到权重（熵越低，权重越高）
        image_weights = F.softmax(-image_entropy/img_t, dim=-1)

        # 计算描述符（文本）的权重
        _, n_des, n_cls = output.shape  # 获取描述符数量和类别数量
        # 创建锚点：使用第一个视图的平均预测作为基准
        anchor = output[0].mean(0)[None, None, :].repeat(n_des, n_cls, 1)
        # 获取每个描述符的输出
        output_des = output[0].unsqueeze(-1)
        # 创建scatter索引
        scatter_indices = torch.arange(n_cls)[None, :, None].repeat(n_des, 1, 1).cuda()
        # 将每个描述符的预测散布到对应的类别位置
        anchor.scatter_(dim=2, index=scatter_indices, src=output_des) # shape: (n_des, n_cls, n_cls)
        # 计算文本熵
        text_entropy = calculate_batch_entropy(anchor)
        # 使用负熵通过softmax得到权重
        text_weights = F.softmax(-text_entropy.t()/text_t, dim=-1) # shape: (n_cls, n_des)

    return image_weights, text_weights

# 基于熵计算图像和文本的权重（测试版本1）
# 修改：先计算文本权重，再用文本权重加权平均来计算图像权重
# 参数:
#   output: 模型输出 shape: (n_views, n_prompts, n_classes)
#   img_t: 图像温度参数，用于控制权重分布的平滑度，默认0.5
#   text_t: 文本温度参数，用于控制权重分布的平滑度，默认0.5
# 返回:
#   image_weights: 图像权重 shape: (n_views,)
#   text_weights: 文本权重 shape: (n_classes, n_descriptors)
@torch.no_grad()
def get_entropy_weight_test1(output, img_t=0.5, text_t=0.5):
    with torch.cuda.amp.autocast():
        _, n_des, n_cls = output.shape  # 获取描述符数量和类别数量
        
        # 第一步：先计算描述符（文本）的权重
        # 创建锚点：使用第一个视图的平均预测作为基准
        anchor = output[0].mean(0)[None, None, :].repeat(n_des, n_cls, 1)
        # 获取每个描述符的输出
        output_des = output[0].unsqueeze(-1)
        # 创建scatter索引
        scatter_indices = torch.arange(n_cls)[None, :, None].repeat(n_des, 1, 1).cuda()
        # 将每个描述符的预测散布到对应的类别位置
        anchor.scatter_(dim=2, index=scatter_indices, src=output_des) # shape: (n_des, n_cls, n_cls)
        # 计算文本熵
        text_entropy = calculate_batch_entropy(anchor)
        # 使用负熵通过softmax得到权重
        text_weights = F.softmax(-text_entropy.t()/text_t, dim=-1) # shape: (n_cls, n_des)
        
        # 第二步：使用文本权重加权平均来计算图像的权重
        # output shape: (n_views, n_des, n_cls)
        # text_weights shape: (n_cls, n_des)
        # 对每个视图，使用文本权重对描述符维度进行加权平均
        # 需要将text_weights转置并扩展以匹配output的形状
        text_weights_expanded = text_weights.t().unsqueeze(0)  # shape: (1, n_des, n_cls)
        # 对每个类别使用对应的文本权重进行加权平均
        weighted_output = (output * text_weights_expanded).sum(dim=1)  # shape: (n_views, n_cls)
        # 计算图像熵
        image_entropy = calculate_batch_entropy(weighted_output)
        # 使用负熵通过softmax得到权重（熵越低，权重越高）
        image_weights = F.softmax(-image_entropy/img_t, dim=-1)

    return image_weights, text_weights

# 基于置信度计算图像和文本的权重（测试版本2）
# 修改：直接使用softmax置信度而不是熵来计算权重
# 参数:
#   output: 模型输出 shape: (n_views, n_prompts, n_classes)
#   img_t: 图像温度参数，用于控制权重分布的平滑度，默认0.5
#   text_t: 文本温度参数，用于控制权重分布的平滑度，默认0.5
# 返回:
#   image_weights: 图像权重 shape: (n_views,)
#   text_weights: 文本权重 shape: (n_classes, n_descriptors)
@torch.no_grad()
def get_entropy_weight_test2(output, img_t=0.5, text_t=0.5):
    with torch.cuda.amp.autocast():
        # 计算图像的权重
        # 对每个视图在所有描述符上取平均，然后计算最大置信度
        image_confidence = output.mean(1).softmax(-1).max(-1)[0]  # shape: (n_views,)
        # 使用置信度通过softmax得到权重（置信度越高，权重越高）
        image_weights = F.softmax(image_confidence/img_t, dim=-1)

        # 计算描述符（文本）的权重
        _, n_des, n_cls = output.shape  # 获取描述符数量和类别数量
        # 创建锚点：使用第一个视图的平均预测作为基准
        anchor = output[0].mean(0)[None, None, :].repeat(n_des, n_cls, 1)
        # 获取每个描述符的输出
        output_des = output[0].unsqueeze(-1)
        # 创建scatter索引
        scatter_indices = torch.arange(n_cls)[None, :, None].repeat(n_des, 1, 1).cuda()
        # 将每个描述符的预测散布到对应的类别位置
        anchor.scatter_(dim=2, index=scatter_indices, src=output_des) # shape: (n_des, n_cls, n_cls)
        # 计算文本置信度（使用最大softmax概率）
        text_confidence = anchor.softmax(-1).max(-1)[0]  # shape: (n_des, n_cls)
        # 使用置信度通过softmax得到权重
        text_weights = F.softmax(text_confidence.t()/text_t, dim=-1) # shape: (n_cls, n_des)

    return image_weights, text_weights

# Sinkhorn算法：用于求解最优传输问题
# 这是一种迭代算法，用于找到两个分布之间的最优传输矩阵
# 参数:
#   K: 核矩阵 (kernel matrix)
#   u: 源分布
#   v: 目标分布
# 返回:
#   T: 最优传输矩阵
def Sinkhorn(K, u, v):
    # 初始化行和列的缩放因子
    r = torch.ones_like(u)
    c = torch.ones_like(v)
    thresh = 1e-2  # 收敛阈值
    # 迭代更新，最多100次
    for i in range(100):
        r0 = r  # 保存上一次的r值用于检查收敛
        # 更新行缩放因子
        r = u / torch.matmul(K, c.unsqueeze(-1)).squeeze(-1)
        # 更新列缩放因子
        c = v / torch.matmul(K.permute(0, 2, 1).contiguous(), r.unsqueeze(-1)).squeeze(-1)
        # 计算误差
        err = (r - r0).abs().mean()
        # 如果收敛则提前退出
        if err.item() < thresh:
            break
    # 计算最优传输矩阵
    T = torch.matmul(r.unsqueeze(-1), c.unsqueeze(-2)) * K
    return T

# 使用最优传输计算加权的logits
# 这是AWT方法的Transportation部分
# 参数:
#   logits: 原始logits shape: (n_views, n_prompts, n_classes)
#   logit_scale: CLIP的logit缩放参数
#   image_weights: 图像权重
#   text_weights: 文本权重
# 返回:
#   weighted_logits: 加权后的logits shape: (1, n_classes)
def optimal_transport(logits, logit_scale, image_weights, text_weights):
    eps = 0.1  # 熵正则化参数
    # 计算相似度（归一化的logits）
    sim = logits / logit_scale.exp()
    sim = sim.permute(2, 0, 1) # 转换为 (n_classes, n_views, n_prompts)

    # 计算Wasserstein距离
    wdist = 1.0 - sim
    with torch.no_grad():
        # 计算核矩阵
        KK = torch.exp(-wdist / eps)
        # 使用Sinkhorn算法求解最优传输矩阵
        T = Sinkhorn(KK, image_weights, text_weights)
        # 转换回原始维度顺序
        T = T.permute(1, 2, 0)
    # 确保传输矩阵没有NaN值
    assert not torch.isnan(T).any()

    # 使用传输矩阵对logits进行加权求和
    return torch.sum(T * logits, dim=(0, 1)).unsqueeze(0)


# AWT评估函数：使用增强、加权和传输方法进行零样本分类评估
# 参数:
#   clip_model: CLIP模型
#   args: 命令行参数
#   order: 方法选择
#   use_test_weight: 是否使用测试版本的权重计算方法
#   weight_method: 权重计算方法选择 ('original', 'test1', 'test2')
@torch.no_grad()
def AWT_evaluation(clip_model, args , order, use_test_weight=False, weight_method='original'):
    # 获取数据集名称和类别名称
    dataset_name = args.test_set
    print("Evaluating: {}".format(dataset_name))
    # 打印当前使用的方法
    if weight_method == 'original':
        weight_desc = "ORIGINAL (Entropy-based)"
    elif weight_method == 'test1':
        weight_desc = "TEST1 (Text-first Entropy)"
    elif weight_method == 'test2':
        weight_desc = "TEST2 (Confidence-based)"
    else:
        weight_desc = "UNKNOWN"
    
    if order==0:
        print(f">>> Using ORIGINAL method: Optimal Transport | Weight: {weight_desc}")
    elif order==1:
        print(f">>> Using ABLATION method: Weighted Cosine Similarity | Weight: {weight_desc}")
    elif order==2:
        print(f">>> Using ABLATION method: OptimalWeighted Cosine Similarity | Weight: {weight_desc}")
    elif order==3:
        print(f">>> Using ABLATION method: Euclidean Distance | Weight: {weight_desc}")
    
    classnames = get_classnames(dataset_name)

    # 加载LLM生成的类别描述
    # ImageNet的变体共享相同的描述文件
    if dataset_name in ['imagenet', 'imagenet_a', 'imagenetv2']:
        description_file = os.path.join(args.descriptor_path, 'imagenet.json')
    else:
        description_file = os.path.join(args.descriptor_path, f'{dataset_name}.json')
    print(f'Using description file: {description_file}')
    # 加载JSON格式的描述文件
    llm_descriptions = json.load(open(description_file))

    # ============== 准备文本特征 ==============
    text_features = []
    # 获取数据集对应的提示模板
    template = CUSTOM_TEMPLATES[dataset_name]
    # 为每个类别生成文本特征
    print(f"Encoding text features for {len(classnames)} classes...")
    for idx, classname in enumerate(classnames):
        prompts = []
        # 使用模板格式化类别名称（将下划线替换为空格）
        prompt = template.format(classname.replace("_", " "))
        # 添加基础提示（不含描述）
        prompts.append(prompt + '.')

        # 添加LLM生成的描述
        assert len(llm_descriptions[classname]) >= args.num_descriptor
        for i in range(args.num_descriptor):
            # 将描述附加到基础提示后
            prompt_desc = prompt + '. ' + llm_descriptions[classname][i]
            prompts.append(prompt_desc)
        # 对所有提示进行分词并移到GPU
        prompts = torch.cat([clip.tokenize(p) for p in prompts]).cuda()

        # 使用CLIP编码文本
        with torch.cuda.amp.autocast():
            text_features.append(clip_model.encode_text(prompts)) # shape: (n_descriptors, d)
        
        # 打印进度
        if (idx + 1) % 10 == 0 or (idx + 1) == len(classnames):
            print(f"  Encoded {idx + 1}/{len(classnames)} classes")

    # 拼接所有类别的文本特征
    text_features = torch.cat(text_features).float() # shape: (n_classes * n_descriptors, d)
    # L2归一化
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    # ==============  为每张图像计算logits ==============
    # 初始化性能指标
    batch_time = AverageMeter('Time', ':6.3f', Summary.NONE)  # 时间统计
    top1 = AverageMeter('Acc@1', ':6.2f', Summary.AVERAGE)  # Top-1准确率

    # 加载预提取的图像特征
    pre_features = torch.load(f"./pre_extracted_feat/{args.arch.replace('/', '')}/seed{args.seed}/{dataset_name}.pth")

    print("number of test samples: {}".format(len(pre_features)))
    # 初始化进度显示器
    progress = ProgressMeter(len(pre_features), [batch_time, top1], prefix='Test: ')

    end = time.time()
    # 遍历所有预提取的图像特征
    for i, (image_features, target) in enumerate(pre_features):
        n_views = image_features.size(0)  # 增强视图的数量
        n_prompt = args.num_descriptor + 1  # 提示数量（1个基础 + N个描述）

        # 计算图像-文本相似度（logits）
        output = clip_model.logit_scale.exp() * image_features @ text_features.t()
        # 重塑为 (n_views, n_prompts, n_classes)
        output = output.view(n_views, -1, n_prompt).permute(0, 2, 1).contiguous()

        # 设置温度参数
        image_temperature = 0.5
        text_temperature = 0.5
        # 基于熵计算图像和文本的权重（Weighting步骤）
        if weight_method == 'test1':
            # 使用测试版本1：先计算文本权重，再用文本权重加权平均计算图像权重
            image_weights, text_weights = get_entropy_weight_test1(output, img_t=image_temperature, text_t=text_temperature)
        elif weight_method == 'test2':
            # 使用测试版本2：直接使用置信度而不是熵
            image_weights, text_weights = get_entropy_weight_test2(output, img_t=image_temperature, text_t=text_temperature)
        else:
            # 使用原始版本：先计算图像权重，文本权重计算独立
            image_weights, text_weights = get_entropy_weight(output, img_t=image_temperature, text_t=text_temperature)
        
        # ========== 消融实验控制 ==========
        if order==0:
            # 原始方法：使用最优传输进行加权聚合（Transportation步骤）
            output_ot = optimal_transport(output, clip_model.logit_scale, image_weights, text_weights)      
        elif order==1:
            # 消融方法：使用加权特征的余弦相似度
            # 1. 对图像视图加权得到加权图像特征 I
            # image_features shape: (n_views, d)
            # image_weights shape: (n_views,)
            weighted_image_features = (image_features * image_weights.unsqueeze(-1)).sum(dim=0)  # shape: (d,)
            weighted_image_features = weighted_image_features / weighted_image_features.norm()  # L2归一化
            
            # 2. 对文本视图加权得到加权文本特征 T
            # text_features shape: (n_classes * n_prompts, d)
            # text_weights shape: (n_classes, n_prompts)
            # 重塑text_features为 (n_classes, n_prompts, d)
            text_features_reshaped = text_features.view(-1, n_prompt, text_features.size(-1))
            # 对每个类别的文本描述加权
            weighted_text_features = (text_features_reshaped * text_weights.unsqueeze(-1)).sum(dim=1)  # shape: (n_classes, d)
            weighted_text_features = weighted_text_features / weighted_text_features.norm(dim=-1, keepdim=True)  # L2归一化
            
            # 3. 计算加权图像特征和加权文本特征的余弦相似度
            output_ot = clip_model.logit_scale.exp() * (weighted_image_features @ weighted_text_features.t()).unsqueeze(0)  # shape: (1, n_classes)
        elif order==2:
            # 优化的余弦相似度做法：图加权 → 与每个文本计算相似度 → 文权重加权求和
            # 1. 对图像视图加权得到加权图像特征 Ix
            # image_features shape: (n_views, d)
            # image_weights shape: (n_views,)
            weighted_image_features = (image_features * image_weights.unsqueeze(-1)).sum(dim=0)  # shape: (d,)
            weighted_image_features = weighted_image_features / weighted_image_features.norm()  # L2归一化
            
            # 2. 加权图像特征 Ix 与每个文本特征 Tj 计算余弦相似度
            # text_features shape: (n_classes * n_prompts, d)
            # 重塑为 (n_classes, n_prompts, d)
            text_features_reshaped = text_features.view(-1, n_prompt, text_features.size(-1))
            # 计算 cos(Ix, Tj) for all j
            # weighted_image_features shape: (d,)
            # text_features_reshaped shape: (n_classes, n_prompts, d)
            similarities = clip_model.logit_scale.exp() * (weighted_image_features @ text_features_reshaped.permute(0, 2, 1))  # shape: (n_classes, n_prompts)
            
            # 3. 使用文本权重对相似度进行加权求和: Σ(WTj * cos(Ix, Tj))
            # text_weights shape: (n_classes, n_prompts)
            # similarities shape: (n_classes, n_prompts)
            output_ot = (text_weights * similarities).sum(dim=1).unsqueeze(0)  # shape: (1, n_classes)

        # 计算准确率
        acc1, = accuracy(output_ot, target, topk=(1,))
        top1.update(acc1[0], 1)

        # 更新时间统计
        batch_time.update(time.time() - end)
        end = time.time()

        # 定期显示进度
        if (i+1) % args.print_freq == 0:
            progress.display(i)

    # 显示最终结果
    print(f'\n *  {dataset_name}')
    progress.display_summary()

    return top1.avg


# 多模态匹配评估函数：待测试[图-文]与检索到的[图-文]进行匹配评估
# 参数:
#   clip_model: CLIP模型
#   args: 命令行参数
@torch.no_grad()
def Multimodal_Enhanced_Classification_evaluation(clip_model, args):
    # 获取数据集名称
    dataset_name = args.test_set
    print("Evaluating Multimodal Enhanced Classification: {}".format(dataset_name))
    print(">>> Using [Test Image-Text] vs [Retrieved Image-Text] matching")
    
    # 加载预提取的多模态特征
    save_dir = f"./pre_extracted_feat/{args.arch.replace('/', '')}/seed{args.seed}"
    retrieved_path = os.path.join(save_dir, f"{dataset_name}_retrieved.pth")
    test_path = os.path.join(save_dir, f"{dataset_name}_test.pth")
    
    print(f"Loading retrieved [image-text] features from: {retrieved_path}")
    print(f"Loading test [image-text] features from: {test_path}")
    
    retrieved_data = torch.load(retrieved_path)  # [(multimodal_features, target), ...] - 检索到的[图-文]
    test_data = torch.load(test_path)            # [(multimodal_features, target), ...] - 待测试的[图-文]
    
    print(f"Retrieved [image-text] samples: {len(retrieved_data)}")
    print(f"Test [image-text] samples: {len(test_data)}")
    
    # 初始化统计器
    batch_time = AverageMeter('Time', ':6.3f', Summary.NONE)
    top1 = AverageMeter('Acc@1', ':6.2f', Summary.AVERAGE)
    progress = ProgressMeter(len(test_data), [batch_time, top1], prefix='Test: ')
    
    end = time.time()
    
    # 对每个待测试样本进行匹配
    for i, (test_features, target) in enumerate(test_data):
        # test_features shape: (n_views, 2*d) - 包含文本和图像特征的拼接 [T'j, I'j]
        # target: 真实标签
        
        # 计算与所有检索样本的相似度
        similarities = []
        
        for retrieved_features, _ in retrieved_data:
            # retrieved_features shape: (n_views, 2*d) - 包含文本和图像特征的拼接 [Ti, Ii]
            
            # 计算两组多模态特征的权重（基于熵的权重计算）
            # 对于待测试的[图-文]特征
            test_weights = F.softmax(-calculate_batch_entropy(test_features) / 0.5, dim=0)
            # 对于检索到的[图-文]特征
            retrieved_weights = F.softmax(-calculate_batch_entropy(retrieved_features) / 0.5, dim=0)
            
            # 加权平均得到代表性的多模态特征
            weighted_test = (test_features * test_weights.unsqueeze(-1)).sum(dim=0)  # 加权的待测试[图-文]特征
            weighted_retrieved = (retrieved_features * retrieved_weights.unsqueeze(-1)).sum(dim=0)  # 加权的检索[图-文]特征
            
            # L2归一化
            weighted_test = weighted_test / weighted_test.norm()
            weighted_retrieved = weighted_retrieved / weighted_retrieved.norm()
            
            # 计算多模态特征间的余弦相似度
            # 这实现了：待测试[图-文] 与 检索到的[图-文] 的匹配
            similarity = clip_model.logit_scale.exp() * torch.dot(weighted_test, weighted_retrieved)
            similarities.append(similarity)
        
        # 转换为张量
        similarities = torch.stack(similarities)
        
        # 根据检索样本的标签来确定预测
        retrieved_labels = torch.tensor([label for _, label in retrieved_data])
        
        # 找到最相似的检索样本的标签作为预测
        max_idx = torch.argmax(similarities)
        predicted_label = retrieved_labels[max_idx].unsqueeze(0)
        
        # 计算准确率
        acc1, = accuracy(predicted_label.unsqueeze(0).float(), target.unsqueeze(0), topk=(1,))
        top1.update(acc1[0], 1)
        
        # 更新时间统计
        batch_time.update(time.time() - end)
        end = time.time()
        
        # 定期显示进度
        if (i+1) % args.print_freq == 0:
            progress.display(i)
    
    # 显示最终结果
    print(f'\n *  {dataset_name} (Multimodal Enhanced Classification)')
    progress.display_summary()
    
    return top1.avg


# 主工作函数
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
    
    # 开始测试 - 多模态增强分类评估
    print("\n" + "="*80)
    print("MULTIMODAL ENHANCED CLASSIFICATION EVALUATION")
    print("="*80 + "\n")
    
    # 执行多模态增强分类评估
    Multimodal_Enhanced_Classification_evaluation(clip_model, args)

# 主程序入口
if __name__ == '__main__':
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description='Multimodal Enhanced Classification evaluation')
    parser.add_argument('--test_set', type=str, help='dataset name')  # 测试数据集名称
    parser.add_argument('-a', '--arch', metavar='ARCH', default='ViT-B/16')  # 模型架构
    parser.add_argument('-p', '--print-freq', default=1000, type=int, metavar='N', help='print frequency (default: 10)')  # 打印频率
    parser.add_argument('--seed', type=int, default=0)  # 随机种子
    parser.add_argument('--descriptor_path', type=str)  # 描述文件路径
    parser.add_argument('--num_descriptor', type=int, default=50)  # 每个类别使用的描述数量

    args = parser.parse_args()
    # 设置随机种子以保证可重复性
    set_random_seed(args.seed)
    # 启动主工作函数
    main_worker(args)
