import torch 
import os 
import argparse 
import json 
from tqdm import tqdm  
from termcolor import colored  
from collections import Counter 
from utils.configuration import setup_config, seed_everything 
from utils.fileios import dump_json, load_json, dump_txt  

from data import DATA_STATS, PROMPTERS, DATA_DISCOVERY  
from data.prompt_identify import prompts_howto  
from agents.vqa_bot import VQABot  
from agents.llm_bot import LLMBot 
from agents.mllm_bot import MLLMBot
from cvd.cdv_captioner import CDVCaptioner  
from retrieval.multimodal_retrieval import MultimodalRetrieval 
from fast_slow_thinking_system import FastSlowThinkingSystem
from utils.util import is_similar
import re 
import hashlib
import time
from collections import defaultdict
import numpy as np


DEBUG = False  # 设置调试模式为关闭状态


def cint2cname(label: int, cname_sheet: list):
    """将类别整数索引转换为类别名称"""
    return cname_sheet[label]


def extract_superidentify(cfg, individual_results):
    """从个体识别结果中提取超类识别结果"""
    words = []  # 初始化单词列表
    for v in individual_results.values():  # 遍历所有个体识别结果
        this_word = v.split(' ')[-1]  # 取最后一个单词作为类别标识
        words.append(this_word.lower())  # 转换为小写并添加到列表
    word_counts = Counter(words)  # 统计每个单词的出现次数
    # print(f"extract_superidentify 中每个单词出现次数: {word_counts}")
    if cfg['dataset_name'] == 'pet':  # 如果是宠物数据集
        return [super_name for super_name, _ in word_counts.most_common(2)]  # 返回出现次数最多的2个超类
    else:  # 其他数据集
        return [super_name for super_name, _ in word_counts.most_common(1)]  # 返回出现次数最多的1个超类



def extract_python_list(text):
    """从文本中提取Python列表格式的内容"""
    pattern = r"\[(.*?)\]"  # 定义匹配方括号内容的正则表达式
    matches = re.findall(pattern, text)  # 查找所有匹配的内容
    return matches  # 返回匹配结果列表


def trim_result2json(raw_reply: str):
    """
    the raw_answer is a dirty output from LLM following our template.
    this function helps to extract the target JSON content contained in the
    output.
    """
    # 从LLM的原始输出中提取JSON格式的内容
    if raw_reply.find("Output JSON:") >= 0:  # 如果包含"Output JSON:"标记
        answer = raw_reply.split("Output JSON:")[1].strip()  # 提取标记后的内容
    else:  # 否则直接使用原始内容
        answer = raw_reply.strip()  # 去除首尾空白字符

    if not answer.startswith('{'): answer = '{' + answer  # 如果开头不是{，则添加

    if not answer.endswith('}'): answer = answer + '}'  # 如果结尾不是}，则添加

    # json_answer = json.loads(answer)  # 注释掉的JSON解析代码
    return answer  # 返回处理后的JSON字符串


def clean_name(name: str):
    """清理类别名称，统一格式"""
    name = name.title() 
    name = name.replace("-", " ")  
    name = name.replace("'s", "") 
    return name  


def extract_names(gussed_names, clean=True):
    """从猜测的名称列表中提取和清理名称"""
    gussed_names = [name.strip() for name in gussed_names]
    if clean:  # 如果需要清理
        gussed_names = [clean_name(name) for name in gussed_names]  
    gussed_names = list(set(gussed_names))  # 去重并转换为列表
    return gussed_names  # 返回处理后的名称列表


def how_to_distinguish(bot, prompt):
    """询问LLM如何区分不同类别"""
    reply = bot.infer(prompt, temperature=0.1) 
    used_tokens = bot.get_used_tokens()  
    print(f"llm used_tokens: {used_tokens},")
    print(20*"=")  
    print(reply)  #
    print(20*"=") 

    return reply  

def main_identify(cfg, bot, data_disco):
    """识别图像的超类"""
    json_super_classes = {}             # img: [attr1, attr2, ..., attrN] - 初始化超类结果字典

    # print(f"现在开始遍历发现集data_disco: {data_disco}")
    for idx, (img, label) in tqdm(enumerate(data_disco)):  # 遍历发现数据集中的图像和标签
        # prompt_identify = "Question: What is the main object in this image (choose from: Car, Flower, or Pokemon)? Answer:"
        
        prompt_identify = "Question: What is the category (car, bird, flower, dog, cat, or Pokemon) of the main object in this image? Answer:" 

        reply, trimmed_reply = bot.describe_attribute(img, prompt_identify) 
        trimmed_reply = trimmed_reply.lower()  
        json_super_classes[str(idx)] = trimmed_reply 

        # DEBUG mode - 调试模式
        if DEBUG and idx >= 2: 
            break  
    # print(f"main_identify 识别结果: {json_super_classes}")
    return json_super_classes  # 返回超类识别结果


def main_describe(cfg, bot, data_disco, prompter, cname_sheet):
    """
    1.调用VQA模型为每个属性生成对应的描述
    2.生成LLMpromot描述
    """
    json_attrs = {}             # img: [attr1, attr2, ..., attrN] - 初始化属性结果字典
    json_llm_prompts = {}       # img: LLM-prompt (has all attrs) - 初始化LLM提示字典

    # 这里是训练集，预先定义好的
    for idx, (img, label) in tqdm(enumerate(data_disco)): 
        if cfg['dataset_name'] == 'pet': 
            # first check what is the animal
            pet_prompt = "Questions: What is the animal in this photo (dog or car)? Answer:"
            pet_re, pet_trimmed_re = bot.describe_attribute(img, pet_prompt) 
            pet_trimmed_re = pet_trimmed_re.lower() 
            # print(pet_trimmed_re)
            if 'dog' in pet_trimmed_re:
                prompter.set_superclass('dog')
            else:
                prompter.set_superclass('cat')

        # generate attributes and per-attribute prompts for VQA bot  获得属性列表
        attrs = prompter.get_attributes()
        # 生成对应属性的promot描述，让LLM进行描述
        attr_prompts = prompter.get_attribute_prompt() 
        if len(attrs) != len(attr_prompts):  # 检查属性列表和提示列表长度是否一致
            raise IndexError("Attribute list should have the same length as attribute prompts")

        print(f"当前idx:{idx}: label={label}")

        iname = cint2cname(label, cname_sheet)
        iname += f"_{idx}"  # 对应用多少个样本作为训练集
        json_attrs[iname] = []  # 初始化该图像的属性列表

        # describe each attrs - 描述每个属性
        pair_attr_reply = []    # (attr1: prompt) - 初始化属性-值对列表
        for attr, p_attr in zip(attrs, attr_prompts):  # 遍历属性和对应的prompt
            re_attr, trimmed_re_attr = bot.describe_attribute(img, p_attr) 
            # print(f"调用bot.describe_attribute得到的reply:{re_attr} \n tritrimmed_re_attr:{trimmed_re_attr}")
            pair_attr_reply.append([attr, trimmed_re_attr])
            json_attrs[iname].append(trimmed_re_attr)  # 将属性值添加到对应的类别描述中
        
        print(f'获得的VQA pair_attr_reply: {pair_attr_reply}\n json_attrs: {json_attrs}')
        # generate LLM prompt - 生成LLM提示
        llm_prompt = prompter.get_llm_prompt(pair_attr_reply)  # 根据属性-值对生成LLM提示
        json_llm_prompts[iname] = llm_prompt 
        print(f'json_llm_prompts: {json_llm_prompts}')
        print(30 * '=')
        print(iname + f" with label {label}") 
        print(30 * '=')
        # print()  # 打印空行
        # print(f"llm_prompt: {llm_prompt}")  # 打印LLM提示
        # print()  # 打印空行
        # print('END' + 30 * '=')  # 打印结束分隔线
        # print()  # 打印空行

        # DEBUG mode - 调试模式
        if DEBUG and idx >= 2:  # 如果开启调试模式且处理了2个以上样本
            break  # 跳出循环

    return json_attrs, json_llm_prompts  # 返回属性结果和LLM提示


def main_guess(cfg, bot, reasoning_prompts):
    """主要猜测函数：基于属性描述推理类别名称"""
    prompt_list = reasoning_prompts  
    replies_raw = {}  
    replies_json_to_save = {}  

    # LLM inferring - LLM推理
    for i, (key, prompt) in tqdm(enumerate(prompt_list.items())):  
        raw_reply = bot.infer(prompt, temperature=0.9)  # use a high temperature for better diversity
        used_tokens = bot.get_used_tokens()  # 获取使用的token数量

        replies_raw[key] = raw_reply  # 将原始回复存储到字典

        print(30 * '=')  # 打印分隔线
        print(f"\t\tinferring [{i}] for {key} used tokens = {used_tokens}") 
        print(30 * '=')  
        print("Raw----")  
        print(raw_reply)  
        print()  

        jsoned_reply = trim_result2json(raw_reply=raw_reply) 

        replies_json_to_save[key] = jsoned_reply  

        print("Trimed----")  
        print(jsoned_reply)
        print()  # 打印空行
        print('END' + 30 * '=')  
        print()  

        # DEBUG - 调试
        if DEBUG and i >= 2:  # 如果开启调试模式且处理了2个以上样本
            break 

    print(30 * '=')  
    print(f"\t\t Finish Discovering, token consumed {bot.get_used_tokens()}"  
          f" = ${bot.get_used_tokens()*0.001*0.002}") 
    print(30 * '=')  
    print('END' + 30 * '=')  
    print()  
    return replies_raw, replies_json_to_save 


def post_process(cfg, jsoned_replies):
    """后处理函数：清理和整理LLM推理结果"""
    reply_list = []  
    num_of_failures = 0  
    # duplicated dict - 重复字典
    for k, v in jsoned_replies.items():  # 遍历JSON回复字典
        print(k)  
        print(v) 
        print()
        print() 
        try:  
            v_json = json.loads(v)  
            reply_list.append(v_json)
        except json.JSONDecodeError:  
            print(f"Failed to decode JSON for key: {k}") 
            num_of_failures += 1  
            continue  

        # v_json = json.loads(v) 
        # reply_list.append(v_json) 

    guessed_names = [] 
    for item in reply_list: 
        guessed_names.extend(list(item.keys()))  

    guessed_names = extract_names(guessed_names, clean=False) 

    if cfg['dataset_name'] in ['pet', 'dog']: 
        clean_gussed_names = []  
        for aitem in guessed_names:
            clean_gussed_names.extend(aitem.split(','))  
        clean_gussed_names = [name.strip() for name in clean_gussed_names]  
        guessed_names = clean_gussed_names  

    print(30 * '=') 
    print(f"\t\t Finished Post-processing")  
    print(30 * '=')  

    print(f"\t\t ---> total discovered names = {len(guessed_names)}")  
    print(guessed_names)  
    print()  
    print(f"\t\t ---> total discovered names = {len(guessed_names)}")  
    print(f"\t\t ---> number of failure entries = {num_of_failures}") 

    print('END' + 30 * '=')  
    print()  
    return guessed_names 


def load_train_samples(cfg, kshot=None):
    """加载K-shot训练样本，返回 {category: [image_paths]}。
    优先从 cfg['path_train_samples'] (JSON) 读取；否则从 cfg['train_root'] 目录扫描。
    """
    samples = {}
    if 'path_train_samples' in cfg and os.path.exists(cfg['path_train_samples']):
        try:
            samples = load_json(cfg['path_train_samples'])
        except Exception as e:
            print(f"failed to load path_train_samples: {cfg['path_train_samples']}, err={e}")
            samples = {}
    elif 'train_root' in cfg and os.path.isdir(cfg['train_root']):
        train_root = cfg['train_root']
        valid_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
        for cname in sorted(os.listdir(train_root)):
            cdir = os.path.join(train_root, cname)
            if not os.path.isdir(cdir):
                continue
            imgs = []
            for fname in sorted(os.listdir(cdir)):
                fpath = os.path.join(cdir, fname)
                ext = os.path.splitext(fname)[1].lower()
                if os.path.isfile(fpath) and ext in valid_exts:
                    imgs.append(fpath)
            if imgs:
                samples[cname] = imgs
    else:
        raise FileNotFoundError("Neither cfg['path_train_samples'] nor cfg['train_root'] is valid.")

    if kshot is not None:
        trimmed = {}
        for cat, paths in samples.items():
            trimmed[cat] = paths[:kshot]
        return trimmed
    return samples


def build_gallery(cfg, mllm_bot, captioner, retrieval, kshot=5,region_num=3, superclass=None, data_discovery=None):
    """构建多模态类别模板库并保存到JSON(向量转list)。"""

    # 读取训练样本
    k = kshot if kshot is not None else int(str(cfg.get('k_shot', '3')))
    # train_samples = load_train_samples(cfg, kshot=k)
    train_samples = defaultdict(list)
    for name, path in data_discovery.subcat_to_sample.items():
        train_samples[name].append(path)
    print(f"loaded train samples for {len(train_samples)} classes, kshot={k}")
    print(f"train_samples: {train_samples}") 

    # 构建模板库
    gallery = retrieval.build_template_gallery(mllm_bot, train_samples, captioner, superclass, kshot, region_num)
    
    return gallery

if __name__ == "__main__":
    """
    CUDA_VISIBLE_DEVICES=1 python discovering.py --mode=build_gallery --config_file_env=./configs/env_machine.yml --config_file_expt=./configs/expts/dog120_all.yml --kshot=5 --region_num=3 --superclass=dog  --gallery_out=./experiments/dog120/gallery/dog120_gallery_concat_atten.json --fusion_method=concat 2>&1 | tee ./logs/build_gallery_dog_concat_atten.log
    """
    parser = argparse.ArgumentParser(description='Discovery', formatter_class=argparse.ArgumentDefaultsHelpFormatter) 

    parser.add_argument('--mode',  
                        type=str, 
                        default='describe', 
                        choices=['identify', 'howto', 'describe', 'guess', 'postprocess', 'build_gallery', 'build_knowledge_base', 'classify', 'evaluate', 'fastonly', 'slowonly', 'fast_slow', 'fast_slow_infer', 'fast_slow_classify'],  # 可选值列表
                        help='operating mode for each stage')  
    parser.add_argument('--config_file_env',  
                        type=str,  
                        default='./configs/env_machine.yml',  # 默认配置文件路径
                        help='location of host environment related config file')  
    parser.add_argument('--config_file_expt',  # 添加实验配置文件参数
                        type=str,  
                        default='./configs/expts/bird200_all.yml', 
                        help='location of host experiment related config file') 
    # arguments for control experiments - 控制实验的参数
    parser.add_argument('--num_per_category',  # 添加每个类别的样本数量参数
                        type=str, 
                        default='3',  
                        choices=['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', 'random'], 
                        )
    # build_gallery 相关
    parser.add_argument('--kshot', type=int, default=None, help='shots per class when building gallery (override cfg)')
    parser.add_argument('--region_num', type=int, default=None, help='region selelct per class when building gallery (override cfg)')
    parser.add_argument('--superclass', type=str, default=None, help='superclass for CDV prompts (override cfg)')
    parser.add_argument('--gallery_out', type=str, default=None, help='path to save built gallery json')
    parser.add_argument('--fusion_method', type=str, default='concat', help='fusion method')
    
    # 快慢思考系统相关参数
    parser.add_argument('--knowledge_base_dir', type=str, default='./knowledge_base', help='knowledge base directory')
    parser.add_argument('--query_image', type=str, default=None, help='query image path for classification')
    parser.add_argument('--test_data_dir', type=str, default=None, help='test data directory for evaluation')
    parser.add_argument('--results_out', type=str, default='./results.json', help='output path for results')
    parser.add_argument('--use_slow_thinking', type=bool, default=None, help='force use slow thinking (None for auto)')
    parser.add_argument('--confidence_threshold', type=float, default=0.8, help='confidence threshold for fast thinking')
    parser.add_argument('--similarity_threshold', type=float, default=0.7, help='similarity threshold for trigger mechanism')
    parser.add_argument('--enable_mllm_intermediate_judge', action='store_true', default=False, help='enable MLLM intermediate judge between fast and slow thinking (for ablation studies)')
    
    # 快慢思考推理与分类分离相关参数
    parser.add_argument('--infer_dir', type=str, default=None, help='directory to save inference results (for fast_slow_infer mode)')
    parser.add_argument('--classify_dir', type=str, default=None, help='directory to save classification results (for fast_slow_classify mode)')

    args = parser.parse_args()  
    print(colored(args, 'blue'))  

    cfg = setup_config(args.config_file_env, args.config_file_expt)  
    print(colored(cfg, 'yellow')) 

    # drop the seed - 设置随机种子
    seed_everything(cfg['seed']) 

    expt_id_suffix = f"_{args.num_per_category}"  # 创建实验ID后缀

    if args.mode == 'build_knowledge_base':
        """
        构建快慢思考系统的知识库
        CUDA_VISIBLE_DEVICES=3 python discovering.py --mode=build_knowledge_base --config_file_env=./configs/env_machine.yml --config_file_expt=./configs/expts/dog120_all.yml --num_per_category=10 --knowledge_base_dir=/data/yjx/MLLM/Try_again/experiments/dog120/knowledge_base 2>&1 | tee ./logs/build_knowledge_base_dog120.log
        """
        # 初始化快慢思考系统
        system = FastSlowThinkingSystem(
            model_tag=cfg['model_size_mllm'],
            model_name=cfg['model_size_mllm'],
            device='cuda' if cfg['host'] in ["xiao"] else 'cpu',
            cfg=cfg,
            enable_mllm_intermediate_judge=args.enable_mllm_intermediate_judge
        )
        
        # 加载训练样本
        data_discovery = DATA_DISCOVERY[cfg['dataset_name']](cfg, folder_suffix=expt_id_suffix)
        train_samples = defaultdict(list)
        # {"Chihuaha": "./datasets/dogs_120/images_discovery_all_3/000.Chihuaha_000000.jpg", "Poodle": "./datasets/dogs_120/images_discovery_all_3/001.Poodle_000000.jpg", ...}
        for name, path in data_discovery.subcat_to_sample.items():
            for p in path:
                train_samples[name].append(p)
        
        print(f"构建知识库，包含 {len(train_samples)} 个类别, dog datasets:{len(DATA_STATS[cfg['dataset_name']]['class_names'])}")
        
        # 构建知识库
        system.load_knowledge_base(args.knowledge_base_dir) # 方便构建stats
        image_kb, text_kb = system.build_knowledge_base(
            train_samples, 
            save_dir=args.knowledge_base_dir,
            augmentation=True
        )
        
        print(f"知识库构建完成，保存到: {args.knowledge_base_dir}")
    
    elif args.mode == 'classify':
        """
        使用快慢思考系统进行单张图像分类
        CUDA_VISIBLE_DEVICES=1 python discovering.py --mode=classify --config_file_env=./configs/env_machine.yml --config_file_expt=./configs/expts/dog120_all.yml --query_image=./test_image.jpg --knowledge_base_dir=/data/yjx/MLLM/Try/experiments/dog120/knowledge_base 2>&1 | tee ././logs/testfast.log
        """
        if args.query_image is None:
            raise ValueError("请提供查询图像路径 --query_image")
        
        # 初始化快慢思考系统
        system = FastSlowThinkingSystem(
            model_tag=cfg['model_size_mllm'],
            model_name=cfg['model_size_mllm'],
            device='cuda' if cfg['host'] in ["xiao"] else 'cpu',
            cfg=cfg,
            enable_mllm_intermediate_judge=args.enable_mllm_intermediate_judge
        )
        
        # 加载知识库
        system.load_knowledge_base(args.knowledge_base_dir)
        
        # 分类图像
        result = system.classify_single_image(
            args.query_image,
            use_slow_thinking=args.use_slow_thinking
        )
        
        # 保存结果
        system.save_results([result], args.results_out)
        
        print(f"分类结果: {result['final_prediction']} (置信度: {result['final_confidence']:.4f})")
        print(f"使用慢思考: {result.get('used_slow_thinking', False)}")
        print(f"结果已保存到: {args.results_out}")

    elif args.mode == 'evaluate':
        """
        在测试数据集上评估快慢思考系统
        CUDA_VISIBLE_DEVICES=1 python discovering.py --mode=evaluate --config_file_env=./configs/env_machine.yml --config_file_expt=./configs/expts/dog120_all.yml --test_data_dir=./test_data --knowledge_base_dir=./knowledge_base_dog120 --results_out=./evaluation_results.json
        """
        if args.test_data_dir is None:
            raise ValueError("请提供测试数据目录 --test_data_dir")
        
        # 初始化快慢思考系统
        system = FastSlowThinkingSystem(
            model_tag=cfg['model_size_mllm'],
            model_name=cfg['model_size_mllm'],
            device='cuda' if cfg['host'] in ["xiao"] else 'cpu',
            cfg=cfg,
            enable_mllm_intermediate_judge=args.enable_mllm_intermediate_judge
        )
        
        # 加载知识库
        system.load_knowledge_base(args.knowledge_base_dir)
        
        # 构建测试样本
        test_samples = defaultdict(list)
        for class_name in os.listdir(args.test_data_dir):
            class_dir = os.path.join(args.test_data_dir, class_name)
            if os.path.isdir(class_dir):
                for img_name in os.listdir(class_dir):
                    if img_name.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                        img_path = os.path.join(class_dir, img_name)
                        test_samples[class_name].append(img_path)
        
        print(f"测试数据集包含 {len(test_samples)} 个类别")
        
        # 评估系统
        evaluation_result = system.evaluate_on_dataset(
            test_samples,
            use_slow_thinking=args.use_slow_thinking
        )
        
        # 保存评估结果
        system.save_results([evaluation_result], args.results_out)
        
        print(f"评估完成，准确率: {evaluation_result['accuracy']:.4f}")
        print(f"快思考比例: {evaluation_result['fast_thinking_ratio']:.4f}")
        print(f"慢思考比例: {evaluation_result['slow_thinking_ratio']:.4f}")
        print(f"结果已保存到: {args.results_out}")
    
    elif args.mode == 'fastonly':
        """
        CUDA_VISIBLE_DEVICES=2 python discovering.py --mode=fastonly --config_file_env=./configs/env_machine.yml --config_file_expt=./configs/expts/dog120_all.yml --test_data_dir=/data/yjx/MLLM/UniFGVR/datasets/dogs_120/images_discovery_all_10 --knowledge_base_dir=/data/yjx/MLLM/Try_again/experiments/dog120/knowledge_base --results_out=./logs/fastonly_eval.json 2>&1 | tee ./logs/fastonly_eval_lcb.log
        """

        # 初始化系统（仅用于加载组件），随后只用fast模块
        system = FastSlowThinkingSystem(
            model_tag=cfg['model_size_mllm'],
            model_name=cfg['model_size_mllm'],
            device='cuda' if cfg['host'] in ["xiao"] else 'cpu',
            cfg=cfg,
            enable_mllm_intermediate_judge=args.enable_mllm_intermediate_judge
        )
        # 加载知识库
        system.load_knowledge_base(args.knowledge_base_dir)

        # 构建测试样本
        test_samples = {}
        img_root = args.test_data_dir
        class_folders = os.listdir(args.test_data_dir)
        for i in range(len(class_folders)):
            cat_name = class_folders[i].split('-')[-1].replace('_', ' ')
            # print(f'cat name:{cat_name}')
            img_path = os.path.join(img_root, class_folders[i])
            file_names = os.listdir(img_path)
            # print(f'img_path:{img_path}\tfilename:{file_names}')
            for name in file_names:
                path = os.path.join(img_path,name)
                if cat_name not in test_samples:
                    test_samples[cat_name] = []
                test_samples[cat_name].append(path)

        print(f'test sample:{test_samples}')
        print(f"[fastonly] 测试数据集包含 {len(test_samples)} 个类别")
        # 仅使用快思考评估
        fast_module = system.fast_thinking
        correct = 0
        total = 0
        correct_slow_true = 0    # 正确且需要 slow thinking
        correct_slow_false = 0   # 正确但不需要 slow thinking 预期
        error_slow_true = 0      # 错误且需要 slow thinking 预期
        error_slow_false = 0     # 错误但不需要 slow thinking
        for true_cat, paths in test_samples.items():
            for path in paths:
                try:
                    fast_res = fast_module.fast_thinking_pipeline(path, top_k=5)
                    # 使用融合Top-1作为fast-only预测，兼容旧逻辑兜底
                    pred = fast_res.get('predicted_fast') or fast_res.get('fused_top1') or fast_res.get('predicted_category') or fast_res.get('img_category', 'unknown')
                    ok = is_similar(pred, true_cat, threshold=0.5)
                    if ok:
                        print(f"succ. pred cate:{pred}, true cate:{true_cat}, need_slow_thinking:{fast_res['need_slow_thinking']}")
                        if fast_res['need_slow_thinking']:
                            correct_slow_true+=1
                        else:
                            correct_slow_false+=1
                        correct += 1
                    else:
                        print(f"failed. pred cate:{pred}, true cate:{true_cat}, need_slow_thinking:{fast_res['need_slow_thinking']}")
                        if fast_res['need_slow_thinking']:
                            error_slow_true += 1
                        else:
                            error_slow_false += 1
                    total += 1

                except Exception as e:
                    print(f'Exception:{e}')
                    total += 1


        acc = correct / total if total > 0 else 0.0
        print(f"✅ 正确预测总数: {correct}")
        print(f"  - 其中需要 slow thinking: {correct_slow_true}")
        print(f"  - 其中不需要 slow thinking: {correct_slow_false}")

        print(f"❌ 错误预测总数: {total - correct}")
        print(f"  - 其中需要 slow thinking: {error_slow_true}")
        print(f"  - 其中不需要 slow thinking: {error_slow_false}")
        print(f"[fastonly] 准确率: {acc:.4f} ({correct}/{total})")
    elif args.mode == 'slowonly':
        """
        CUDA_VISIBLE_DEVICES=3 python discovering.py --mode=slowonly --config_file_env=./configs/env_machine.yml --config_file_expt=./configs/expts/dog120_all.yml --test_data_dir=/data/yjx/MLLM/UniFGVR/datasets/dogs_120/images_discovery_all_10 --knowledge_base_dir=/data/yjx/MLLM/Try/experiments/dog120/knowledge_base --results_out=./logs/slowonly_eval.json 2>&1 | tee ./logs/slowonly_eval.log
        """

        # 初始化系统（仅用于加载组件），随后只用slow模块
        system = FastSlowThinkingSystem(
            model_tag=cfg['model_size_mllm'],
            model_name=cfg['model_size_mllm'],
            device='cuda' if cfg['host'] in ["xiao"] else 'cpu',
            cfg=cfg,
            enable_mllm_intermediate_judge=args.enable_mllm_intermediate_judge
        )
        # 加载知识库
        system.load_knowledge_base(args.knowledge_base_dir)

        # 构建测试样本
        test_samples = {}
        img_root = args.test_data_dir
        class_folders = os.listdir(args.test_data_dir)
        for i in range(len(class_folders)):
            cat_name = class_folders[i].split('-')[-1].replace('_', ' ')
            img_path = os.path.join(img_root, class_folders[i])
            file_names = os.listdir(img_path)
            for name in file_names:
                path = os.path.join(img_path,name)
                if cat_name not in test_samples:
                    test_samples[cat_name] = []
                test_samples[cat_name].append(path)

        print(f'test sample:{test_samples}')
        print(f"[slowonly] 测试数据集包含 {len(test_samples)} 个类别")
        
        # 仅使用慢思考评估
        slow_module = system.slow_thinking
        fast_module = system.fast_thinking  # 慢思考需要快思考结果作为输入
        correct = 0
        total = 0
        
        for true_cat, paths in test_samples.items():
            for path in paths:
                try:
                    # 先执行快思考获取结果（慢思考需要这个输入）
                    fast_res = fast_module.fast_thinking_pipeline(path, top_k=5)
                    
                    # 执行慢思考
                    slow_res = slow_module.slow_thinking_pipeline(path, fast_res, top_k=5)
                     
                    # 使用慢思考的最终预测
                    pred = slow_res.get('predicted_category', 'unknown')
                    ok = is_similar(pred, true_cat, threshold=0.5)
                    
                    if ok:
                        print(f"succ. pred cate:{pred}, true cate:{true_cat}, confidence:{slow_res.get('confidence', 0):.4f}")
                        correct += 1
                    else:
                        print(f"failed. pred cate:{pred}, true cate:{true_cat}, confidence:{slow_res.get('confidence', 0):.4f}")
                    
                    total += 1

                except Exception as e:
                    print(f'Exception:{e}')
                    total += 1

        acc = correct / total if total > 0 else 0.0
        print(f"✅ 正确预测总数: {correct}")
        print(f"❌ 错误预测总数: {total - correct}")
        print(f"[slowonly] 准确率: {acc:.4f} ({correct}/{total})")
        
    # fast_slow 模式：对测试集执行“快思考→必要时慢思考→最终融合”的整体验证
    elif args.mode == 'fast_slow':  # 进入 fast_slow 评估分支
        """
        CUDA_VISIBLE_DEVICES=2 python discovering.py --mode=fast_slow --config_file_env=./configs/env_machine.yml --config_file_expt=./configs/expts/dog120_all.yml --test_data_dir=/data/yjx/MLLM/UniFGVR/datasets/dogs_120/images_discovery_all_1 --knowledge_base_dir=/data/yjx/MLLM/Try_again/experiments/dog120/knowledge_base --results_out=./logs/fast_and_slow_eval.json 2>&1 | tee ./logs/fast_and_slow_update_lcb_1_context256.log
        """  # 用法示例：演示如何从命令行启动该模式

        # 初始化完整的快慢思考系统（内部会初始化知识库构建器、快/慢思考模块、MLLM等）
        system = FastSlowThinkingSystem(
            model_tag=cfg['model_size_mllm'],  # 使用配置中的多模态大模型标识
            model_name=cfg['model_size_mllm'], # 模型名称（与 tag 一致）
            device='cuda' if cfg['host'] in ["xiao"] else 'cpu',  # 按主机配置选择设备
            cfg=cfg,  # 传递完整实验配置
            enable_mllm_intermediate_judge=args.enable_mllm_intermediate_judge  # 是否启用MLLM中间判别（可做消融）
        )
        # 加载已构建好的知识库（图像/文本向量、统计信息等），供检索与判别使用
        system.load_knowledge_base(args.knowledge_base_dir)

        # 构建测试样本映射：{ 真值类别: [图像路径, ...] }
        test_samples = {}  # 用于保存每个类别对应的所有测试图片
        img_root = args.test_data_dir  # 测试集根目录
        class_folders = os.listdir(args.test_data_dir)  # 列出类别子目录
        for i in range(len(class_folders)):
            cat_name = class_folders[i].split('-')[-1].replace('_', ' ')  # 从目录名解析类别名（约定：用 '-' 分割并替换 '_' 为空格）
            img_path = os.path.join(img_root, class_folders[i])  # 顶层类别目录路径
            file_names = os.listdir(img_path)  # 获取该类别下的所有文件名
            for name in file_names:
                path = os.path.join(img_path,name)  # 组成单张图片的路径
                if cat_name not in test_samples:
                    test_samples[cat_name] = []  # 首次出现该类别时初始化列表
                test_samples[cat_name].append(path)  # 加入该类别的测试图片

        print(f'test sample:{test_samples}')  # 打印测试样本映射，便于调试核对
        print(f"[fast and slow] 测试数据集包含 {len(test_samples)} 个类别")  # 打印类别总数
        
        # 使用完整的快慢思考系统进行评估（统计多项指标）
        correct = 0  # 预测正确的样本数
        total = 0    # 评估的样本总数
        fast_only_correct = 0    # 未触发慢思考且预测正确的样本数
        slow_triggered = 0       # 触发过慢思考的样本数
        slow_triggered_correct = 0  # 触发慢思考且最终预测正确的样本数
        
        # 逐类别遍历，显示进度条
        from tqdm import tqdm  # 引入进度条库
        for true_cat, paths in tqdm(test_samples.items(), desc="Processing fast and slow thinking"):  # true_cat 为真值类别名
            for path in paths:  # 遍历该类别下的每一张图片
                # 使用完整的快慢思考系统进行单张图片分类（自动判断是否进入慢思考）
                result = system.classify_single_image(path, use_slow_thinking=None, top_k=5)
                
                pred = result.get('final_prediction', 'unknown')  # 取得最终类别预测
                ok = is_similar(pred, true_cat, threshold=0.5)    # 与真值进行相似匹配（大小写/空格等鲁棒）
                used_slow = result.get('used_slow_thinking', False)  # 记录是否触发慢思考
                
                if ok:
                    # 预测正确：打印详情并累加计数
                    print(f"succ. pred cate:{pred}, true cate:{true_cat}, used_slow:{used_slow}, confidence:{result.get('final_confidence', 0):.4f}")
                    correct += 1  # 总正确数 +1
                    if not used_slow:
                        fast_only_correct += 1  # 仅快思考就正确的数量 +1
                    if used_slow:
                        slow_triggered_correct += 1  # 触发慢思考且正确的数量 +1
                else:
                    # 预测失败：打印详情
                    print(f"failed. pred cate:{pred}, true cate:{true_cat}, used_slow:{used_slow}, confidence:{result.get('final_confidence', 0):.4f}")
                    # 如果需要也可以在此统计“触发慢思考但失败”的数量
                    # if used_slow:
                    #     slow_triggered_correct += 1  # 即使错误也统计（此处保持关闭）
                
                if used_slow:
                    slow_triggered += 1  # 样本进入过慢思考，累加触发数
                
                total += 1  # 样本总数 +1（不论成功与否）
        
        # 汇总评估指标
        acc = correct / total if total > 0 else 0.0  # 总体准确率
        fast_only_acc = fast_only_correct / (total-slow_triggered) if total > 0 else 0.0  # 仅快思考部分的准确率（注意：若分母为0会报错，此处保持原逻辑）
        slow_trigger_ratio = slow_triggered / total if total > 0 else 0.0  # 慢思考触发比例
        slow_trigger_acc = slow_triggered_correct / slow_triggered if slow_triggered > 0 else 0.0  # 触发慢思考样本的准确率
        
        # 打印评估结果
        print(f"✅ 正确预测总数: {correct}")
        print(f"  - 其中仅快思考正确: {fast_only_correct}")
        print(f"  - 其中慢思考触发且正确: {slow_triggered_correct}")
        print(f"❌ 错误预测总数: {total - correct}")
        print(f"📊 慢思考触发数量: {slow_triggered}")
        print(f"[fast and slow] 总体准确率: {acc:.4f} ({correct}/{total})")
        print(f"[fast and slow] 快思考准确率: {fast_only_acc:.4f}")
        print(f"[fast and slow] 慢思考触发比例: {slow_trigger_ratio:.4f}")
        print(f"[fast and slow] 慢思考准确率: {slow_trigger_acc:.4f}")
    
    elif args.mode == 'build_gallery':
        """
        构建多模态类别模板库
        """
        print("进入build_gallery模式")
        try:
            from agents.mllm_bot import MLLMBot
            from cvd.cdv_captioner import CDVCaptioner
            from retrieval.multimodal_retrieval import MultimodalRetrieval
            print("成功导入所需模块")
        except Exception as e:
            print(f"导入模块失败: {e}")
            raise
        
        # 使用MLLM单例，避免重复加载（显存优化）
        print("获取MLLM Bot实例（单例模式）...")
        from utils.mllm_singleton import get_mllm_bot
        mllm_bot = get_mllm_bot(
            model_tag=cfg['model_size_mllm'],
            device='cuda' if cfg['host'] in ["xiao"] else 'cpu'
        )
        print("MLLM Bot获取完成")
        
        print("初始化CDV Captioner...")
        captioner = CDVCaptioner()
        print("CDV Captioner初始化完成")
        
        print("初始化多模态检索模块...")
        retrieval = MultimodalRetrieval(
            fusion_method=args.fusion_method,
            device='cuda' if cfg['host'] in ["xiao"] else 'cpu'
        )
        print("多模态检索模块初始化完成")
        
        # 加载发现数据集
        print("加载发现数据集...")
        data_discovery = DATA_DISCOVERY[cfg['dataset_name']](cfg, folder_suffix=expt_id_suffix)
        print(f"发现数据集加载完成，包含 {len(data_discovery.samples)} 个样本")
        
        # 构建gallery
        print("开始构建gallery...")
        gallery = build_gallery(
            cfg, mllm_bot, captioner, retrieval,
            kshot=args.kshot,
            region_num=args.region_num,
            superclass=args.superclass,
            data_discovery=data_discovery
        )
        print("Gallery构建完成")
        
        # 保存gallery
        if args.gallery_out:
            import json
            import os
            os.makedirs(os.path.dirname(args.gallery_out), exist_ok=True)
            # 将numpy数组转换为列表以便JSON序列化
            gallery_serializable = {}
            for cat, feat in gallery.items():
                gallery_serializable[cat] = feat.tolist()
            
            with open(args.gallery_out, 'w') as f:
                json.dump(gallery_serializable, f, indent=2)
            print(f"Gallery saved to: {args.gallery_out}")
        
        print(f"Gallery built with {len(gallery)} categories")
    
    elif args.mode == 'fast_slow_infer':
        """
        快慢思考推理模式：保存快思考和慢思考的推理结果，不进行最终分类
        CUDA_VISIBLE_DEVICES=0 python discovering.py --mode=fast_slow_infer --config_file_env=./configs/env_machine.yml --config_file_expt=./configs/expts/dog120_all.yml --test_data_dir=./datasets/dogs_120/images_discovery_all_1 --knowledge_base_dir=./experiments/dog120/knowledge_base --infer_dir=./experiments/dog120/infer
        """
        if args.test_data_dir is None:
            raise ValueError("请提供测试数据目录 --test_data_dir")
        
        # 自动生成推理结果保存目录（基于数据集名称）
        if args.infer_dir is None:
            dataset_name = cfg['dataset_name']
            dataset_num = len(DATA_STATS[dataset_name]['class_names'])
            args.infer_dir = f"./experiments/{dataset_name}{dataset_num}/infer"
        
        print(f"推理结果将保存到: {args.infer_dir}")
        os.makedirs(args.infer_dir, exist_ok=True)
        
        # 初始化完整的快慢思考系统
        system = FastSlowThinkingSystem(
            model_tag=cfg['model_size_mllm'],
            model_name=cfg['model_size_mllm'],
            device='cuda' if cfg['host'] in ["xiao"] else 'cpu',
            cfg=cfg,
            enable_mllm_intermediate_judge=args.enable_mllm_intermediate_judge
        )
        # 加载知识库
        system.load_knowledge_base(args.knowledge_base_dir)

        # 构建测试样本
        test_samples = {}
        img_root = args.test_data_dir
        class_folders = os.listdir(args.test_data_dir)
        for i in range(len(class_folders)):
            cat_name = class_folders[i].split('-')[-1].replace('_', ' ')
            img_path = os.path.join(img_root, class_folders[i])
            file_names = os.listdir(img_path)
            for name in file_names:
                path = os.path.join(img_path,name)
                if cat_name not in test_samples:
                    test_samples[cat_name] = []
                test_samples[cat_name].append(path)

        print(f"[fast_slow_infer] 测试数据集包含 {len(test_samples)} 个类别")
        
        # 执行推理并保存结果
        total_processed = 0
        from tqdm import tqdm
        for true_cat, paths in tqdm(test_samples.items(), desc="Processing inference"):
            for path in paths:
                try:
                    # 执行快思考
                    fast_result = system.fast_thinking.fast_thinking_pipeline(path, top_k=5)
                    
                    # 判断是否需要慢思考
                    need_slow_thinking = fast_result["need_slow_thinking"]
                    
                    inference_data = {
                        "query_image": path,
                        "true_category": true_cat,
                        "fast_result": fast_result,
                        "need_slow_thinking": need_slow_thinking,
                        "slow_result": None,
                        # 保存分类前必须的所有信息
                        "fast_top_k": fast_result.get("img_results", [])[:5] + fast_result.get("text_results", [])[:5],  # 快思考Top-K候选
                        "fast_fused_results": fast_result.get("fused_results", [])[:5],  # 融合后的Top-K
                        "timestamp": time.time()
                    }
                    
                    # 如果需要慢思考，执行慢思考
                    if need_slow_thinking:
                        slow_result = system.slow_thinking.slow_thinking_pipeline_update(path, fast_result, top_k=5)
                        inference_data["slow_result"] = slow_result
                        # 保存慢思考的Top-K候选信息
                        inference_data["slow_top_k"] = slow_result.get("enhanced_results", [])[:5] if slow_result else []
                    
                    # 保存推理结果
                    base_name = os.path.splitext(os.path.basename(path))[0]
                    safe_cat_name = true_cat.replace(' ', '_').replace('/', '_')
                    infer_file = os.path.join(args.infer_dir, f"{safe_cat_name}_{base_name}.json")
                    
                    dump_json(infer_file, inference_data)
                    total_processed += 1
                    
                    if total_processed % 100 == 0:
                        print(f"已处理 {total_processed} 个样本")
                        
                except Exception as e:
                    print(f"处理失败 {path}: {e}")
                    continue
        
        print(f"推理完成！共处理 {total_processed} 个样本")
        print(f"推理结果已保存到: {args.infer_dir}")
    
    elif args.mode == 'fast_slow_classify':
        """
        快慢思考分类模式：加载推理结果，执行分类逻辑并统计指标
        CUDA_VISIBLE_DEVICES=0 python discovering.py --mode=fast_slow_classify --config_file_env=./configs/env_machine.yml --config_file_expt=./configs/expts/dog120_all.yml --infer_dir=./experiments/dog120/infer --classify_dir=./experiments/dog120/classify
        """
        # 自动生成推理结果加载目录和分类结果保存目录
        if args.infer_dir is None:
            dataset_name = cfg['dataset_name']
            dataset_num = len(DATA_STATS[dataset_name]['class_names'])
            args.infer_dir = f"./experiments/{dataset_name}{dataset_num}/infer"
        
        if args.classify_dir is None:
            dataset_name = cfg['dataset_name']
            dataset_num = len(DATA_STATS[dataset_name]['class_names'])
            args.classify_dir = f"./experiments/{dataset_name}{dataset_num}/classify"
        
        if not os.path.exists(args.infer_dir):
            raise ValueError(f"推理结果目录不存在: {args.infer_dir}")
        
        print(f"从目录加载推理结果: {args.infer_dir}")
        print(f"分类结果将保存到: {args.classify_dir}")
        os.makedirs(args.classify_dir, exist_ok=True)
        
        # 初始化系统（用于最终决策，如果需要的话）
        system = FastSlowThinkingSystem(
            model_tag=cfg['model_size_mllm'],
            model_name=cfg['model_size_mllm'],
            device='cuda' if cfg['host'] in ["xiao"] else 'cpu',
            cfg=cfg,
            enable_mllm_intermediate_judge=args.enable_mllm_intermediate_judge
        )
        
        # 加载所有推理结果文件
        infer_files = [f for f in os.listdir(args.infer_dir) if f.endswith('.json')]
        print(f"找到 {len(infer_files)} 个推理结果文件")
        
        # 统计指标
        correct = 0
        total = 0
        fast_only_correct = 0
        slow_triggered = 0
        slow_triggered_correct = 0
        
        classification_results = []
        
        from tqdm import tqdm
        for infer_file in tqdm(infer_files, desc="Processing classification"):
            try:
                infer_path = os.path.join(args.infer_dir, infer_file)
                inference_data = load_json(infer_path)
                
                query_image = inference_data["query_image"]
                true_cat = inference_data["true_category"]
                fast_result = inference_data["fast_result"]
                need_slow_thinking = inference_data["need_slow_thinking"]
                slow_result = inference_data.get("slow_result")
                
                # 执行分类逻辑（完整的三种路径）
                if not need_slow_thinking:
                    # 路径1: 仅快思考分类
                    final_prediction = fast_result["predicted_category"]
                    final_confidence = fast_result["confidence"]
                    used_slow_thinking = False
                    fast_slow_consistent = True
                    decision_path = "fast_only"
                else:
                    # 使用慢思考结果
                    if slow_result is None:
                        print(f"警告: {infer_file} 需要慢思考但没有慢思考结果")
                        continue
                    
                    fast_pred = fast_result.get("predicted_category", "unknown")
                    slow_pred = slow_result["predicted_category"]
                    used_slow_thinking = True
                    
                    # 检查快慢思考是否一致
                    if fast_pred != slow_pred and not is_similar(fast_pred, slow_pred, threshold=0.5):
                        # 路径3: 快慢不一致，需要最终裁决
                        fast_slow_consistent = False
                        decision_path = "final_arbitration"
                        
                        # 这里可以实现不同的裁决策略：
                        # 策略1: 直接用慢思考结果（当前默认）
                        final_prediction = slow_pred
                        final_confidence = slow_result["confidence"]
                        
                        # 策略2: 可选MLLM最终裁决（需要时可启用）
                        # if system and hasattr(system, 'final_arbitration'):
                        #     final_prediction, final_confidence = system.final_arbitration(
                        #         query_image, fast_result, slow_result
                        #     )
                        
                        print(f"快慢不一致: fast={fast_pred}, slow={slow_pred}, 裁决结果={final_prediction}")
                    else:
                        # 路径2: 快慢思考一致，直接用慢思考结果
                        final_prediction = slow_pred
                        final_confidence = slow_result["confidence"]
                        fast_slow_consistent = True
                        decision_path = "slow_consistent"
                
                # 评估预测结果
                is_correct = is_similar(final_prediction, true_cat, threshold=0.5)
                
                if is_correct:
                    correct += 1
                    if not used_slow_thinking:
                        fast_only_correct += 1
                    if used_slow_thinking:
                        slow_triggered_correct += 1
                        
                if used_slow_thinking:
                    slow_triggered += 1
                
                total += 1
                
                # 保存分类结果
                result = {
                    "query_image": query_image,
                    "true_category": true_cat,
                    "final_prediction": final_prediction,
                    "final_confidence": final_confidence,
                    "used_slow_thinking": used_slow_thinking,
                    "fast_slow_consistent": fast_slow_consistent,
                    "decision_path": decision_path,  # 记录决策路径
                    "is_correct": is_correct,
                    "fast_prediction": fast_result.get("predicted_category", "unknown"),
                    "fast_confidence": fast_result.get("confidence", 0.0),
                    "slow_prediction": slow_result["predicted_category"] if slow_result else None,
                    "slow_confidence": slow_result["confidence"] if slow_result else None
                }
                
                classification_results.append(result)
                
            except Exception as e:
                print(f"处理分类失败 {infer_file}: {e}")
                continue
        
        # 计算并打印指标
        acc = correct / total if total > 0 else 0.0
        fast_only_acc = fast_only_correct / (total-slow_triggered) if (total-slow_triggered) > 0 else 0.0
        slow_trigger_ratio = slow_triggered / total if total > 0 else 0.0
        slow_trigger_acc = slow_triggered_correct / slow_triggered if slow_triggered > 0 else 0.0
        
        print(f"✅ 正确预测总数: {correct}")
        print(f"  - 其中仅快思考正确: {fast_only_correct}")
        print(f"  - 其中慢思考触发且正确: {slow_triggered_correct}")
        print(f"❌ 错误预测总数: {total - correct}")
        print(f"📊 慢思考触发数量: {slow_triggered}")
        print(f"[fast_slow_classify] 总体准确率: {acc:.4f} ({correct}/{total})")
        print(f"[fast_slow_classify] 快思考准确率: {fast_only_acc:.4f}")
        print(f"[fast_slow_classify] 慢思考触发比例: {slow_trigger_ratio:.4f}")
        print(f"[fast_slow_classify] 慢思考准确率: {slow_trigger_acc:.4f}")
        
        # 保存分类结果
        results_file = os.path.join(args.classify_dir, "classification_results.json")
        dump_json(results_file, {
            "summary": {
                "total_samples": total,
                "correct_predictions": correct,
                "accuracy": acc,
                "fast_only_correct": fast_only_correct,
                "fast_only_accuracy": fast_only_acc,
                "slow_triggered": slow_triggered,
                "slow_trigger_ratio": slow_trigger_ratio,
                "slow_triggered_correct": slow_triggered_correct,
                "slow_trigger_accuracy": slow_trigger_acc
            },
            "detailed_results": classification_results
        })
        
        print(f"分类结果已保存到: {results_file}")
    
    else:
        raise NotImplementedError 

