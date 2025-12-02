from PIL import Image
import torchvision.transforms as transforms
import os
import random
import shutil
from copy import deepcopy
import numpy as np
from torchvision.datasets.folder import default_loader
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from data.data_stats import CALTECH256_STATS
import pathlib
from data.utils import get_swav_transform


SUPERCLASS = 'object'
CLASSUNIT = 'categories'


caltech256_how_to1 = f"""
Your task is to tell me what are the useful attributes for distinguishing specific {SUPERCLASS} {CLASSUNIT} \
in a photo of an {SUPERCLASS}. \

Specifically, you can complete the task by following the instructions below: \
1 - I give you an example delimited by <> about what are the useful attributes for distinguishing bird species in \
a photo of a bird. You should understand and learn this example carefully. \
2 - List the useful attributes for distinguishing specific {SUPERCLASS} {CLASSUNIT} in a photo of an {SUPERCLASS}. \
3 - Output a Python list object that contains the listed useful attributes. \

=== \
<bird species> \
The useful attributes for distinguishing bird species in a photo of a bird: \
['bill shape', 'wing color', 'upperparts color', 'underparts color', 'breast pattern', \
'back color', 'tail shape', 'upper tail color', 'head pattern', 'breast color', \
'throat color', 'eye color', 'bill length', 'forehead color', 'under tail color', \
'nape color', 'belly color', 'wing shape', 'size', 'shape', \
'back pattern', 'tail pattern', 'belly pattern', 'primary color', 'leg color', \
'bill color', 'crown color', 'wing pattern', 'habitat'] \
=== \

=== \
<{SUPERCLASS} {CLASSUNIT}> \
The useful attributes for distinguishing specific {SUPERCLASS} {CLASSUNIT} in a photo of an {SUPERCLASS}: \
=== \
"""


caltech256_how_to2 = f"""
Please tell me what are the useful visual attributes for distinguishing {SUPERCLASS} {CLASSUNIT} from its appearance \
in a photo, like the example of about what are the useful visual attributes for distinguishing bird species in a \
photo I give you later. List the useful visual attributes for distinguishing {SUPERCLASS} {CLASSUNIT} in a photo and \
output a Python list object that contains the listed useful visual attributes. \

=== \
Question: What are the useful visual attributes for distinguishing bird species in a photo of a bird? \
=== \
Answer: ['bill shape', 'wing color', 'upperparts color', 'underparts color', 'breast pattern', \
'back color', 'tail shape', 'upper tail color', 'head pattern', 'breast color', \
'throat color', 'eye color', 'bill length', 'forehead color', 'under tail color', \
'nape color', 'belly color', 'wing shape', 'size', 'shape', \
'back pattern', 'tail pattern', 'belly pattern', 'primary color', 'leg color', \
'bill color', 'crown color', 'wing pattern', 'habitat'] \
=== \
Question: What are the useful visual attributes for distinguishing {SUPERCLASS} {CLASSUNIT} in a photo of an {SUPERCLASS}? \
=== \
Answer: \
"""


def _transform(n_px):
    return transforms.Compose([
        transforms.Resize(n_px, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(n_px),
        lambda image: image.convert("RGB"),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),   # ImageNet
    ])


class Caltech256Prompter:
    def __init__(self):
        self.supercategory = "object"
        self.superunit = 'category'

        self.first_question = "general"
        self.attributes = [
            'shape', 'size', 'color', 'texture', 'material', 'pattern', 'orientation', 
            'parts configuration', 'structural features', 'surface details', 
            'overall appearance', 'distinctive markings', 'proportions', 'edges and contours',
            'spatial arrangement', 'visual complexity', 'key distinguishing features',
            'functional characteristics', 'contextual elements'
        ]

    def _generate_howmany_prompt(self, attr):
        return f"Questions: How many {attr} does the {self.supercategory} in this photo have? Answer:"

    def _generate_whatis_prompt(self, attr):
        return f"Questions: What is the {attr} of the {self.supercategory} in this photo? Answer:"

    def _generate_statement_prompt(self, attr):
        return f"Describe the {attr} of the {self.supercategory} in this photo."

    def get_attributes(self):
        list_attributes = ['General Description']
        list_attributes.extend(self.attributes)
        return list_attributes

    def get_attribute_prompt(self):
        # a series of questions
        list_prompts = ["Describe this image in details."]

        for attr in self.attributes:
            list_prompts.append(self._generate_statement_prompt(attr))

        return list_prompts

    def get_llm_prompt(self, attr_descr_pairs):
        prompt = f"""
        I have a photo of an {self.supercategory}. 
        Your task is to perform the following actions:
        1 - Summarize the information you get about the {self.supercategory} from the general description and \
        attribute descriptions delimited by triple backticks with five sentences.
        2 - The description might not be correct and accurate. So I need you to infer and list three possible detailed object category names (e.g., Accordion, Airplanes, Anchor) of the \
        {self.supercategory} in this photo based on the information you get.
        3 - Output a JSON object that uses the following format
        <three possible detailed object category names>: [
                <first sentence of the summary>,
                <second sentence of the summary>,
                <third sentence of the summary>,
                <fourth sentence of the summary>,
                <fifth sentence of the summary>,
        ]

        Use the following format to perform the aforementioned tasks:
        General Description: '''general description of the photo'''
        Attributes List:
        - '''attribute name''': '''attribute description'''
        - '''attribute name''': '''attribute description'''
        - ...
        - '''attribute name''': '''attribute description'''
        Summary: <summary>
        Three possible detailed object category names: <three possible detailed object category names>
        Output JSON: <output JSON object>

        '''{attr_descr_pairs[0][0]}''': '''{attr_descr_pairs[0][1]}'''
        Attributes List:
        - '''{attr_descr_pairs[1][0]}''': '''{attr_descr_pairs[1][1]}'''
        - '''{attr_descr_pairs[2][0]}''': '''{attr_descr_pairs[2][1]}'''
        - '''{attr_descr_pairs[3][0]}''': '''{attr_descr_pairs[3][1]}'''
        - '''{attr_descr_pairs[4][0]}''': '''{attr_descr_pairs[4][1]}'''
        - '''{attr_descr_pairs[5][0]}''': '''{attr_descr_pairs[5][1]}'''
        - '''{attr_descr_pairs[6][0]}''': '''{attr_descr_pairs[6][1]}'''
        - '''{attr_descr_pairs[7][0]}''': '''{attr_descr_pairs[7][1]}'''
        - '''{attr_descr_pairs[8][0]}''': '''{attr_descr_pairs[8][1]}'''
        - '''{attr_descr_pairs[9][0]}''': '''{attr_descr_pairs[9][1]}'''
        - '''{attr_descr_pairs[10][0]}''': '''{attr_descr_pairs[10][1]}'''
        - '''{attr_descr_pairs[11][0]}''': '''{attr_descr_pairs[11][1]}'''
        - '''{attr_descr_pairs[12][0]}''': '''{attr_descr_pairs[12][1]}'''
        - '''{attr_descr_pairs[13][0]}''': '''{attr_descr_pairs[13][1]}'''
        - '''{attr_descr_pairs[14][0]}''': '''{attr_descr_pairs[14][1]}'''
        - '''{attr_descr_pairs[15][0]}''': '''{attr_descr_pairs[15][1]}'''
        - '''{attr_descr_pairs[16][0]}''': '''{attr_descr_pairs[16][1]}'''
        - '''{attr_descr_pairs[17][0]}''': '''{attr_descr_pairs[17][1]}'''
        - '''{attr_descr_pairs[18][0]}''': '''{attr_descr_pairs[18][1]}'''
        """
        return prompt


class Caltech256Discovery:
    """Caltech256数据集的发现集加载器（基于JSON文件）"""
    
    def __init__(self, cfg, folder_suffix=''):
        # 使用配置中的实验目录路径加载JSON文件
        json_path = os.path.join(cfg['expt_dir'], 'images_split', f'images_discovery_all{folder_suffix}.json')
        print(f"构建发现集,json_path: {json_path}")
        
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"JSON文件不存在: {json_path}")
        
        # 加载JSON文件
        import json
        with open(json_path, 'r') as f:
            self.data = json.load(f)
        
        self.classes = CALTECH256_STATS['class_names']
        self.samples = []
        self.targets = []
        self.subcategories = []
        
        # 解析JSON数据 - 格式: [class_name, class_id, image_list]
        for class_data in self.data:
            class_name = class_data[0]
            class_id = class_data[1]
            image_list = class_data[2]
            
            for img_path in image_list:
                self.samples.append(img_path)
                self.subcategories.append(class_name)
                self.targets.append(class_id)
        
        # 创建子类别到样本的映射
        from collections import defaultdict
        self.subcat_to_sample = defaultdict(list)
        for subcat, sample in zip(self.subcategories, self.samples):
            self.subcat_to_sample[subcat].append(sample)
        
        print(f'从JSON加载 {len(self.data)} 个类别，共 {len(self.samples)} 张图像')
        self.index = 0

    def __len__(self):
        return len(self.samples)

    def __iter__(self):
        return self

    def __next__(self):
        if self.index >= len(self.samples):
            raise StopIteration
        img = Image.open(self.samples[self.index]).convert("RGB")
        target = self.targets[self.index]
        self.index += 1
        return img, target


class Caltech256Dataset(Dataset):
    """Caltech-256 Dataset"""
    def __init__(self, root, train=True, transform=None, limit=0):
        if train:
            data_dir = "256_ObjectCategories/"
        else:
            data_dir = "256_ObjectCategories/"

        data_dir = os.path.join(root, data_dir)

        self.loader = default_loader
        self.data_dir = data_dir
        self.data = []
        self.target = []
        self.train = train
        self.transform = transform

        self._load_dataset(limit)

        self.uq_idxs = np.array(range(len(self)))
        self.target_transform = None

        self.classes = CALTECH256_STATS['class_names']

    def _load_dataset(self, limit=0):
        """加载数据集"""
        if not os.path.exists(self.data_dir):
            raise FileNotFoundError(f"数据集目录不存在: {self.data_dir}")
        
        # 遍历所有类别目录
        for class_idx, class_name in enumerate(self.classes):
            if limit and class_idx >= limit:
                break
            
            # 查找对应的文件夹（可能有编号前缀）
            class_folder = None
            for folder_name in os.listdir(self.data_dir):
                if os.path.isdir(os.path.join(self.data_dir, folder_name)):
                    # 移除编号前缀
                    folder_name_clean = folder_name
                    if '.' in folder_name:
                        folder_name_clean = folder_name.split('.', 1)[1]
                    
                    # 标准化比较
                    folder_normalized = folder_name_clean.replace('-', ' ').title()
                    class_normalized = class_name.replace(' ', '').lower()
                    folder_normalized_clean = folder_normalized.replace(' ', '').lower()
                    
                    if folder_normalized_clean == class_normalized or class_name.lower() in folder_normalized.lower():
                        class_folder = folder_name
                        break
            
            if class_folder is None:
                continue
            
            class_dir = os.path.join(self.data_dir, class_folder)
            if not os.path.isdir(class_dir):
                continue
            
            # 获取该类别的所有图像
            image_files = []
            for f in os.listdir(class_dir):
                if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                    image_files.append(os.path.join(class_dir, f))
            
            for img_path in image_files:
                self.data.append(img_path)
                self.target.append(class_idx)

    def __getitem__(self, idx):
        image = self.loader(self.data[idx])
        target = self.target[idx]

        if self.transform is not None:
            image = self.transform(image)

        if self.target_transform is not None:
            target = self.target_transform(target)

        idx = self.uq_idxs[idx]

        return image, target, self.data[idx]  # just for visualization

    def __len__(self):
        return len(self.data)


def build_caltech256_prompter(cfg: dict):
    prompter = Caltech256Prompter()
    return prompter


def build_caltech256_discovery(cfg: dict, folder_suffix=''):
    set_to_discover = Caltech256Discovery(cfg, folder_suffix=folder_suffix)
    return set_to_discover


def build_caltech256_test(cfg):
    data_path = pathlib.Path(cfg['data_dir'])
    tfms = _transform(cfg['image_size'])

    dataset = Caltech256Dataset(data_path, train=False, transform=tfms)

    dataloader = DataLoader(dataset, batch_size=cfg['batch_size'], shuffle=True, num_workers=cfg['num_workers'],
                            pin_memory=True)
    return dataloader


def build_caltech256_swav_train(cfg):
    data_path = pathlib.Path(cfg['data_dir'])
    tfms = get_swav_transform(cfg['image_size'])

    dataset = Caltech256Dataset(data_path, train=True, transform=tfms)

    dataloader = DataLoader(dataset, batch_size=cfg['batch_size'], shuffle=True, num_workers=cfg['num_workers'],
                            pin_memory=True)
    return dataloader

