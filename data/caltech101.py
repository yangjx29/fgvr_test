import os
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from data.data_stats import CALTECH101_STATS
import pathlib
import random
import shutil
from copy import deepcopy
from data.utils import get_swav_transform


SUPERCLASS = 'object'
CLASSUNIT = 'categories'


caltech101_how_to1 = f"""
Your task is to tell me what are the useful attributes for distinguishing {SUPERCLASS} {CLASSUNIT} in a photo of an {SUPERCLASS}.

Specifically, you can complete the task by following the instructions below:
1 - I give you an example delimited by <> about what are the useful attributes for distinguishing bird species in 
a photo of a bird. You should understand and learn this example carefully.
2 - List the useful attributes for distinguishing {SUPERCLASS} {CLASSUNIT} in a photo of an {SUPERCLASS}.
3 - Output a Python list object that contains the listed useful attributes.

===
<bird species>
The useful attributes for distinguishing bird species in a photo of a bird:
['bill shape', 'wing color', 'upperparts color', 'underparts color', 'breast pattern',
'back color', 'tail shape', 'upper tail color', 'head pattern', 'breast color',
'throat color', 'eye color', 'bill length', 'forehead color', 'under tail color',
'nape color', 'belly color', 'wing shape', 'size', 'shape',
'back pattern', 'tail pattern', 'belly pattern', 'primary color', 'leg color',
'bill color', 'crown color', 'wing pattern', 'habitat']
===

===
<{SUPERCLASS} {CLASSUNIT}>
The useful attributes for distinguishing {SUPERCLASS} {CLASSUNIT} in a photo of an {SUPERCLASS}:
===
"""


caltech101_how_to2 = f"""
Please tell me what are the useful attributes for distinguishing {SUPERCLASS} {CLASSUNIT} in a photo of an {SUPERCLASS} according to the 
example of about what are the useful attributes for distinguishing bird species in a photo of a bird. Output a Python 
list object that contains the listed useful attributes.

===
Question: What are the useful attributes for distinguishing bird species in a photo of a bird?
===
Answer: ['bill shape', 'wing color', 'upperparts color', 'underparts color', 'breast pattern',
'back color', 'tail shape', 'upper tail color', 'head pattern', 'breast color',
'throat color', 'eye color', 'bill length', 'forehead color', 'under tail color',
'nape color', 'belly color', 'wing shape', 'size', 'shape',
'back pattern', 'tail pattern', 'belly pattern', 'primary color', 'leg color',
'bill color', 'crown color', 'wing pattern', 'habitat']
===
Question: What are the useful attributes for distinguishing {SUPERCLASS} {CLASSUNIT} in a photo of an {SUPERCLASS}?
===
Answer:
"""


def _transform(n_px):
    return transforms.Compose([
        transforms.Resize(n_px, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(n_px),
        lambda image: image.convert("RGB"),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),   # ImageNet
    ])


class Caltech101Prompter:
    def __init__(self):
        self.supercategory = "object"
        self.first_question = "general"
        self.attributes = [
            'shape', 'size', 'color', 'texture', 'material', 'pattern', 'orientation', 
            'parts configuration', 'structural features', 'surface details', 
            'overall appearance', 'distinctive markings', 'proportions', 'edges and contours',
            'spatial arrangement', 'visual complexity', 'key distinguishing features'
        ]

    def _generate_question_prompt(self, attr):
        return f"Questions: What is the {attr} of the {self.supercategory} in this photo. Answer:"

    def _generate_statement_prompt(self, attr):
        return f" What is the {attr} of the object:"

    def get_attributes(self):
        list_attributes = ['General Description']
        list_attributes.extend(self.attributes)
        return list_attributes

    def get_attribute_prompt(self):
        list_prompts = ["Look at this photo carefully. Describe what you see in detail, including the object's appearance, features, and any notable characteristics. Be specific and descriptive."]
        for attr in self.attributes:
            list_prompts.append(self._generate_statement_prompt(attr))
        return list_prompts

    def get_llm_prompt(self, list_attr_val):
        prompt = f"""
        I have a photo of an {self.supercategory}. 
        Your task is to perform the following actions:
        1 - Summarize the information you get about the {self.supercategory} from the general description and 
        attribute descriptions delimited by triple backticks with five sentences.
        2 - Infer and list three possible category names of the {self.supercategory} in this photo based on the 
        information you get.
        3 - Output a JSON object that uses the following format
        <three possible category names>: [
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
        Three possible object category names: <three possible object category names>
        Output JSON: <output JSON object>

        '''{list_attr_val[0][0]}''': '''{list_attr_val[0][1]}'''
        Attributes List:
        - '''{list_attr_val[1][0]}''': '''{list_attr_val[1][1]}'''
        - '''{list_attr_val[2][0]}''': '''{list_attr_val[2][1]}'''
        - '''{list_attr_val[3][0]}''': '''{list_attr_val[3][1]}'''
        - '''{list_attr_val[4][0]}''': '''{list_attr_val[4][1]}'''
        - '''{list_attr_val[5][0]}''': '''{list_attr_val[5][1]}'''
        - '''{list_attr_val[6][0]}''': '''{list_attr_val[6][1]}'''
        - '''{list_attr_val[7][0]}''': '''{list_attr_val[7][1]}'''
        - '''{list_attr_val[8][0]}''': '''{list_attr_val[8][1]}'''
        - '''{list_attr_val[9][0]}''': '''{list_attr_val[9][1]}'''
        - '''{list_attr_val[10][0]}''': '''{list_attr_val[10][1]}'''
        - '''{list_attr_val[11][0]}''': '''{list_attr_val[11][1]}'''
        - '''{list_attr_val[12][0]}''': '''{list_attr_val[12][1]}'''
        - '''{list_attr_val[13][0]}''': '''{list_attr_val[13][1]}'''
        - '''{list_attr_val[14][0]}''': '''{list_attr_val[14][1]}'''
        - '''{list_attr_val[15][0]}''': '''{list_attr_val[15][1]}'''
        - '''{list_attr_val[16][0]}''': '''{list_attr_val[16][1]}'''
        """
        return prompt


class Caltech101Discovery:
    """Caltech101数据集的发现集加载器（基于JSON文件）"""
    
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
        
        self.classes = CALTECH101_STATS['class_names']
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


class Caltech101Dataset(Dataset):
    """Caltech-101 Dataset"""
    def __init__(self, root, train=True, transform=None):
        self.root = root
        self.train = train
        self.transform = transform
        self.classes = CALTECH101_STATS['class_names']
        
        # 加载图像路径和标签
        self.data = []
        self.target = []
        self._load_dataset()

    def _load_dataset(self):
        # Caltech-101的数据在101_ObjectCategories目录下
        categories_dir = os.path.join(self.root, '101_ObjectCategories')
        if not os.path.exists(categories_dir):
            raise FileNotFoundError(f"数据集目录不存在: {categories_dir}")
        
        # 遍历所有类别目录
        for class_idx, class_name in enumerate(self.classes):
            # 将类别名转换为文件夹名（下划线格式）
            folder_name = class_name.replace(' ', '_').lower()
            class_dir = os.path.join(categories_dir, folder_name)
            
            if not os.path.exists(class_dir):
                # 尝试其他可能的格式
                folder_name_alt = class_name.replace(' ', '_')
                class_dir = os.path.join(categories_dir, folder_name_alt)
                if not os.path.exists(class_dir):
                    # 尝试原始格式
                    class_dir = os.path.join(categories_dir, class_name)
                    if not os.path.exists(class_dir):
                        continue
            
            # 获取该类别的所有图像
            image_files = []
            for f in os.listdir(class_dir):
                if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                    image_files.append(os.path.join(class_dir, f))
            
            # 根据train/test划分（这里简化处理，实际应该根据split文件）
            for img_path in image_files:
                self.data.append(img_path)
                self.target.append(class_idx)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        image_path = self.data[index]
        target = self.target[index]
        image = Image.open(image_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, target, image_path


def build_caltech101_prompter(cfg: dict):
    prompter = Caltech101Prompter()
    return prompter


def build_caltech101_discovery(cfg: dict, folder_suffix=''):
    set_to_discover = Caltech101Discovery(cfg, folder_suffix=folder_suffix)
    return set_to_discover


def build_caltech101_test(cfg):
    data_path = pathlib.Path(cfg['data_dir'])
    tfms = _transform(cfg['image_size'])

    dataset = Caltech101Dataset(data_path, train=False, transform=tfms)

    dataloader = DataLoader(dataset, batch_size=cfg['batch_size'], shuffle=True, num_workers=cfg['num_workers'],
                            pin_memory=True)
    return dataloader


def build_caltech101_swav_train(cfg):
    data_path = pathlib.Path(cfg['data_dir'])
    tfms = get_swav_transform(cfg['image_size'])

    dataset = Caltech101Dataset(data_path, train=True, transform=tfms)

    dataloader = DataLoader(dataset, batch_size=cfg['batch_size'], shuffle=True, num_workers=cfg['num_workers'],
                            pin_memory=True)
    return dataloader

