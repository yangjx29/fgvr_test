import os
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from data.data_stats import IMAGENET_SKETCH_STATS
import pathlib
import random
import shutil
from copy import deepcopy
from data.utils import get_swav_transform


SUPERCLASS = 'object'
CLASSUNIT = 'categories'


imagenet_sketch_how_to1 = f"""
Your task is to tell me what are the useful attributes for distinguishing {SUPERCLASS} {CLASSUNIT} in a sketch drawing of an {SUPERCLASS}.

Specifically, you can complete the task by following the instructions below:
1 - I give you an example delimited by <> about what are the useful attributes for distinguishing bird species in 
a photo of a bird. You should understand and learn this example carefully.
2 - List the useful attributes for distinguishing {SUPERCLASS} {CLASSUNIT} in a sketch drawing of an {SUPERCLASS}.
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
The useful attributes for distinguishing {SUPERCLASS} {CLASSUNIT} in a sketch drawing of an {SUPERCLASS}:
===
"""


imagenet_sketch_how_to2 = f"""
Please tell me what are the useful attributes for distinguishing {SUPERCLASS} {CLASSUNIT} in a sketch drawing of an {SUPERCLASS} according to the 
example of about what are the useful attributes for distinguishing bird species in a photo of a bird. Output a Python 
list object that contains the listed useful attributes.

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
The useful attributes for distinguishing {SUPERCLASS} {CLASSUNIT} in a sketch drawing of an {SUPERCLASS}:
===
"""


class ImageNetSketchPrompter:
    def __init__(self):
        self.supercategory = "object"
        self.first_question = "general"
        self.attributes = [
            'shape', 'size', 'outline', 'structure', 'pattern', 
            'form', 'contour', 'lines', 'strokes', 'details',
            'orientation', 'position', 'context', 'style', 'edges',
            'proportions', 'silhouette', 'texture', 'composition', 'features'
        ]
    
    def get_attributes(self):
        list_attributes = ['General Description']
        list_attributes.extend(self.attributes)
        return list_attributes
    
    def get_attribute_prompt(self):
        list_prompts = ["Look at this sketch carefully and describe what you see in general terms."]
        for attr in self.attributes:
            list_prompts.append(self._generate_statement_prompt(attr))
        return list_prompts
    
    def get_llm_prompt(self, list_attr_val):
        if len(list_attr_val) == 1:
            return f"""Your task is to tell me what are the useful attributes for distinguishing {self.supercategory} categories in a sketch drawing of an {self.supercategory}.

Specifically, you can complete the task by following the instructions below:
1 - I give you an example delimited by <> about what are the useful attributes for distinguishing bird species in a photo of a bird. You should understand and learn this example carefully.
2 - List the useful attributes for distinguishing {self.supercategory} categories in a sketch drawing of an {self.supercategory}.
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
<{self.supercategory} categories>
The useful attributes for distinguishing {self.supercategory} categories in a sketch drawing of an {self.supercategory}:
===
"""
        elif len(list_attr_val) == len(self.attributes) + 1:
            return f"""Please tell me what are the useful attributes for distinguishing {self.supercategory} categories in a sketch drawing of an {self.supercategory} according to the example of about what are the useful attributes for distinguishing bird species in a photo of a bird. Output a Python list object that contains the listed useful attributes.

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
<{self.supercategory} categories>
The useful attributes for distinguishing {self.supercategory} categories in a sketch drawing of an {self.supercategory}:
===
"""
        else:
            return ""
    
    def _generate_statement_prompt(self, attribute):
        return f"Look at this sketch carefully and describe the {attribute} of what you see."


class ImageNetSketchDiscovery:
    def __init__(self, cfg, folder_suffix=''):
        # 使用JSON文件加载数据
        json_path = os.path.join(cfg['expt_dir'], 'images_split', f'images_discovery_all{folder_suffix}.json')
        print(f"构建ImageNet-Sketch发现集,json_path: {json_path}")
        
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"JSON文件不存在: {json_path}")
        
        # 加载JSON文件
        import json
        with open(json_path, 'r') as f:
            self.data = json.load(f)
        
        self.samples = []
        self.targets = []
        self.subcategories = []
        
        # 解析JSON数据 - 格式: [class_name, class_id, image_list]
        for class_data in self.data:
            class_name = class_data[0]
            class_id = class_data[1]
            image_list = class_data[2]
            
            # 为每个图片创建样本
            for img_file in image_list:
                # 检查img_file是否已经是完整路径
                if img_file.startswith('./') or img_file.startswith('/') or os.path.isabs(img_file):
                    # 已经是完整路径，直接使用
                    img_path = img_file
                elif img_file.startswith('datasets/'):
                    # 是相对datasets目录的路径，直接使用
                    img_path = f"./{img_file}"
                else:
                    # 是其他相对路径，需要拼接基础路径
                    img_path = os.path.join(cfg['data_dir'], img_file)
                self.samples.append(img_path)
                self.subcategories.append(class_name)
                self.targets.append(class_id)
        
        # 创建子类别到样本的映射
        from collections import defaultdict
        self.subcat_to_sample = defaultdict(list)
        for subcat, sample in zip(self.subcategories, self.samples):
            self.subcat_to_sample[subcat].append(sample)
    
    def __getitem__(self, index):
        img_path = self.samples[index]
        subcategory = self.subcategories[index]
        img = Image.open(img_path).convert('RGB')
        return img, subcategory, img_path
    
    def __len__(self):
        return len(self.samples)


class ImageNetSketchDataset(Dataset):
    def __init__(self, root, train=True, transform=None):
        self.root = root
        self.train = train
        self.transform = transform
        
        if train:
            img_root = os.path.join(root, 'images_train')
        else:
            img_root = os.path.join(root, 'images_test')
        
        self.samples = []
        self.classes = []
        
        if os.path.exists(img_root):
            class_folders = [d for d in os.listdir(img_root) 
                           if os.path.isdir(os.path.join(img_root, d))]
            class_folders.sort()
            
            for class_id in class_folders:
                class_path = os.path.join(img_root, class_id)
                img_files = os.listdir(class_path)
                img_files = [f for f in img_files if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                for img_file in img_files:
                    img_path = os.path.join(class_path, img_file)
                    self.samples.append((img_path, class_folders.index(class_id)))
                self.classes.append(class_id)
    
    def __getitem__(self, index):
        img_path, target = self.samples[index]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, target, img_path
    
    def __len__(self):
        return len(self.samples)


def build_imagenet_sketch_prompter(cfg: dict):
    return ImageNetSketchPrompter()


def build_imagenet_sketch_discovery(cfg: dict, folder_suffix=''):
    return ImageNetSketchDiscovery(cfg, folder_suffix=folder_suffix)


def build_imagenet_sketch_test(cfg):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    
    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])
    
    test_dataset = ImageNetSketchDataset(cfg['data_dir'], train=False, transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=cfg['batch_size'], 
                            shuffle=False, num_workers=cfg['num_workers'])
    return test_loader


def build_imagenet_sketch_swav_train(cfg):
    train_transform = get_swav_transform()
    
    train_dataset = ImageNetSketchDataset(cfg['data_dir'], train=True, transform=train_transform)
    train_loader = DataLoader(train_dataset, batch_size=cfg['batch_size'],
                             shuffle=True, num_workers=cfg['num_workers'], drop_last=True)
    return train_loader


def _transform(cfg):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])
    
    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])
    
    return train_transform, test_transform
