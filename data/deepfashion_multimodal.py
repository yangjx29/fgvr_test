import os
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from data.data_stats import DEEPFASHION_MULTIMODAL_STATS
import pathlib
import random
import shutil
from copy import deepcopy
from data.utils import get_swav_transform


SUPERCLASS = 'fashion'
CLASSUNIT = 'categories'


deepfashion_multimodal_how_to1 = f"""
Your task is to tell me what are the useful attributes for distinguishing {SUPERCLASS} {CLASSUNIT} in a photo of {SUPERCLASS} items.

Specifically, you can complete the task by following the instructions below:
1 - I give you an example delimited by <> about what are the useful attributes for distinguishing bird species in 
a photo of a bird. You should understand and learn this example carefully.
2 - List the useful attributes for distinguishing {SUPERCLASS} {CLASSUNIT} in a photo of {SUPERCLASS} items.
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
The useful attributes for distinguishing {SUPERCLASS} {CLASSUNIT} in a photo of {SUPERCLASS} items:
===
"""


deepfashion_multimodal_how_to2 = f"""
Please tell me what are the useful attributes for distinguishing {SUPERCLASS} {CLASSUNIT} in a photo of {SUPERCLASS} items according to the 
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
Question: What are the useful attributes for distinguishing {SUPERCLASS} {CLASSUNIT} in a photo of {SUPERCLASS} items?
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


class DeepFashionMultimodalPrompter:
    def __init__(self):
        self.supercategory = "fashion"
        self.first_question = "general"
        self.attributes = [
            'garment type', 'sleeve length', 'collar style', 'neckline', 'fit style',
            'fabric texture', 'color scheme', 'pattern design', 'length', 'waist style',
            'closure type', 'pocket style', 'hemline', 'cuff style', 'overall silhouette',
            'decoration details', 'brand characteristics'
        ]

    def _generate_question_prompt(self, attr):
        return f"Questions: What is the {attr} of the {self.supercategory} item in this photo. Answer:"

    def _generate_statement_prompt(self, attr):
        return f" What is the {attr} of the fashion item:"

    def get_attributes(self):
        list_attributes = ['General Description']
        list_attributes.extend(self.attributes)
        return list_attributes

    def get_attribute_prompt(self):
        list_prompts = ["Look at this photo carefully. Describe what you see in detail, including the fashion item's appearance, style, features, and any notable characteristics. Be specific and descriptive."]
        for attr in self.attributes:
            list_prompts.append(self._generate_statement_prompt(attr))
        return list_prompts

    def get_llm_prompt(self, list_attr_val):
        prompt = f"""
        I have a photo of a {self.supercategory} item. 
        Your task is to perform the following actions:
        1 - Summarize the information you get about the {self.supercategory} item from the general description and 
        attribute descriptions delimited by triple backticks with five sentences.
        2 - Infer and list three possible category names of the {self.supercategory} item in this photo based on the 
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
        Three possible fashion category names: <three possible fashion category names>
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


class DeepFashionMultimodalDiscovery:
    def __init__(self, root, folder_suffix=''):
        img_root = os.path.join(root, f'images_discovery_all{folder_suffix}')
        print(f"构建发现集,img_root: {img_root}")
        self.class_folders = os.listdir(img_root)  # ["MEN-Denim", "WOMEN-Dresses", ...]
        for i in range(len(self.class_folders)):
            self.class_folders[i] = os.path.join(img_root, self.class_folders[i])

        self.classes = DEEPFASHION_MULTIMODAL_STATS['class_names']
        self.samples = []
        self.targets = []
        self.subcategories = []
        
        for folder in self.class_folders:
            folder_name = folder.split('/')[-1]
            # 从文件夹名称映射到类别索引
            class_name = folder_name  # DeepFashion Multimodal的文件夹名就是类别名
            
            # 查找对应的类别索引
            label = None
            for idx, cls_name in enumerate(self.classes):
                if cls_name == class_name:
                    label = idx
                    break
            
            if label is None:
                continue  # 跳过无法匹配的类别
            
            file_names = os.listdir(folder)
            for name in file_names:
                if name.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                    self.subcategories.append(class_name)
                    self.samples.append(os.path.join(folder, name))
                    self.targets.append(label)
        
        # 添加subcat_to_sample属性，按类别名分组图片路径
        from collections import defaultdict
        self.subcat_to_sample = defaultdict(list)
        for subcat, sample in zip(self.subcategories, self.samples):
            self.subcat_to_sample[subcat].append(sample)
        
        print(f'subcat_to_sample: {len(self.subcat_to_sample)} classes')
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


class DeepFashionMultimodalDataset(Dataset):
    """DeepFashion Multimodal Dataset"""
    def __init__(self, root, train=True, transform=None):
        self.root = root
        self.train = train
        self.transform = transform
        self.classes = DEEPFASHION_MULTIMODAL_STATS['class_names']
        
        # 加载图像路径和标签
        self.data = []
        self.target = []
        self._load_dataset()

    def _load_dataset(self):
        # DeepFashion Multimodal的数据在images目录下，按类别组织
        images_dir = os.path.join(self.root, 'images')
        if not os.path.exists(images_dir):
            # 如果没有images目录，尝试从images_discovery_all获取
            images_dir = os.path.join(self.root, 'images_discovery_all')
            if not os.path.exists(images_dir):
                raise FileNotFoundError(f"数据集目录不存在: {images_dir}")
        
        # 遍历所有类别目录
        for class_idx, class_name in enumerate(self.classes):
            class_dir = os.path.join(images_dir, class_name)
            
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


def build_deepfashion_multimodal_prompter(cfg: dict):
    prompter = DeepFashionMultimodalPrompter()
    return prompter


def build_deepfashion_multimodal_discovery(cfg: dict, folder_suffix=''):
    set_to_discover = DeepFashionMultimodalDiscovery(cfg['data_dir'], folder_suffix=folder_suffix)
    return set_to_discover


def build_deepfashion_multimodal_test(cfg):
    data_path = pathlib.Path(cfg['data_dir'])
    tfms = _transform(cfg['image_size'])

    dataset = DeepFashionMultimodalDataset(data_path, train=False, transform=tfms)

    dataloader = DataLoader(dataset, batch_size=cfg['batch_size'], shuffle=True, num_workers=cfg['num_workers'],
                            pin_memory=True)
    return dataloader


def build_deepfashion_multimodal_swav_train(cfg):
    data_path = pathlib.Path(cfg['data_dir'])
    tfms = get_swav_transform(cfg['image_size'])

    dataset = DeepFashionMultimodalDataset(data_path, train=True, transform=tfms)

    dataloader = DataLoader(dataset, batch_size=cfg['batch_size'], shuffle=True, num_workers=cfg['num_workers'],
                            pin_memory=True)
    return dataloader

