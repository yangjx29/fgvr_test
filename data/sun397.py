import os
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from data.data_stats import SUN397_STATS
import pathlib
import random
import shutil
from copy import deepcopy
from data.utils import get_swav_transform


SUPERCLASS = 'scene'
CLASSUNIT = 'categories'


sun397_how_to1 = f"""
Your task is to tell me what are the useful attributes for distinguishing {SUPERCLASS} {CLASSUNIT} in a photo of a {SUPERCLASS}.

Specifically, you can complete the task by following the instructions below:
1 - I give you an example delimited by <> about what are the useful attributes for distinguishing bird species in 
a photo of a bird. You should understand and learn this example carefully.
2 - List the useful attributes for distinguishing {SUPERCLASS} {CLASSUNIT} in a photo of a {SUPERCLASS}.
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
The useful attributes for distinguishing {SUPERCLASS} {CLASSUNIT} in a photo of a {SUPERCLASS}:
===
"""


sun397_how_to2 = f"""
Please tell me what are the useful attributes for distinguishing {SUPERCLASS} {CLASSUNIT} in a photo of a {SUPERCLASS} according to the 
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
Question: What are the useful attributes for distinguishing {SUPERCLASS} {CLASSUNIT} in a photo of a {SUPERCLASS}?
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


class Sun397Prompter:
    def __init__(self):
        self.supercategory = "scene"
        self.first_question = "general"
        self.attributes = [
            'scene type', 'spatial layout', 'architectural style', 'lighting conditions',
            'time of day', 'weather conditions', 'color palette', 'texture and materials',
            'scale and perspective', 'human presence', 'object composition',
            'environmental context', 'structural elements', 'atmospheric qualities',
            'functional purpose', 'cultural context', 'geographical features'
        ]

    def _generate_question_prompt(self, attr):
        return f"Questions: What is the {attr} of the {self.supercategory} in this photo. Answer:"

    def _generate_statement_prompt(self, attr):
        return f" What is the {attr} of the scene:"

    def get_attributes(self):
        list_attributes = ['General Description']
        list_attributes.extend(self.attributes)
        return list_attributes

    def get_attribute_prompt(self):
        list_prompts = ["Look at this photo carefully. Describe what you see in detail, including the scene's appearance, features, and any notable characteristics. Be specific and descriptive."]
        for attr in self.attributes:
            list_prompts.append(self._generate_statement_prompt(attr))
        return list_prompts

    def get_llm_prompt(self, list_attr_val):
        prompt = f"""
        I have a photo of a {self.supercategory}. 
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
        Three possible scene category names: <three possible scene category names>
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


class Sun397Discovery:
    """
    SUN397 Discovery数据集加载器
    
    注意：SUN397有嵌套的类别结构（如 a/abbey, a/apartment_building/outdoor），
    但discovery集是扁平化的（每个类别一个目录）。
    类别名需要保持完整的嵌套路径格式。
    """
    def __init__(self, root, folder_suffix=''):
        img_root = os.path.join(root, f'images_discovery_all{folder_suffix}')
        print(f"构建发现集,img_root: {img_root}")
        
        # discovery集是扁平化的，每个类别一个目录
        self.class_folders = os.listdir(img_root)
        for i in range(len(self.class_folders)):
            self.class_folders[i] = os.path.join(img_root, self.class_folders[i])

        self.classes = SUN397_STATS['class_names']
        self.samples = []
        self.targets = []
        self.subcategories = []
        
        for folder in self.class_folders:
            # discovery集中的文件夹名是扁平化的（如 "abbey"），需要映射到嵌套格式（如 "a/abbey"）
            folder_name = folder.split('/')[-1]
            
            # 查找对应的嵌套类别名
            # 首先尝试精确匹配（如果discovery集使用了嵌套格式）
            class_name = None
            label = None
            
            # 尝试直接匹配（如果discovery集已经使用了嵌套格式）
            for idx, cls_name in enumerate(self.classes):
                # 提取嵌套类别名的最后部分进行比较
                cls_name_parts = cls_name.split('/')
                cls_name_last = cls_name_parts[-1]
                
                if cls_name_last == folder_name or cls_name == folder_name:
                    class_name = cls_name
                    label = idx
                    break
            
            # 如果还是找不到，尝试模糊匹配（处理下划线和空格）
            if label is None:
                folder_name_normalized = folder_name.replace('_', ' ').lower()
                for idx, cls_name in enumerate(self.classes):
                    cls_name_parts = cls_name.split('/')
                    cls_name_last = cls_name_parts[-1].replace('_', ' ').lower()
                    if cls_name_last == folder_name_normalized:
                        class_name = cls_name
                        label = idx
                        break

            # 如果仍然找不到，尝试基于词袋的嵌套类别匹配（处理 "indoor casino"、"urban canal" 等多词情况）
            if label is None:
                # 将文件夹名拆分为小写词列表（支持空格和下划线）
                folder_tokens = sorted(folder_name.lower().replace('_', ' ').split())
                if folder_tokens:
                    for idx, cls_name in enumerate(self.classes):
                        # 跳过前缀字母（如 a/badlands 的 "a"），只使用后续路径片段
                        parts = cls_name.split('/')[1:]
                        cls_tokens = []
                        for part in parts:
                            cls_tokens.extend(part.lower().replace('_', ' ').split())
                        if sorted(cls_tokens) == folder_tokens:
                            class_name = cls_name
                            label = idx
                            break
            
            if label is None:
                print(f"Warning: Could not find label for class folder: {folder_name}")
                continue
            
            file_names = os.listdir(folder)
            for name in file_names:
                if name.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                    self.subcategories.append(class_name)  # 使用完整的嵌套类别名
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


class Sun397Dataset(Dataset):
    """SUN397 Dataset"""
    def __init__(self, root, train=True, transform=None):
        self.root = root
        self.train = train
        self.transform = transform
        self.classes = SUN397_STATS['class_names']
        
        # 加载图像路径和标签
        self.data = []
        self.target = []
        self._load_dataset()

    def _load_dataset(self):
        # SUN397的数据在images目录下，有嵌套结构
        images_dir = os.path.join(self.root, 'images')
        if not os.path.exists(images_dir):
            # 如果没有images目录，尝试从images_discovery_all获取
            images_dir = os.path.join(self.root, 'images_discovery_all')
            if not os.path.exists(images_dir):
                raise FileNotFoundError(f"数据集目录不存在: {images_dir}")
        
        # 遍历所有类别目录（嵌套结构）
        for class_idx, class_name in enumerate(self.classes):
            # 将类别名（如 "a/abbey"）转换为路径
            class_path = class_name.replace('/', os.sep)
            class_dir = os.path.join(images_dir, class_path)
            
            if not os.path.exists(class_dir):
                continue
            
            # 获取该类别的所有图像
            image_files = []
            for f in os.listdir(class_dir):
                if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                    image_files.append(os.path.join(class_dir, f))
            
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


def build_sun397_prompter(cfg: dict):
    prompter = Sun397Prompter()
    return prompter


def build_sun397_discovery(cfg: dict, folder_suffix=''):
    set_to_discover = Sun397Discovery(cfg['data_dir'], folder_suffix=folder_suffix)
    return set_to_discover


def build_sun397_test(cfg):
    data_path = pathlib.Path(cfg['data_dir'])
    tfms = _transform(cfg['image_size'])

    dataset = Sun397Dataset(data_path, train=False, transform=tfms)

    dataloader = DataLoader(dataset, batch_size=cfg['batch_size'], shuffle=True, num_workers=cfg['num_workers'],
                            pin_memory=True)
    return dataloader


def build_sun397_swav_train(cfg):
    data_path = pathlib.Path(cfg['data_dir'])
    tfms = get_swav_transform(cfg['image_size'])

    dataset = Sun397Dataset(data_path, train=True, transform=tfms)

    dataloader = DataLoader(dataset, batch_size=cfg['batch_size'], shuffle=True, num_workers=cfg['num_workers'],
                            pin_memory=True)
    return dataloader

