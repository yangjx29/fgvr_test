from PIL import Image
import torchvision.transforms as transforms
import os
import random
import shutil
from copy import deepcopy
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from data.data_stats import AIRCRAFT_STATS
import pathlib
from data.utils import get_swav_transform
from collections import defaultdict


SUPERCLASS = 'aircraft'
CLASSUNIT = 'variants'


aircraft_how_to1 = f"""
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

aircraft_how_to2 = f"""
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


class AircraftPrompter:
    def __init__(self):
        self.supercategory = "aircraft"
        self.first_question = "general"
        self.attributes = [
            'wing shape', 'wing position', 'engine count', 'engine position',
            'fuselage length', 'fuselage width', 'nose shape', 'cockpit windows',
            'tail shape', 'tail configuration', 'landing gear type', 'wing span',
            'aircraft size', 'livery color scheme', 'wing tip design', 'engine nacelle shape',
            'vertical stabilizer', 'horizontal stabilizer', 'fuselage cross-section',
            'overall aircraft proportions'
        ]

    def _generate_question_prompt(self, attr):
        return f"Questions: What is the {attr} of the {self.supercategory} in this photo? Answer:"

    def get_attributes(self):
        list_attributes = ['General Description']
        list_attributes.extend(self.attributes)
        return list_attributes

    def get_attribute_prompt(self):
        list_prompts = ["Describe this image in details."]
        for attr in self.attributes:
            list_prompts.append(self._generate_question_prompt(attr))
        return list_prompts

    def get_llm_prompt(self, attr_descr_pairs):
        prompt = f"""
        I have a photo of an {self.supercategory}. 
        Your task is to perform the following actions:
        1 - Summarize the information you get about the {self.supercategory} from the general description and \
        attribute descriptions delimited by triple backticks with five sentences.
        2 - The description might not be correct and accurate. So I need you to infer and list three possible detailed aircraft variant names \
        of the {self.supercategory} in this photo based on the information you get.
        3 - Output a JSON object that uses the following format
        <three possible aircraft variant names>: [
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
        - ...
        Summary: <summary>
        Three possible aircraft variants: <three possible aircraft variants>
        Output JSON: <output JSON object>

        '''{attr_descr_pairs[0][0]}''': '''{attr_descr_pairs[0][1]}'''
        Attributes List:"""
        
        for i in range(1, min(len(attr_descr_pairs), 21)):
            prompt += f"\n        - '''{attr_descr_pairs[i][0]}''': '''{attr_descr_pairs[i][1]}'''"
        
        return prompt


class AircraftDiscovery100:
    def __init__(self, root, folder_suffix=''):
        img_root = os.path.join(root, f'images_discovery_all{folder_suffix}')

        self.class_folders = os.listdir(img_root)
        for i in range(len(self.class_folders)):
            self.class_folders[i] = os.path.join(img_root, self.class_folders[i])

        self.samples = []
        self.targets = []
        self.subcategories = []
        
        for folder in self.class_folders:
            # 从文件夹名称提取类别名称
            class_name = os.path.basename(folder)
            file_names = os.listdir(folder)

            for name in file_names:
                if name.lower().endswith(('.jpg', '.jpeg', '.png')):
                    self.samples.append(os.path.join(folder, name))
                    self.subcategories.append(class_name)

        self.classes = AIRCRAFT_STATS['class_names']
        
        # 添加subcat_to_sample属性，按类别名分组图片路径
        self.subcat_to_sample = defaultdict(list)
        for subcat, sample in zip(self.subcategories, self.samples):
            self.subcat_to_sample[subcat].append(sample)
        
        self.index = 0

    def __len__(self):
        return len(self.samples)

    def __iter__(self):
        return self

    def __next__(self):
        if self.index >= len(self.samples):
            raise StopIteration
        img_path = self.samples[self.index]
        target = self.index
        self.index += 1
        return img_path, target


class AircraftDataset(Dataset):
    """
    FGVC-Aircraft Dataset
    """
    def __init__(self, root, train=True, transform=None, limit=0):
        self.root = root
        self.train = train
        self.transform = transform
        
        self.data = []
        self.target = []
        self.classes = AIRCRAFT_STATS['class_names']
        
        # Load from split files
        if train:
            split_file = os.path.join(root, 'images_variant_train.txt')
        else:
            split_file = os.path.join(root, 'images_variant_test.txt')
        
        images_dir = os.path.join(root, 'images')
        
        with open(split_file, 'r') as f:
            for line in f:
                parts = line.strip().split(maxsplit=1)
                if len(parts) == 2:
                    img_id, variant_name = parts
                    img_path = os.path.join(images_dir, f"{img_id}.jpg")
                    if os.path.exists(img_path):
                        self.data.append(img_path)
                        try:
                            label = self.classes.index(variant_name)
                            self.target.append(label)
                        except ValueError:
                            print(f"Warning: {variant_name} not in class list")

        self.uq_idxs = np.array(range(len(self)))

    def __getitem__(self, idx):
        image = Image.open(self.data[idx]).convert('RGB')
        target = self.target[idx]

        if self.transform is not None:
            image = self.transform(image)

        idx = self.uq_idxs[idx]
        return image, target, self.data[idx]

    def __len__(self):
        return len(self.data)


def build_aircraft_prompter(cfg: dict = None):
    prompter = AircraftPrompter()
    return prompter


def build_aircraft100_discovery(cfg: dict, folder_suffix=''):
    set_to_discover = AircraftDiscovery100(cfg['data_dir'], folder_suffix=folder_suffix)
    return set_to_discover


def build_aircraft100_test(cfg):
    data_path = pathlib.Path(cfg['data_dir'])
    tfms = _transform(cfg['image_size'])

    dataset = AircraftDataset(data_path, train=False, transform=tfms)

    dataloader = DataLoader(dataset, batch_size=cfg['batch_size'], shuffle=True, num_workers=cfg['num_workers'],
                            pin_memory=True)
    return dataloader


def build_aircraft100_swav_train(cfg):
    data_path = pathlib.Path(cfg['data_dir'])
    tfms = get_swav_transform(cfg['image_size'])

    dataset = AircraftDataset(data_path, train=True, transform=tfms)

    dataloader = DataLoader(dataset, batch_size=cfg['batch_size'], shuffle=True, num_workers=cfg['num_workers'],
                            pin_memory=True)
    return dataloader

