import os
from PIL import Image
import torchvision.transforms as transforms
import pathlib
from typing import Any, Callable, Optional, Union, Tuple
from typing import Sequence
import numpy as np
from torchvision.datasets.folder import default_loader
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from data.data_stats import UCF101_STATS
import random
import shutil
from copy import deepcopy
from data.utils import get_swav_transform
import json


SUPERCLASS = 'human action'
CLASSUNIT = 'action types'


ucf101_how_to1 = f"""
Your task is to tell me what are the useful attributes for distinguishing {SUPERCLASS} {CLASSUNIT} in a photo or video frame.

Specifically, you can complete the task by following the instructions below:
1 - I give you an example delimited by <> about what are the useful attributes for distinguishing bird species in 
a photo of a bird. You should understand and learn this example carefully.
2 - List the useful attributes for distinguishing {SUPERCLASS} {CLASSUNIT} in a photo or video frame.
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
The useful attributes for distinguishing {SUPERCLASS} {CLASSUNIT} in a photo or video frame:
===
"""

ucf101_how_to2 = f"""
Please tell me what are the useful attributes for distinguishing {SUPERCLASS} {CLASSUNIT} in a photo or video frame according to the 
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
Question: What are the useful attributes for distinguishing {SUPERCLASS} {CLASSUNIT} in a photo or video frame?
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


class UCF101Prompter:
    def __init__(self):
        self.supercategory = 'human action'
        self.first_question = "general"

        self.action_attributes = [
            'body posture', 'body movement', 'arm position', 'leg position', 
            'hand gesture', 'head position', 'facial expression', 'body orientation',
            'movement speed', 'movement direction', 'action intensity', 'action duration',
            'object interaction', 'tool usage', 'environment context', 'background setting',
            'number of people', 'person interaction', 'clothing type', 'equipment used',
            'action phase', 'motion trajectory', 'body balance', 'weight distribution',
            'action purpose', 'action complexity', 'spatial relationship', 'temporal sequence'
        ]

    def _generate_whatis_prompt(self, attr):
        return f"Questions: What is the {attr} in this photo. Answer:"

    def _generate_statement_prompt(self, attr):
        return f"Describe the {attr} in this photo."

    def get_attributes(self):
        list_attributes = ['General Description']
        list_attributes.extend(self.action_attributes)
        return list_attributes

    def get_attribute_prompt(self):
        list_prompts = ["Describe this image in details."]
        for attr in self.action_attributes:
            list_prompts.append(self._generate_statement_prompt(attr))
        return list_prompts

    def get_llm_prompt(self, list_attr_val):
        prompt = f"""
        I have a photo showing a {self.supercategory}. 
        Your task is to perform the following actions:
        1 - Summarize the information you get about the {self.supercategory} from the general description and attribute description \
        delimited by triple backticks with five sentences.
        2 - Infer and list three possible action types shown in this photo based on the information you get.
        3 - Output a JSON object that uses the following format
        <three possible action types>: [
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
        Three possible action types: <three possible action types>
        Output JSON: <output JSON object>

        '''{list_attr_val[0][0]}''': '''{list_attr_val[0][1]}'''

        Attributes List:
        """
        
        # Add all attributes dynamically
        for i in range(1, min(len(list_attr_val), 29)):
            prompt += f"        - '''{list_attr_val[i][0]}''': '''{list_attr_val[i][1]}'''\n"
        
        return prompt


class UCF101Discovery:
    def __init__(self, data_dir, folder_suffix=''):
        self.data_dir = data_dir
        self.folder_suffix = folder_suffix
        self.class_names = UCF101_STATS['class_names']
        self.num_classes = UCF101_STATS['num_classes']

    def get_class_names(self):
        return self.class_names

    def get_num_classes(self):
        return self.num_classes


class UCF101Dataset(Dataset):
    """UCF-101 Dataset for action recognition"""
    
    def __init__(self, root, train=True, transform=None, loader=default_loader):
        self.root = pathlib.Path(root)
        self.train = train
        self.transform = transform
        self.loader = loader
        
        # Load from JSON files
        if train:
            json_file = self.root / 'images_split' / 'images_train.json'
        else:
            json_file = self.root / 'images_split' / 'images_test.json'
        
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Parse JSON data
        self.samples = []
        self.targets = []
        self.classes = []
        
        for class_info in data:
            class_name, class_id, image_paths = class_info
            if class_name not in self.classes:
                self.classes.append(class_name)
            
            for img_path in image_paths:
                # Convert relative path to absolute path
                full_path = self.root / img_path
                self.samples.append((str(full_path), class_id))
                self.targets.append(class_id)
        
        self.data = [s[0] for s in self.samples]
        self.target = self.targets

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[idx]
        sample = self.loader(path)

        if self.transform is not None:
            sample = self.transform(sample)

        return sample, target, path


def build_ucf101_prompter(cfg: dict):
    prompter = UCF101Prompter()
    return prompter


def build_ucf101_discovery(cfg: dict, folder_suffix=''):
    set_to_discover = UCF101Discovery(cfg['data_dir'], folder_suffix=folder_suffix)
    return set_to_discover


def build_ucf101_test(cfg):
    data_path = pathlib.Path(cfg['data_dir'])
    tfms = _transform(cfg['image_size'])

    dataset = UCF101Dataset(data_path, train=False, transform=tfms)

    dataloader = DataLoader(dataset, batch_size=cfg['batch_size'], shuffle=True, num_workers=cfg['num_workers'],
                            pin_memory=True)
    return dataloader


def build_ucf101_swav_train(cfg):
    data_path = pathlib.Path(cfg['data_dir'])
    tfms = get_swav_transform(cfg['image_size'])

    dataset = UCF101Dataset(data_path, train=True, transform=tfms)

    dataloader = DataLoader(dataset, batch_size=cfg['batch_size'], shuffle=True, num_workers=cfg['num_workers'],
                            pin_memory=True)
    return dataloader


if __name__ == "__main__":
    root = "/home/hdl/datasets/fgvr_fixed/ucf_101"
    tfms = _transform(224)

    dataset = UCF101Dataset(root, train=True, transform=tfms)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=8, pin_memory=True)

    print(f'Num All Classes: {len(set(dataset.target))}')
    print(f"Num All Images: {len(dataset.data)}")
    print(f'Len set: {len(dataset)}')
    
    if len(dataset) > 0:
        test_idx = 0
        print(f"Image {dataset.data[test_idx]} has Label {dataset.target[test_idx]} whose class name is {dataset.classes[dataset.target[test_idx]]}")
