import os
import torch
from torchvision import datasets
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from data.data_stats import BIRD_STATS
import pathlib
import random
import shutil
from copy import deepcopy
from data.utils import get_swav_transform

# 定义超类与子类名称，用于后续 prompt 构造
SUPERCLASS = 'bird'
CLASSUNIT = 'species'

# ----------------------------- 构造训练示例 Prompt -----------------------------
bird_how_to1 = f"""
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
# 上面构造了一个 instruct prompt，包括示例、任务步骤、Bird 属性清单模板等。


# ----------------------------- 第二个 Prompt 模板 -----------------------------
bird_how_to2 = f"""
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
# 第二个 Prompt 用于训练模型回答 "给出属性列表"

# ----------------------------- 图像变换: resize+crop+normalize -----------------------------
def _transform(n_px):
    return transforms.Compose([
        transforms.Resize(n_px, interpolation=transforms.InterpolationMode.BICUBIC),  # 双三次插值 resize
        transforms.CenterCrop(n_px),  # 中心裁剪
        lambda image: image.convert("RGB"),  # 确保图像为 RGB
        transforms.ToTensor(),  # 转换为 Tensor
        # CLIP 风格归一化
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
                             (0.26862954, 0.26130258, 0.27577711)),
    ])


# =============================================================================
#                                BirdPrompter
# =============================================================================
class BirdPrompter:
    """生成Bird属性相关的提示词，供LLM描述与推断使用"""
    def __init__(self):
        self.first_question = "general"
        self.attributes = ['bill shape','wing color','upperparts color','underparts color','breast pattern','back color','tail shape','upper tail color','head pattern','breast color','throat color','eye color','bill length','forehead color','under tail color','nape color','belly color','wing shape','size','shape','back pattern','tail pattern','belly pattern','primary color','leg color','bill color','crown color','wing pattern','habitat']

    def _generate_question_prompt(self, attr):
        # habitat 使用问句格式
        return f"Questions: What is the {attr} of the bird in this image. Answer:"

    def _generate_statement_prompt(self, attr):
        # 其他属性使用描述格式
        return f"Describe the {attr} of the bird in this image."

    def get_attributes(self):
        # 返回属性列表（含 general description）
        list_attributes = ['General Description']
        list_attributes.extend(self.attributes)
        return list_attributes

    def get_attribute_prompt(self):
        # 为每个属性构造 prompt 句子
        list_prompts = ["Describe this image in details."]

        for attr in self.attributes:
            if attr == 'habitat':
                list_prompts.append(self._generate_question_prompt(attr))
            else:
                list_prompts.append(self._generate_statement_prompt(attr))

        return list_prompts

    def get_llm_prompt(self, list_attr_val):
        # list_attr_val = [(属性名, 属性值), ...]
        # 长度应 = 1 (general) + 29 个属性        
        if len(list_attr_val) != 1 + len(self.attributes):
            raise IndexError(
                f"The length of the given attribute-value pair list does not fit the attribute list."
                f"CUB has {1 + len(self.attributes)} attributes, but the given length is {len(list_attr_val)}.")

        prompt = f"""
        I have a photo of a bird. 
        Your task is to perform the following actions:
        1 - Summarize the information you get about the bird from the general description and attribute description \
        delimited by triple backticks with five sentences.
        2 - Infer and list three possible species name of the bird in this photo based on the information you get.
        3 - Output a JSON object that uses the following format
        <three possible species>: [
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
        Three possible species: <three possible species names>
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
        - '''{list_attr_val[17][0]}''': '''{list_attr_val[17][1]}'''
        - '''{list_attr_val[18][0]}''': '''{list_attr_val[18][1]}'''
        - '''{list_attr_val[19][0]}''': '''{list_attr_val[19][1]}'''
        - '''{list_attr_val[20][0]}''': '''{list_attr_val[20][1]}'''
        - '''{list_attr_val[21][0]}''': '''{list_attr_val[21][1]}'''
        - '''{list_attr_val[22][0]}''': '''{list_attr_val[22][1]}'''
        - '''{list_attr_val[23][0]}''': '''{list_attr_val[23][1]}'''
        - '''{list_attr_val[24][0]}''': '''{list_attr_val[24][1]}'''
        - '''{list_attr_val[25][0]}''': '''{list_attr_val[25][1]}'''
        - '''{list_attr_val[26][0]}''': '''{list_attr_val[26][1]}'''
        - '''{list_attr_val[27][0]}''': '''{list_attr_val[27][1]}'''
        - '''{list_attr_val[28][0]}''': '''{list_attr_val[28][1]}'''
        - '''{list_attr_val[29][0]}''': '''{list_attr_val[29][1]}'''
        """
        return prompt

# =============================================================================
#                                BirdDiscovery200
# =============================================================================
class BirdDiscovery200:
    def __init__(self, root, folder_suffix=''):
        img_root = os.path.join(root, f'images_discovery_all{folder_suffix}')

        # 读取类别文件夹
        self.class_folders = os.listdir(img_root)  # 100 x 1
        for i in range(len(self.class_folders)):
            self.class_folders[i] = os.path.join(img_root, self.class_folders[i])

        self.samples = []       # 图像路径列表
        self.targets = []       # 标签
        self.subcategories = [] # 子类名

        for folder in self.class_folders:
            # 类别编号取文件夹前3位，如 "003.Blue Jay"
            label = int(folder.split('/')[-1][:3])
            label = label - 1
            file_names = os.listdir(folder)

            for name in file_names:
                self.targets.append(label)
                self.samples.append(os.path.join(folder, name))
                # 从文件夹名称提取类别名称
                class_name = folder.split('/')[-1].split('.', 1)[1] if '.' in folder.split('/')[-1] else folder.split('/')[-1]
                self.subcategories.append(class_name)

        self.classes = BIRD_STATS['class_names']
        
        # 添加subcat_to_sample属性，按类别名分组图片路径
        from collections import defaultdict
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
        img = Image.open(self.samples[self.index]).convert("RGB")
        target = self.targets[self.index]
        self.index += 1
        return img, target


# =============================================================================
#                           BirdDataset (CUB-200-2011)
# =============================================================================
class BirdDataset(datasets.ImageFolder):
    """
    Wrapper for the CUB-200-2011 dataset.
    Method DatasetBirds.__getitem__() returns tuple of image and its corresponding label.
    Dataset per https://github.com/slipnitskaya/caltech-birds-advanced-classification
    """
    def __init__(self, root, train=True, transform=None, loader=datasets.folder.default_loader):
        img_root = os.path.join(root, 'images')
        super(BirdDataset, self).__init__(root=img_root, transform=None, loader=loader)

        self.redefine_class_to_idx()

        self.transform_ = transform
        self.target_transform_ = None
        self.train = train

        # ------------------ 根据 train_test_split.txt 筛选 sample ------------------
        path_to_splits = os.path.join(root, 'train_test_split.txt')
        indices_to_use = list()
        with open(path_to_splits, 'r') as in_file:
            for line in in_file:
                idx, use_train = line.strip('\n').split(' ', 2)
                if bool(int(use_train)) == self.train:
                    indices_to_use.append(int(idx))

        # ------------------ 根据 images.txt 获得文件名 ------------------
        path_to_index = os.path.join(root, 'images.txt')
        filenames_to_use = set()
        with open(path_to_index, 'r') as in_file:
            for line in in_file:
                idx, fn = line.strip('\n').split(' ', 2)
                if int(idx) in indices_to_use:
                    filenames_to_use.add(fn)

        # 生成 {“文件名”：在self.imgs中的下标}
        img_paths_cut = {'/'.join(img_path.rsplit('/', 2)[-2:]): idx for idx, (img_path, lb) in enumerate(self.imgs)}
        imgs_to_use = [self.imgs[img_paths_cut[fn]] for fn in filenames_to_use]

        # 根据 filenames 筛选 self.imgs
        _, targets_to_use = list(zip(*imgs_to_use))

        # 重写 ImageFolder 的 imgs/samples 属性
        self.imgs = self.data = self.samples = imgs_to_use
        self.data = [entry[0] for entry in self.imgs]
        self.targets = targets_to_use
        self.target = self.targets
        self.classes = BIRD_STATS['class_names']

    def __getitem__(self, index):
        # generate one sample
        sample, target = super(BirdDataset, self).__getitem__(index)

        if self.transform_ is not None:
            sample = self.transform_(sample)

        # return sample, target, index
        return sample, target, self.samples[index][0]         # 返回图像路径用于可视化

    def redefine_class_to_idx(self):
        # 重定义 class_to_idx 字典，将文件名转换为人类可读的类别名
        adjusted_dict = {}
        for k, v in self.class_to_idx.items():
            k = k.split('.')[-1].replace('_', ' ')
            split_key = k.split(' ')
            if len(split_key) > 2:
                k = '-'.join(split_key[:-1]) + " " + split_key[-1]
            adjusted_dict[k] = v
        self.class_to_idx = adjusted_dict

# =============================================================================
#                     构建 prompter 与 dataset/dataloader
# =============================================================================
def build_bird_prompter(cfg: dict):
    prompter = BirdPrompter()
    return prompter


def build_bird200_discovery(cfg: dict, folder_suffix=''):
    set_to_discover = BirdDiscovery200(cfg['data_dir'], folder_suffix=folder_suffix)
    return set_to_discover


def build_bird200_test(cfg):
    data_path = pathlib.Path(cfg['data_dir'])
    tfms = _transform(cfg['image_size'])

    dataset = BirdDataset(data_path, train=False, transform=tfms)

    dataloader = DataLoader(dataset, batch_size=cfg['batch_size'], shuffle=True, num_workers=cfg['num_workers'],
                            pin_memory=True)
    return dataloader


def build_bird200_swav_train(cfg):
    data_path = pathlib.Path(cfg['data_dir'])
    tfms = get_swav_transform(cfg['image_size'])

    dataset = BirdDataset(data_path, train=False, transform=tfms)

    dataloader = DataLoader(dataset, batch_size=cfg['batch_size'], shuffle=True, num_workers=cfg['num_workers'],
                            pin_memory=True)
    return dataloader


# =============================================================================
#                              Discovery Set 构造
# =============================================================================
def create_discovery_set(root, files, targets, class_names, num_per_category=3, selection='largest'):
    """
    为每类选出 num_per_category 张图片，复制到 root 下对应类别文件夹。
    selection = random 或 largest（挑选文件大小最大的）
    """
    target_image_dict = {i: [] for i in range(len(class_names))}
    cleaned_names = deepcopy(class_names)
    cleaned_names = [name.replace(' ', '_') for name in cleaned_names]
    cleaned_names = [name.replace('-', '_') for name in cleaned_names]

    folder_template = "{}.{}"
    file_template = "{}.{}_{}"

    for label, image in zip(targets, files):
        target_image_dict[label].append(image)

    num_copied = 0
    for label, images in target_image_dict.items():
        image_paths = deepcopy(images)

        if selection == 'random':
            random.shuffle(image_paths)
        else:
            image_paths.sort(key=lambda x: os.path.getsize(x), reverse=True)
            image_paths = image_paths[:50]
            random.shuffle(image_paths)

        selected_images = image_paths[:num_per_category]

        destination_folder = folder_template.format(str(label).zfill(3), cleaned_names[label])
        destination_folder = os.path.join(root, destination_folder)
        if not os.path.exists(destination_folder):
            os.makedirs(destination_folder)

        for src_img in selected_images:
            dest_img = file_template.format(str(label).zfill(3), cleaned_names[label], src_img.split('/')[-1])
            destination_path = os.path.join(destination_folder, dest_img)
            shutil.copyfile(src_img, destination_path)
            num_copied += 1
            print(f"[{num_copied}]  copied <{destination_path}> from <{src_img}>")


# ----------------------------- 生成长尾分布 -----------------------------
def generate_long_tail_distribution(num_categories=200):
    import numpy as np
    import matplotlib.pyplot as plt

    categories = num_categories

    # Generate power-law distributed random numbers.
    # The parameter 'a' controls the shape of the distribution.
    # Smaller 'a' gives a heavier tail.
    a = 2.0
    samples = np.random.zipf(a, categories)

    # Adjust samples to ensure they're within the range 1 to 10
    adjusted_samples = np.clip(samples, 1, 10)

    # Sort samples in descending order
    sorted_samples = np.sort(adjusted_samples)[::-1]

    # Verification: Print number of images for first 20 categories
    for i in range(20):
        print(f"Category {i + 1}: {sorted_samples[i]} images")

    # Optional: Plot to see the distribution
    plt.bar(range(categories), sorted_samples)
    plt.xlabel('Categories')
    plt.ylabel('Number of Images')
    plt.title('Distribution of Images across Categories')
    plt.show()

# ----------------------------- 长尾 discovery set 构造 -----------------------------
def create_long_tail_discovery_set(root, files, targets, class_names, selection='largest'):
    distribution = [
        10, 10, 10, 10, 10, 10, 10,  8,  8,  8,  8,  7,  7,  7,  5,  5,  5,
        5,  5,  5,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  3,  3,  3,
        3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  2,  2,
        2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,
        2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,
        1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,
        1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,
        1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,
        1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,
        1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,
        1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,
        1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1
                    ]

    # selection = 'random' or 'largest'
    target_image_dict = {i: [] for i in range(len(class_names))}
    cleaned_names = deepcopy(class_names)
    cleaned_names = [name.replace(' ', '_') for name in cleaned_names]
    cleaned_names = [name.replace('-', '_') for name in cleaned_names]

    folder_template = "{}.{}"
    file_template = "{}.{}_{}"

    for label, image in zip(targets, files):
        target_image_dict[label].append(image)

    num_copied = 0
    for label, images in target_image_dict.items():
        num_per_category = distribution[label]
        image_paths = deepcopy(images)

        if selection == 'random':
            random.shuffle(image_paths)
        else:
            image_paths.sort(key=lambda x: os.path.getsize(x), reverse=True)
            image_paths = image_paths[:50]
            random.shuffle(image_paths)

        selected_images = image_paths[:num_per_category]

        destination_folder = folder_template.format(str(label).zfill(3), cleaned_names[label])
        destination_folder = os.path.join(root, destination_folder)
        if not os.path.exists(destination_folder):
            os.makedirs(destination_folder)

        for src_img in selected_images:
            dest_img = file_template.format(str(label).zfill(3), cleaned_names[label], src_img.split('/')[-1])
            destination_path = os.path.join(destination_folder, dest_img)
            shutil.copyfile(src_img, destination_path)
            num_copied += 1
            print(f"[{num_copied}]  copied <{destination_path}> from <{src_img}>")


# =============================================================================
#                                 主函数调试
# =============================================================================
if __name__ == "__main__":
    root = "/home/miu/GoldsGym/global_datasets/CUB_200_2011/CUB_200_2011"
    tfms = _transform(224)
    cfg = {
        "data_dir": root,
        "image_size": 224,
        "batch_size": 32,
        "num_workers": 8,
    }
    dataset = BirdDataset(root, train=False, transform=tfms)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=8, pin_memory=True)

    print(f'Num All Classes: {len(set(dataset.targets))}')
    print(f"Num All Images: {len(dataset.samples)}")

    print(f'Len set: {len(dataset)}')
    print(f"Image {dataset.samples[-1]} has Label {dataset.targets[-1]} whose class name {dataset.classes[dataset.targets[-1]]}")


    # build_bird200_test(cfg)
    # discovery_set = build_bird200_discovery(cfg)
    # print(f"Num of images to discovery = {len(discovery_set)}")

    # print("Create random unlabeled set as wild discovery set")
    # num_per_category_list = [1, 2, 4, 5, 6, 7, 8, 9]
    # # num_per_category_list = [3]
    # for num_per_category in num_per_category_list:
    #     root_disco = os.path.join(root, f"images_discovery_all_{num_per_category}")
    #
    #     if not os.path.exists(root_disco):
    #         os.makedirs(root_disco)
    #
    #     create_discovery_set(root_disco, dataset.data, dataset.target, dataset.classes,
    #                          num_per_category=num_per_category,
    #                          selection='largest')

    # print("Create long-tailed unlabeled set as wild discovery set")
    # root_disco = os.path.join(root, f"images_discovery_all_random")
    #
    # if not os.path.exists(root_disco):
    #     os.makedirs(root_disco)
    #
    # create_long_tail_discovery_set(
    #     root_disco, dataset.data, dataset.target, dataset.classes, selection='largest'
    # )