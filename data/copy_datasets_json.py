#!/usr/bin/env python3
"""
Script to copy JSON files from datasets directory to experiments directory
with modified paths to be relative to the current working directory.
Supports all datasets defined in datasets_list.yml configuration.
"""

import os
import json
import shutil
import argparse
import yaml
from pathlib import Path

# 脚本使用相对路径，工作目录为data/目录所在路径


def load_json_file(file_path):
    """Load JSON data from file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_json_file(data, file_path):
    """Save JSON data to file."""
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def modify_paths_in_json(data, dataset_name):
    """
    Modify image paths in JSON data to be relative to current working directory.
    
    Args:
        data: JSON data (list or dict)
        dataset_name: Name of the dataset (e.g., 'car_196')
    
    Returns:
        Modified JSON data
    """
    def fix_path(path, dataset_name):
        """Fix path for datasets with special directory structures"""
        original_path = path
        
        # Special path mappings for different datasets
        if dataset_name == 'caltech101':
            # caltech101: images/ -> 101_ObjectCategories/
            if path.startswith('images/'):
                path = path.replace('images/', '101_ObjectCategories/')
        elif dataset_name == 'caltech256':
            # caltech256: images/ -> 256_ObjectCategories/
            if path.startswith('images/'):
                path = path.replace('images/', '256_ObjectCategories/')
        elif dataset_name == 'flowers_102':
            # flowers_102: images/category/image.jpg -> jpg/image.jpg
            if path.startswith('images/'):
                # Remove 'images/' and category subdirectory
                path_parts = path.split('/')
                if len(path_parts) >= 3:
                    # Keep only the filename after removing images/ and category/
                    path = f"jpg/{path_parts[-1]}"
        elif dataset_name == 'food_101':
            # food_101: images/ -> jpg/
            if path.startswith('images/'):
                path = path.replace('images/', 'jpg/')
        elif dataset_name == 'CUB_200_2011':
            # CUB_200_2011: images/ -> CUB_200_2011/images/
            if path.startswith('images/'):
                path = path.replace('images/', 'CUB_200_2011/images/')
        elif dataset_name == 'fgvc_aircraft':
            # fgvc_aircraft: images/category/image.jpg -> images/image.jpg
            if path.startswith('images/'):
                # Remove 'images/' and category subdirectory
                path_parts = path.split('/')
                if len(path_parts) >= 3:
                    # Keep only the filename after removing images/ and category/
                    path = f"images/{path_parts[-1]}"
        elif dataset_name == 'dogs_120':
            # dogs_120: images/category/image.jpg -> Images/nXXXXXX-category/image.jpg
            if path.startswith('images/'):
                # Remove 'images/' prefix, keep category and filename
                path_parts = path.split('/')
                if len(path_parts) >= 3:
                    category = path_parts[1]
                    filename = path_parts[2]
                    # Find the actual directory name by scanning the Images directory
                    import os
                    dataset_root = f"./datasets/{dataset_name}"
                    images_dir = f"{dataset_root}/Images"
                    if os.path.exists(images_dir):
                        # Find directory that contains the category name
                        for dir_name in os.listdir(images_dir):
                            if f"-{category}" in dir_name:
                                path = f"Images/{dir_name}/{filename}"
                                break
        elif dataset_name == 'SUN397':
            # SUN397: images/ -> images/ (no change needed, already correct)
            # Path format: images/a/abbey/sun_xxx.jpg
            # Keep as is since it's already in correct format
            pass
        elif dataset_name == 'birdsnap':
            # birdsnap: images/ -> images/ (no change needed, already correct)
            # Path format: images/Class_Name/image.jpg (class names normalized with underscores)
            # Keep as is since it's already in correct format
            pass
        
        # Convert to full path relative to current working directory
        modified_path = f"./datasets/{dataset_name}/{path}"
        
        # Debug info for path changes
        if original_path != path:
            print(f"  Path fix: {original_path} -> {path}")
        
        return modified_path
    
    if isinstance(data, list):
        # Handle images.json format: [[class_name, class_id, [image_paths]], ...]
        modified_data = []
        for class_info in data:
            if isinstance(class_info, list) and len(class_info) >= 3:
                class_name, class_id, image_paths = class_info[0], class_info[1], class_info[2]
                modified_paths = []
                for path in image_paths:
                    modified_path = fix_path(path, dataset_name)
                    modified_paths.append(modified_path)
                modified_data.append([class_name, class_id, modified_paths])
            else:
                modified_data.append(class_info)
        return modified_data
    
    elif isinstance(data, dict):
        # Handle split_dataset_images.json format: {"train": [...], "test": [...], "val": [...]}
        modified_data = {}
        for split_name, split_data in data.items():
            if isinstance(split_data, list):
                modified_split_data = []
                for item in split_data:
                    if isinstance(item, list) and len(item) >= 3:
                        class_name, class_id, image_path = item[0], item[1], item[2]
                        # Convert single image path
                        modified_path = fix_path(image_path, dataset_name)
                        modified_split_data.append([class_name, class_id, modified_path])
                    else:
                        modified_split_data.append(item)
                modified_data[split_name] = modified_split_data
            else:
                modified_data[split_name] = split_data
        return modified_data
    
    return data


def copy_dataset_json_files(dataset_name, source_dir, target_dir):
    """
    Copy and modify JSON files for a specific dataset.
    
    Args:
        dataset_name: Name of the dataset (e.g., 'car_196')
        source_dir: Source directory containing JSON files
        target_dir: Target directory to save modified JSON files
    """
    # Define JSON files to copy
    json_files = [
        'images.json',
        'images_test.json',
        'images_val.json',  # May not exist for all datasets
        'images_train.json',
        f'split_{dataset_name.replace("_", "")}_images.json',
        'images_discovery_all.json',
        'images_discovery_all_1.json',
        'images_discovery_all_2.json',
        'images_discovery_all_3.json',
        'images_discovery_all_4.json',
        'images_discovery_all_5.json',
        'images_discovery_all_6.json',
        'images_discovery_all_7.json',
        'images_discovery_all_8.json',
        'images_discovery_all_9.json',
        'images_discovery_all_10.json',
        'images_discovery_random.json'
    ]
    
    # Create target directory if it doesn't exist
    os.makedirs(target_dir, exist_ok=True)
    
    copied_files = []
    skipped_files = []
    
    for json_file in json_files:
        source_path = os.path.join(source_dir, json_file)
        target_path = os.path.join(target_dir, json_file)
        
        if os.path.exists(source_path):
            print(f"Processing {json_file}...")
            
            # Load original JSON data
            try:
                data = load_json_file(source_path)
                
                # Modify paths in the JSON data
                modified_data = modify_paths_in_json(data, dataset_name)
                
                # Save modified JSON data
                save_json_file(modified_data, target_path)
                copied_files.append(json_file)
                print(f"  ✓ Copied and modified {json_file}")
                
            except Exception as e:
                print(f"  ✗ Error processing {json_file}: {e}")
                skipped_files.append(json_file)
        else:
            print(f"  - Skipped {json_file} (file not found)")
            skipped_files.append(json_file)
    
    print(f"\nSummary for {dataset_name}:")
    print(f"  Copied: {len(copied_files)} files")
    print(f"  Skipped: {len(skipped_files)} files")
    
    if copied_files:
        print(f"  Copied files: {', '.join(copied_files)}")
    if skipped_files:
        print(f"  Skipped files: {', '.join(skipped_files)}")


def get_datasets_from_config(config_file, specific_dataset=None):
    """
    Get dataset configurations from YAML file.
    
    Args:
        config_file: Path to the YAML configuration file
        specific_dataset: If provided, only return this dataset
    
    Returns:
        List of dataset configurations
    """
    with open(config_file, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    experiments_root = config.get('experiments_root', './experiments')
    dataset_mapping = config.get('dataset_mapping', {})
    
    datasets = []
    
    # All supported datasets
    supported_datasets = [
        'car', 'pet', 'aircraft', 'eurosat', 'food', 'dtd', 
        'caltech101', 'caltech256', 'deepfashion_multimodal', 
        'sun397', 'imagenet_a', 'imagenet_r', 'dog', 'bird', 'flower'
    ]
    
    # Filter datasets based on specific_dataset or use all supported
    datasets_to_process = [specific_dataset] if specific_dataset else supported_datasets
    
    for dataset_key in datasets_to_process:
        if dataset_key in dataset_mapping:
            dataset_config = dataset_mapping[dataset_key]
            data_dir = dataset_config['data_dir']
            
            # Handle special subdirectory for CUB_200_2011
            if 'special_subdir' in dataset_config:
                source_dir = f'./datasets/{data_dir}/{dataset_config["special_subdir"]}/images_split'
            else:
                source_dir = f'./datasets/{data_dir}/images_split'
            
            datasets.append({
                'key': dataset_key,
                'name': data_dir,
                'source_dir': source_dir,
                'target_dir': f'./{experiments_root}/{dataset_config["experiment_dir"]}/images_split'
            })
        else:
            print(f"Warning: Dataset '{dataset_key}' not found in configuration")
    
    return datasets


def main():
    """Main function to copy JSON files for datasets."""
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Copy JSON files for datasets')
    parser.add_argument('--dataset', type=str, help='Specific dataset to process (e.g., pet, car, aircraft)')
    parser.add_argument('--config', type=str, default='./configs/datasets_list.yml',
                       help='Path to configuration file')
    args = parser.parse_args()
    
    config_file = args.config
    
    print("Starting JSON file copying process...")
    print("=" * 50)
    
    if args.dataset:
        print(f"Processing specific dataset: {args.dataset}")
    else:
        print("Processing all supported datasets")
    
    # Get datasets from configuration
    datasets = get_datasets_from_config(config_file, args.dataset)
    
    if not datasets:
        print("No datasets found to process!")
        return
    
    print(f"Found {len(datasets)} dataset(s) to process")
    print("-" * 50)
    
    for dataset in datasets:
        print(f"\nProcessing dataset: {dataset['key']} ({dataset['name']})")
        print(f"Source: {dataset['source_dir']}")
        print(f"Target: {dataset['target_dir']}")
        print("-" * 30)
        
        # Check if source directory exists
        if not os.path.exists(dataset['source_dir']):
            print(f"  ✗ Source directory does not exist: {dataset['source_dir']}")
            continue
        
        copy_dataset_json_files(
            dataset_name=dataset['name'],
            source_dir=dataset['source_dir'],
            target_dir=dataset['target_dir']
        )
    
    print("\n" + "=" * 50)
    print("JSON file copying process completed!")


if __name__ == "__main__":
    main()
