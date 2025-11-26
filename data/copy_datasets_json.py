#!/usr/bin/env python3
"""
Script to copy JSON files from datasets directory to experiments directory
with modified paths to be relative to the current working directory.
"""

import os
import json
import shutil
from pathlib import Path


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
    if isinstance(data, list):
        # Handle images.json format: [[class_name, class_id, [image_paths]], ...]
        modified_data = []
        for class_info in data:
            if isinstance(class_info, list) and len(class_info) >= 3:
                class_name, class_id, image_paths = class_info[0], class_info[1], class_info[2]
                modified_paths = []
                for path in image_paths:
                    # Convert relative path to full path relative to current working directory
                    modified_path = f"./datasets/{dataset_name}/{path}"
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
                        modified_path = f"./datasets/{dataset_name}/{image_path}"
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


def main():
    """Main function to copy JSON files for datasets."""
    
    # Load dataset configuration from YAML
    config_file = '/home/hdl/project/fgvr_test_new/configs/datasets_list.yml'
    with open(config_file, 'r', encoding='utf-8') as f:
        import yaml
        config = yaml.safe_load(f)
    
    # Define dataset configurations based on YAML mapping
    experiments_root = config.get('experiments_root', './experiments')
    datasets = []
    
    # Add car dataset as specified
    car_config = config['dataset_mapping']['car']
    datasets.append({
        'name': 'car_196',
        'source_dir': f'/home/hdl/project/fgvr_test_new/datasets/{car_config["data_dir"]}/images_split',
        'target_dir': f'/home/hdl/project/fgvr_test_new/{experiments_root}/{car_config["experiment_dir"]}/images_split'
    })
    
    print("Starting JSON file copying process...")
    print("=" * 50)
    
    for dataset in datasets:
        print(f"\nProcessing dataset: {dataset['name']}")
        print(f"Source: {dataset['source_dir']}")
        print(f"Target: {dataset['target_dir']}")
        print("-" * 30)
        
        copy_dataset_json_files(
            dataset_name=dataset['name'],
            source_dir=dataset['source_dir'],
            target_dir=dataset['target_dir']
        )
    
    print("\n" + "=" * 50)
    print("JSON file copying process completed!")


if __name__ == "__main__":
    main()
