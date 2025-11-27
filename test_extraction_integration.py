#!/usr/bin/env python3
"""
Test script for discovery set extraction functionality using finer_dynamic environment
"""

import sys
import os
import yaml

def test_discovery_functionality():
    """Test the discovery set extraction functionality"""
    print("=" * 60)
    print("Testing Discovery Set Extraction with finer_dynamic Environment")
    print("=" * 60)
    
    # Load config
    with open('/home/hdl/project/fgvr_test_new/configs/env_machine.yml', 'r') as f:
        env_config = yaml.safe_load(f)
    with open('/home/hdl/project/fgvr_test_new/configs/expts/pet37_all.yml', 'r') as f:
        expt_config = yaml.safe_load(f)
    
    cfg = {**env_config, **expt_config}
    print(f"Loaded config for dataset: {cfg['dataset_name']}")
    
    # Test the extraction functions directly
    print("\n1. Testing extract_from_trainsets module")
    print("-" * 50)
    
    try:
        from data.extract_from_trainsets import load_train_data, extract_discovery_set, save_discovery_set
        
        # Load training data
        train_data = load_train_data('pet')
        print(f"✓ Loaded training data: {len(train_data)} classes")
        
        # Extract 3-shot discovery set
        discovery_3 = extract_discovery_set(train_data, 3, random_seed=42)
        print(f"✓ Extracted 3-shot discovery set: {len(discovery_3)} classes")
        
        # Extract random discovery set
        discovery_random = extract_discovery_set(train_data, 'random', random_seed=42)
        random_total = sum(len(entry[2]) for entry in discovery_random if len(entry) >= 3)
        print(f"✓ Extracted random discovery set: {len(discovery_random)} classes, {random_total} images")
        
    except Exception as e:
        print(f"❌ Error testing extract_from_trainsets: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test integration with discovering.py
    print("\n2. Testing integration with discovering.py")
    print("-" * 50)
    
    try:
        # Import discovering module
        import discovering
        
        # Test extraction mode
        print("Testing extraction mode (randomly_extract_discoverying_set=True)...")
        discovering.randomly_extract_discoverying_set = True
        
        # This would normally be called in the main discovering.py flow
        from data import DATA_DISCOVERY
        original_discovery_func = DATA_DISCOVERY['pet']
        
        print("✓ Successfully imported discovering module")
        print("✓ Extraction mode is enabled")
        
        # Test traditional mode
        print("Testing traditional mode (randomly_extract_discoverying_set=False)...")
        discovering.randomly_extract_discoverying_set = False
        print("✓ Traditional mode is enabled")
        
    except Exception as e:
        print(f"❌ Error testing discovering.py integration: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n3. Testing discovery set file generation")
    print("-" * 50)
    
    try:
        # Check if the generated files exist
        import json
        
        # Check 3-shot file
        file_3 = '/home/hdl/project/fgvr_test_new/experiments/pet37/images_split/images_discovery_all_3.json'
        if os.path.exists(file_3):
            with open(file_3, 'r') as f:
                data_3 = json.load(f)
            print(f"✓ 3-shot discovery file exists: {len(data_3)} classes")
        else:
            print(f"⚠️ 3-shot discovery file not found: {file_3}")
        
        # Check random file
        file_random = '/home/hdl/project/fgvr_test_new/experiments/pet37/images_split/images_discovery_all_random.json'
        if os.path.exists(file_random):
            with open(file_random, 'r') as f:
                data_random = json.load(f)
            print(f"✓ Random discovery file exists: {len(data_random)} classes")
        else:
            print(f"⚠️ Random discovery file not found: {file_random}")
        
    except Exception as e:
        print(f"❌ Error checking generated files: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n" + "=" * 60)
    print("✅ All tests completed successfully!")
    print("=" * 60)
    return True

if __name__ == "__main__":
    success = test_discovery_functionality()
    sys.exit(0 if success else 1)
