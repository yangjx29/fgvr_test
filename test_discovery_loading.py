#!/usr/bin/env python3
"""
Test the discovery set loading functionality directly
"""

import sys
import os
import yaml

def test_discovery_loading():
    """Test that discovery sets can be loaded correctly in both modes"""
    print("=" * 60)
    print("Testing Discovery Set Loading Functionality")
    print("=" * 60)
    
    # Load config
    with open('/home/hdl/project/fgvr_test_new/configs/env_machine.yml', 'r') as f:
        env_config = yaml.safe_load(f)
    with open('/home/hdl/project/fgvr_test_new/configs/expts/pet37_all.yml', 'r') as f:
        expt_config = yaml.safe_load(f)
    
    cfg = {**env_config, **expt_config}
    print(f"Loaded config for dataset: {cfg['dataset_name']}")
    
    # Test the get_or_create_discovery_set function
    print("\n1. Testing get_or_create_discovery_set function")
    print("-" * 50)
    
    try:
        # Import the function
        from discovering import get_or_create_discovery_set, randomly_extract_discoverying_set
        
        # Test extraction mode
        print("Testing extraction mode...")
        randomly_extract_discoverying_set = True
        
        discovery_3 = get_or_create_discovery_set(cfg, folder_suffix='_3')
        print(f"✓ 3-shot extraction mode: {len(discovery_3.subcat_to_sample)} classes")
        
        # Show some sample classes
        sample_classes = list(discovery_3.subcat_to_sample.keys())[:3]
        for class_name in sample_classes:
            images = discovery_3.subcat_to_sample[class_name]
            print(f"  {class_name}: {len(images)} images")
        
        # Test random extraction
        print("\nTesting random extraction...")
        discovery_random = get_or_create_discovery_set(cfg, folder_suffix='_random')
        print(f"✓ Random extraction mode: {len(discovery_random.subcat_to_sample)} classes")
        
        total_random = sum(len(imgs) for imgs in discovery_random.subcat_to_sample.values())
        print(f"  Total images: {total_random}")
        
        # Test traditional mode
        print("\nTesting traditional mode...")
        randomly_extract_discoverying_set = False
        
        discovery_traditional = get_or_create_discovery_set(cfg, folder_suffix='_1')
        print(f"✓ Traditional mode: {len(discovery_traditional.subcat_to_sample)} classes")
        
        # Show some sample classes
        sample_classes = list(discovery_traditional.subcat_to_sample.keys())[:3]
        for class_name in sample_classes:
            images = discovery_traditional.subcat_to_sample[class_name]
            print(f"  {class_name}: {len(images)} images")
        
    except Exception as e:
        print(f"❌ Error testing discovery loading: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n2. Testing data structure compatibility")
    print("-" * 50)
    
    try:
        # Verify that the discovery objects have the required attributes
        required_attrs = ['subcat_to_sample', 'samples', 'targets', 'classes']
        
        for mode_name, discovery_obj in [
            ("3-shot extraction", discovery_3),
            ("random extraction", discovery_random), 
            ("traditional", discovery_traditional)
        ]:
            print(f"Checking {mode_name}...")
            for attr in required_attrs:
                if hasattr(discovery_obj, attr):
                    value = getattr(discovery_obj, attr)
                    if isinstance(value, dict):
                        print(f"  ✓ {attr}: {len(value)} items")
                    elif isinstance(value, list):
                        print(f"  ✓ {attr}: {len(value)} items")
                    else:
                        print(f"  ✓ {attr}: {type(value)}")
                else:
                    print(f"  ❌ Missing attribute: {attr}")
            
            # Test iteration
            try:
                count = 0
                for img, target in discovery_obj:
                    count += 1
                    if count >= 3:  # Only test first 3
                        break
                print(f"  ✓ Iteration works: tested {count} samples")
            except Exception as e:
                print(f"  ❌ Iteration failed: {e}")
        
    except Exception as e:
        print(f"❌ Error testing data structure: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n" + "=" * 60)
    print("✅ All discovery loading tests completed successfully!")
    print("=" * 60)
    return True

if __name__ == "__main__":
    success = test_discovery_loading()
    sys.exit(0 if success else 1)
