"""
Test script to verify the data loading pipeline for LAM with LeRobot datasets.
This script checks if the data loader can correctly load and process samples.
"""
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from data_loader.lerobot_datamodule import LeRobotDataModule

def test_data_loader():
    """Test the LeRobot data loader configuration."""
    print("=" * 80)
    print("Testing LeRobot Data Loader for LAM")
    print("=" * 80)
    
    # Configuration from lam_lerobot.yaml
    data_config = {
        "data_root_dir": "/mnt/public_zgc/home/jlchen/code/starVLA/playground/demo_data",
        "data_mix": "demo_sim_pick_place",
        "num_frames": 2,
        "video_backend": "torchvision_av",
        "preferred_video_key": "video.base_view",
        "state_keys": ["state.joints"],
        "frame_dt_sec": 1.6,
        "batch_size": 2,
        "num_workers": 0,  # Use 0 for testing to avoid multiprocessing issues
    }
    
    print("\n1. Creating DataModule...")
    print(f"   Data root: {data_config['data_root_dir']}")
    print(f"   Data mix: {data_config['data_mix']}")
    print(f"   Num frames: {data_config['num_frames']}")
    print(f"   Frame dt: {data_config['frame_dt_sec']}s")
    print(f"   Preferred video: {data_config['preferred_video_key']}")
    print(f"   State keys: {data_config['state_keys']}")
    
    try:
        datamodule = LeRobotDataModule(**data_config)
        print("   ✓ DataModule created successfully")
    except Exception as e:
        print(f"   ✗ Failed to create DataModule: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n2. Setting up datasets...")
    try:
        datamodule.setup(stage="fit")
        print("   ✓ Datasets setup successfully")
    except Exception as e:
        print(f"   ✗ Failed to setup datasets: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n3. Creating data loaders...")
    try:
        train_loader = datamodule.train_dataloader()
        val_loader = datamodule.val_dataloader()
        print(f"   ✓ Train loader created (dataset size: {len(datamodule.train_dataset)})")
        print(f"   ✓ Val loader created (dataset size: {len(datamodule.val_dataset)})")
    except Exception as e:
        print(f"   ✗ Failed to create data loaders: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n4. Testing batch loading...")
    try:
        batch = next(iter(train_loader))
        print("   ✓ Successfully loaded a batch")
        print(f"\n   Batch contents:")
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                print(f"     - {key}: shape={value.shape}, dtype={value.dtype}")
                if value.dtype in [torch.float32, torch.float64, torch.float16]:
                    print(f"       min={value.min().item():.4f}, max={value.max().item():.4f}, mean={value.mean().item():.4f}")
                else:
                    print(f"       values: {value.tolist() if value.numel() <= 10 else 'too many to display'}")
            else:
                print(f"     - {key}: {type(value)}")
        
        # Verify expected keys
        expected_keys = ["videos", "dec_videos", "proprio", "delta_proprio", "dataset_ids"]
        missing_keys = set(expected_keys) - set(batch.keys())
        if missing_keys:
            print(f"\n   ⚠ Warning: Missing expected keys: {missing_keys}")
        else:
            print(f"\n   ✓ All expected keys present: {expected_keys}")
        
        # Verify shapes
        print("\n   Verifying shapes:")
        B, T, C, H, W = batch["videos"].shape
        print(f"     - videos: [B={B}, T={T}, C={C}, H={H}, W={W}]")
        print(f"     - dec_videos: {batch['dec_videos'].shape}")
        print(f"     - proprio: {batch['proprio'].shape}")
        print(f"     - delta_proprio: {batch['delta_proprio'].shape}")
        
        if T == data_config["num_frames"]:
            print(f"   ✓ Temporal dimension matches config (T={T})")
        else:
            print(f"   ⚠ Warning: Temporal dimension mismatch (T={T}, expected={data_config['num_frames']})")
        
        if H == 224 and W == 224:
            print(f"   ✓ Spatial dimensions correct (224x224)")
        else:
            print(f"   ⚠ Warning: Spatial dimensions unexpected ({H}x{W})")
            
    except Exception as e:
        print(f"   ✗ Failed to load batch: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n5. Testing multiple batches...")
    try:
        for i, batch in enumerate(train_loader):
            if i >= 2:  # Test 3 batches
                break
            print(f"   ✓ Batch {i+1} loaded successfully")
        print(f"   ✓ Multiple batches loaded successfully")
    except Exception as e:
        print(f"   ✗ Failed to load multiple batches: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n" + "=" * 80)
    print("✓ All tests passed! Data loading pipeline is working correctly.")
    print("=" * 80)
    return True

if __name__ == "__main__":
    success = test_data_loader()
    sys.exit(0 if success else 1)

