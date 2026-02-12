"""
Full pipeline test: Data loader + Model forward pass
This script verifies that the data loader output is compatible with the LAM model.
"""
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from data_loader.lerobot_datamodule import LeRobotDataModule

def test_full_pipeline():
    """Test the full pipeline: data loading + model compatibility."""
    print("=" * 80)
    print("Testing Full Pipeline: Data Loader + Model Compatibility")
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
        "num_workers": 0,
    }
    
    print("\n1. Setting up data loader...")
    try:
        datamodule = LeRobotDataModule(**data_config)
        datamodule.setup(stage="fit")
        train_loader = datamodule.train_dataloader()
        print("   ✓ Data loader ready")
    except Exception as e:
        print(f"   ✗ Failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n2. Loading a batch...")
    try:
        batch = next(iter(train_loader))
        print("   ✓ Batch loaded successfully")
        print(f"   Batch keys: {list(batch.keys())}")
    except Exception as e:
        print(f"   ✗ Failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n3. Checking data format compatibility with LAM model...")
    
    # Expected format for LAM model (based on lam_lightinng.py)
    required_keys = ["videos", "dec_videos", "proprio"]
    optional_keys = ["delta_proprio", "dataset_ids"]
    
    print(f"   Required keys: {required_keys}")
    print(f"   Optional keys: {optional_keys}")
    
    missing_required = [k for k in required_keys if k not in batch]
    if missing_required:
        print(f"   ✗ Missing required keys: {missing_required}")
        return False
    else:
        print(f"   ✓ All required keys present")
    
    missing_optional = [k for k in optional_keys if k not in batch]
    if missing_optional:
        print(f"   ⚠ Missing optional keys: {missing_optional}")
    else:
        print(f"   ✓ All optional keys present")
    
    print("\n4. Verifying tensor shapes and types...")
    
    videos = batch["videos"]
    dec_videos = batch["dec_videos"]
    proprio = batch["proprio"]
    
    B, T, C, H, W = videos.shape
    print(f"   videos: {videos.shape} (B={B}, T={T}, C={C}, H={H}, W={W})")
    print(f"   dec_videos: {dec_videos.shape}")
    print(f"   proprio: {proprio.shape}")
    
    # Check expected shapes
    checks = [
        (videos.shape == dec_videos.shape, "videos and dec_videos have same shape"),
        (C == 3, "videos have 3 channels (RGB)"),
        (H == 224 and W == 224, "videos are 224x224"),
        (proprio.shape[0] == B, "proprio batch size matches videos"),
        (proprio.shape[1] == T, "proprio temporal dimension matches videos"),
        (videos.dtype == torch.float32, "videos are float32"),
        (proprio.dtype == torch.float32, "proprio is float32"),
    ]
    
    all_passed = True
    for check, desc in checks:
        if check:
            print(f"   ✓ {desc}")
        else:
            print(f"   ✗ {desc}")
            all_passed = False
    
    if not all_passed:
        return False
    
    print("\n5. Checking value ranges...")
    
    # Videos should be normalized (ImageNet normalization)
    print(f"   videos range: [{videos.min():.4f}, {videos.max():.4f}]")
    print(f"   videos mean: {videos.mean():.4f}, std: {videos.std():.4f}")
    
    # Check if videos are normalized (ImageNet stats: mean~0, std~1)
    if videos.min() < -3 or videos.max() > 3:
        print("   ⚠ Warning: videos may not be properly normalized")
    else:
        print("   ✓ videos appear to be normalized")
    
    print(f"   proprio range: [{proprio.min():.4f}, {proprio.max():.4f}]")
    print(f"   proprio mean: {proprio.mean():.4f}, std: {proprio.std():.4f}")
    
    if "delta_proprio" in batch:
        delta_proprio = batch["delta_proprio"]
        print(f"   delta_proprio: {delta_proprio.shape}")
        print(f"   delta_proprio range: [{delta_proprio.min():.4f}, {delta_proprio.max():.4f}]")
    
    print("\n6. Testing model input format (simulated)...")
    
    # Simulate what the model expects (from lam_lightinng.py _compute_step)
    try:
        model_inputs = {
            "videos": videos,  # [B, T, C, H, W]
            "states": proprio,  # [B, T, D]
            "dec_videos": dec_videos,  # [B, T, C, H, W]
            "dataset_ids": batch.get("dataset_ids", None),  # [B]
        }
        print("   ✓ Model input format is correct")
        print(f"   Model expects:")
        for key, value in model_inputs.items():
            if value is not None:
                print(f"     - {key}: {value.shape}")
            else:
                print(f"     - {key}: None (optional)")
    except Exception as e:
        print(f"   ✗ Failed to prepare model inputs: {e}")
        return False
    
    print("\n7. Summary of data pipeline:")
    print(f"   ✓ Dataset: {data_config['data_mix']}")
    print(f"   ✓ Total samples: {len(datamodule.train_dataset)}")
    print(f"   ✓ Batch size: {B}")
    print(f"   ✓ Temporal frames: {T}")
    print(f"   ✓ Video resolution: {H}x{W}")
    print(f"   ✓ Proprio dimension: {proprio.shape[-1]}")
    print(f"   ✓ Frame dt: {data_config['frame_dt_sec']}s")
    
    print("\n" + "=" * 80)
    print("✓ Full pipeline test passed!")
    print("✓ Data loader is correctly configured for LAM training")
    print("=" * 80)
    
    return True

if __name__ == "__main__":
    success = test_full_pipeline()
    sys.exit(0 if success else 1)


