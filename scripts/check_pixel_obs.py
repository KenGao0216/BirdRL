"""
Verify pixel observation pipeline.

Checks:
1. Observation shape is (1, 84, 84)
2. Observation dtype is uint8
3. Values are in [0, 255]
4. Frame is not all-black or all-white
5. Saves sample frames to disk for visual inspection
"""

import sys
sys.path.insert(0, '.')

import numpy as np
import cv2
import os

from env.flappy_env import FlappyBirdEnv
from env.wrappers import PixelObservationWrapper


def main():
    print("=" * 50)
    print("PIXEL OBSERVATION SANITY CHECK")
    print("=" * 50)
    
    # Create wrapped environment
    base_env = FlappyBirdEnv(render_mode="rgb_array")
    env = PixelObservationWrapper(base_env)
    
    print(f"\nObservation space: {env.observation_space}")
    print(f"  Shape: {env.observation_space.shape}")
    print(f"  Dtype: {env.observation_space.dtype}")
    print(f"  Low:   {env.observation_space.low.min()}")
    print(f"  High:  {env.observation_space.high.max()}")
    
    # Reset and check
    obs, info = env.reset(seed=42)
    
    print(f"\nObservation after reset:")
    print(f"  Shape: {obs.shape}")
    print(f"  Dtype: {obs.dtype}")
    print(f"  Min:   {obs.min()}")
    print(f"  Max:   {obs.max()}")
    print(f"  Mean:  {obs.mean():.1f}")
    
    # Verify shape and dtype
    assert obs.shape == (1, 84, 84), f"Expected (1, 84, 84), got {obs.shape}"
    assert obs.dtype == np.uint8, f"Expected uint8, got {obs.dtype}"
    assert env.observation_space.contains(obs), "Observation not in space!"
    
    # Verify not all-black
    assert obs.max() > 0, "Frame is all black"
    
    # Verify not all same value
    assert obs.std() > 1.0, f"Frame has very low variance: {obs.std():.2f}"
    
    print("\n  All basic checks passed")
    
    # Save sample frames for visual inspection
    output_dir = "debug_frames"
    os.makedirs(output_dir, exist_ok=True)
    
    # Save initial frame
    frame_2d = obs[0]  # Remove channel dim: (84, 84)
    cv2.imwrite(f"{output_dir}/pixel_obs_reset.png", frame_2d)
    
    # Also save the full-resolution RGB frame for comparison
    full_frame = base_env.render()
    if full_frame is not None:
        # OpenCV expects BGR, our frame is RGB
        full_bgr = cv2.cvtColor(full_frame, cv2.COLOR_RGB2BGR)
        cv2.imwrite(f"{output_dir}/full_rgb_frame.png", full_bgr)
    
    # Run a few steps and save frames
    print("\n  Running 5 steps and saving frames...")
    for i in range(5):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        
        assert obs.shape == (1, 84, 84), f"Step {i}: shape {obs.shape}"
        assert obs.dtype == np.uint8, f"Step {i}: dtype {obs.dtype}"
        assert env.observation_space.contains(obs), f"Step {i}: obs not in space"
        
        frame_2d = obs[0]
        cv2.imwrite(f"{output_dir}/pixel_obs_step{i+1}.png", frame_2d)
        
        print(f"    Step {i+1}: action={'FLAP' if action else '----'}, "
              f"range=[{obs.min()}, {obs.max()}], mean={obs.mean():.1f}")
        
        if terminated or truncated:
            print(f"    Episode ended at step {i+1}")
            break
    
    print(f"\n  Saved debug frames to {output_dir}/")
    print(f"  Check these files to verify the observations look correct:")
    print(f"    - full_rgb_frame.png  (original 288x512 RGB)")
    print(f"    - pixel_obs_reset.png (processed 84x84 grayscale)")
    print(f"    - pixel_obs_step*.png (subsequent frames)")
    
    # Final multi-episode check
    print("\n  Running 10 episodes to check stability...")
    for ep in range(10):
        obs, _ = env.reset(seed=ep)
        steps = 0
        while True:
            obs, _, terminated, truncated, _ = env.step(env.action_space.sample())
            steps += 1
            
            assert env.observation_space.contains(obs), \
                f"Episode {ep}, step {steps}: obs not in space"
            
            if terminated or truncated:
                break
    
    print("  10 episodes completed without errors")
    
    print("\n" + "=" * 50)
    print("ALL PIXEL OBSERVATION CHECKS PASSED")
    print("=" * 50)
    
    env.close()


if __name__ == "__main__":
    main()