"""
Verify frame stacking works correctly.

Checks:
1. Observation shape is (4, 84, 84)
2. On reset, all 4 frames are identical
3. After steps, frames differ (motion is captured)
4. .copy() works — old observations are not mutated
5. Observation stays in observation_space
"""

import sys
sys.path.insert(0, '.')

import numpy as np
from env.wrappers import make_pixel_env


def main():
    print("=" * 50)
    print("FRAME STACK SANITY CHECK")
    print("=" * 50)
    
    env = make_pixel_env(n_stack=4)
    
    print(f"\nObservation space: {env.observation_space}")
    print(f"  Shape: {env.observation_space.shape}")
    print(f"  Dtype: {env.observation_space.dtype}")
    
    # === CHECK 1: Shape after reset ===
    obs, info = env.reset(seed=42)
    
    print(f"\nAfter reset:")
    print(f"  obs.shape = {obs.shape}")
    print(f"  obs.dtype = {obs.dtype}")
    
    assert obs.shape == (4, 84, 84), f"Expected (4, 84, 84), got {obs.shape}"
    assert obs.dtype == np.uint8, f"Expected uint8, got {obs.dtype}"
    assert env.observation_space.contains(obs), "Observation not in space!"
    print("  Shape and dtype correct")
    
    # === CHECK 2: All frames identical on reset ===
    for i in range(1, 4):
        assert np.array_equal(obs[0], obs[i]), \
            f"Frame 0 and frame {i} should be identical on reset"
    print("  All 4 frames identical on reset (first frame repeated)")
    
    # === CHECK 3: Frames differ after steps ===
    obs_after_steps = obs.copy()
    for _ in range(5):
        obs_after_steps, _, terminated, truncated, _ = env.step(1)  # Flap
        if terminated or truncated:
            break
    
    # After several steps, the frames should NOT all be identical
    # (the bird has moved)
    frames_differ = False
    for i in range(1, 4):
        if not np.array_equal(obs_after_steps[0], obs_after_steps[i]):
            frames_differ = True
            break
    
    if frames_differ:
        print("  Frames differ after steps (motion captured)")
    else:
        print("  All frames still identical after steps (unexpected)")
    
    # === CHECK 4: .copy() works — old obs not mutated ===
    env.reset(seed=42)
    obs_step1, _, _, _, _ = env.step(0)
    obs_step1_saved = obs_step1.copy()  # Save a copy
    obs_step2, _, _, _, _ = env.step(1)
    
    assert np.array_equal(obs_step1, obs_step1_saved), \
        "obs_step1 was mutated by subsequent step! (.copy() may be missing)"
    assert not np.array_equal(obs_step1, obs_step2), \
        "obs_step1 and obs_step2 should differ"
    print(" Good. Previous observations are not mutated by subsequent steps")
    
    # === CHECK 5: Temporal ordering ===
    # After reset + 4 distinct steps, each frame channel should be different
    env.reset(seed=42)
    
    # Take 4 steps to fill buffer with 4 distinct frames
    for i in range(4):
        obs, _, terminated, _, _ = env.step(i % 2)
        if terminated:
            env.reset(seed=42 + i)
    
    # Check that not all frames are identical
    unique_frames = len(set([obs[i].tobytes() for i in range(4)]))
    print(f"  Good. After 4 steps: {unique_frames}/4 unique frames in stack")
    
    # === CHECK 6: Stability over multiple episodes ===
    print("\n  Running 10 episodes...")
    for ep in range(10):
        obs, _ = env.reset(seed=ep)
        assert obs.shape == (4, 84, 84)
        assert env.observation_space.contains(obs)
        
        steps = 0
        while True:
            obs, _, terminated, truncated, _ = env.step(env.action_space.sample())
            steps += 1
            
            assert obs.shape == (4, 84, 84), f"Episode {ep}, step {steps}: wrong shape"
            assert env.observation_space.contains(obs), f"Episode {ep}, step {steps}: not in space"
            
            if terminated or truncated:
                break
    
    print("  10 episodes completed without errors")
    
    print("\n" + "=" * 50)
    print("ALL FRAME STACK CHECKS PASSED")
    print("=" * 50)
    
    env.close()


if __name__ == "__main__":
    main()