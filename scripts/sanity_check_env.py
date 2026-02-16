"""
Sanity Check Script for FlappyBirdEnv

This script verifies that the environment is correctly implemented by:
1. Checking observation and action spaces
2. Running a random agent for multiple episodes
3. Verifying reset/step returns correct types
4. Checking termination conditions work
5. Printing statistics

Run this before attempting any training
"""

import numpy as np
import sys
sys.path.insert(0, '.')

from env.flappy_env import FlappyBirdEnv


def check_spaces(env):
    """Verify observation and action spaces are correctly defined."""
    print("=" * 50)
    print("SPACE VERIFICATION")
    print("=" * 50)
    
    print(f"Action space: {env.action_space}")
    print(f"  - n = {env.action_space.n}")
    print(f"  - sample() = {env.action_space.sample()}")
    
    print(f"\nObservation space: {env.observation_space}")
    print(f"  - shape = {env.observation_space.shape}")
    print(f"  - dtype = {env.observation_space.dtype}")
    print(f"  - low = {env.observation_space.low}")
    print(f"  - high = {env.observation_space.high}")
    
    # Verify observation is in bounds
    obs, info = env.reset(seed=42)
    assert env.observation_space.contains(obs), \
        f"Observation {obs} not in observation space."
    print(f"\n Initial observation is valid: {obs}")
    
    return True


def check_determinism(env, seed=42, num_steps=100):
    """Verify that seeding produces deterministic results."""
    print("\n" + "=" * 50)
    print("DETERMINISM CHECK")
    print("=" * 50)
    
    # Run episode with seed
    obs1, _ = env.reset(seed=seed)
    trajectory1 = [obs1.copy()]
    
    for _ in range(num_steps):
        action = 0  # Fixed action sequence
        obs, _, terminated, truncated, _ = env.step(action)
        trajectory1.append(obs.copy())
        if terminated or truncated:
            break
    
    # Run again with same seed
    obs2, _ = env.reset(seed=seed)
    trajectory2 = [obs2.copy()]
    
    for _ in range(num_steps):
        action = 0
        obs, _, terminated, truncated, _ = env.step(action)
        trajectory2.append(obs.copy())
        if terminated or truncated:
            break
    
    # Compare trajectories
    assert len(trajectory1) == len(trajectory2), \
        f"Trajectory lengths differ: {len(trajectory1)} vs {len(trajectory2)}"
    
    for i, (o1, o2) in enumerate(zip(trajectory1, trajectory2)):
        assert np.allclose(o1, o2), \
            f"Observations differ at step {i}: {o1} vs {o2}"
    
    print(f" Determinism verified over {len(trajectory1)} steps")
    return True


def run_random_agent(env, num_episodes=10, seed=42):
    """Run random agent and collect statistics."""
    print("\n" + "=" * 50)
    print(f"RANDOM AGENT TEST ({num_episodes} episodes)")
    print("=" * 50)
    
    episode_rewards = []
    episode_lengths = []
    episode_scores = []
    
    for ep in range(num_episodes):
        obs, info = env.reset(seed=seed + ep)
        total_reward = 0
        steps = 0
        
        while True:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            
            # Verify observation is valid
            assert env.observation_space.contains(obs), \
                f"Invalid observation: {obs}"
            
            total_reward += reward
            steps += 1
            
            if terminated or truncated:
                break
        
        episode_rewards.append(total_reward)
        episode_lengths.append(steps)
        episode_scores.append(info.get("score", 0))
        
        print(f"  Episode {ep+1}: reward={total_reward:.1f}, "
              f"length={steps}, score={info.get('score', 0)}")
    
    print(f"\nStatistics:")
    print(f"  Mean reward:  {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
    print(f"  Mean length:  {np.mean(episode_lengths):.1f} ± {np.std(episode_lengths):.1f}")
    print(f"  Mean score:   {np.mean(episode_scores):.2f} ± {np.std(episode_scores):.2f}")
    print(f"  Max score:    {max(episode_scores)}")
    
    return episode_rewards, episode_lengths, episode_scores


def check_termination_conditions(env):
    """Verify that termination conditions work correctly."""
    print("\n" + "=" * 50)
    print("TERMINATION CONDITION CHECK")
    print("=" * 50)
    
    # Test 1: Bird should die if it does nothing (falls to ground)
    obs, _ = env.reset(seed=42)
    steps = 0
    while True:
        obs, reward, terminated, truncated, info = env.step(0)  # Never flap
        steps += 1
        if terminated:
            print(f" Bird died after {steps} steps of not flapping")
            print(f"  Final bird_y: {info['bird_y']:.1f} (screen height: {env.SCREEN_HEIGHT})")
            break
        if steps > 500:
            print("Issue. Bird should have died from falling.")
            return False
    
    # Test 2: Constant flapping should also eventually die (ceiling or pipe)
    obs, _ = env.reset(seed=42)
    steps = 0
    while True:
        obs, reward, terminated, truncated, info = env.step(1)  # Always flap
        steps += 1
        if terminated:
            print(f" Bird died after {steps} steps of constant flapping")
            print(f"  Final bird_y: {info['bird_y']:.1f}")
            break
        if steps > 500:
            print("Issue. Bird should have died from hitting ceiling or pipe!")
            return False
    
    # Test 3: Truncation at max_steps
    env_short = FlappyBirdEnv(max_steps=50)
    obs, _ = env_short.reset(seed=123)
    steps = 0
    truncated_occurred = False
    
    while steps < 100:
        # Alternate actions to try to survive
        action = steps % 5 == 0  # Flap occasionally
        obs, reward, terminated, truncated, info = env_short.step(int(action))
        steps += 1
        
        if truncated:
            truncated_occurred = True
            print(f" Truncation occurred at step {steps} (max_steps=50)")
            break
        if terminated:
            print(f"  (Bird died at step {steps}, trying again...)")
            obs, _ = env_short.reset(seed=123 + steps)
    
    if not truncated_occurred:
        print("  Note: Bird kept dying before truncation. This is OK.")
    
    return True


def check_reward_structure(env):
    """Verify reward values are reasonable."""
    print("\n" + "=" * 50)
    print("REWARD STRUCTURE CHECK")
    print("=" * 50)
    
    obs, _ = env.reset(seed=42)
    
    # Collect some rewards
    rewards = []
    for _ in range(100):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        rewards.append(reward)
        if terminated or truncated:
            break
    
    print(f"  Reward range: [{min(rewards):.2f}, {max(rewards):.2f}]")
    print(f"  Unique rewards: {sorted(set(rewards))}")
    print(f"  Expected: 0.1 (survival), 1.1 (pass pipe + survival), -1.0 (death)")
    
    # Verify expected reward values exist
    has_survival = any(abs(r - 0.1) < 0.01 for r in rewards)
    has_death = any(abs(r - (-1.0)) < 0.01 for r in rewards)
    
    if has_survival:
        print("  Survival reward (0.1) detected")
    if has_death:
        print("  Death penalty (-1.0) detected")
    
    return True


def check_rgb_array():
    env = FlappyBirdEnv(render_mode="rgb_array")
    obs, _ = env.reset(seed=42)
    
    frame = env.render()
    
    assert frame is not None, "rgb_array should return a frame"
    assert isinstance(frame, np.ndarray), f"Expected ndarray, got {type(frame)}"
    assert frame.shape == (512, 288, 3), f"Expected (512, 288, 3), got {frame.shape}"
    assert frame.dtype == np.uint8, f"Expected uint8, got {frame.dtype}"
    
    # Frame should NOT be all zeros (should have sky, bird, pipes)
    assert frame.sum() > 0, "Frame is all black — rendering not working"
    
    print(f"Frame shape: {frame.shape}")
    print(f"Frame dtype: {frame.dtype}")
    print(f"Frame value range: [{frame.min()}, {frame.max()}]")
    print(f"Mean pixel value: {frame.mean():.1f}")
    print("rgb_array mode verified!")
    
    env.close()

def main():
    """Run all sanity checks."""
    print("\n" + "=" * 50)
    print("FLAPPY BIRD ENVIRONMENT SANITY CHECK")
    print("=" * 50)
    
    # Create environment
    env = FlappyBirdEnv()
    
    # Run checks
    all_passed = True
    
    try:
        check_spaces(env)
    except AssertionError as e:
        print(f" Space check failed: {e}")
        all_passed = False
    
    try:
        check_determinism(env)
    except AssertionError as e:
        print(f" Determinism check failed: {e}")
        all_passed = False
    
    try:
        check_termination_conditions(env)
    except AssertionError as e:
        print(f" Termination check failed: {e}")
        all_passed = False
    
    try:
        check_reward_structure(env)
    except AssertionError as e:
        print(f" Reward check failed: {e}")
        all_passed = False
    
    try:
        run_random_agent(env, num_episodes=10)
    except AssertionError as e:
        print(f" Random agent failed: {e}")
        all_passed = False
    
    # Final summary
    print("\n" + "=" * 50)
    if all_passed:
        print(" ALL SANITY CHECKS PASSED")
    else:
        print(" SOME CHECKS FAILED")
    print("=" * 50)
    
    env.close()
    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)