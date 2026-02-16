"""
Quick test: Can DQN learn Flappy Bird with feature observations
"""

import sys
sys.path.insert(0, '.')

import os
import time
import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv
from env.flappy_env import FlappyBirdEnv


def make_feature_env():
    return FlappyBirdEnv(render_mode=None, max_steps=1000)


def evaluate(model, num_episodes=50):
    env = FlappyBirdEnv(render_mode=None, max_steps=2000)
    scores, lengths, rewards = [], [], []
    
    for ep in range(num_episodes):
        obs, info = env.reset(seed=ep)
        total_reward = 0
        steps = 0
        
        while True:
            action, _ = model.predict(obs[np.newaxis, ...], deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action.item())
            total_reward += reward
            steps += 1
            if terminated or truncated:
                break
        
        scores.append(info.get("score", 0))
        lengths.append(steps)
        rewards.append(total_reward)
    
    env.close()
    return scores, lengths, rewards


def main():
    print("=" * 50)
    print("FEATURE-BASED DQN TEST")
    print("=" * 50)
    
    train_env = DummyVecEnv([make_feature_env])
    
    model = DQN(
        policy="MlpPolicy",
        env=train_env,
        learning_rate=1e-4,
        buffer_size=50000,
        batch_size=32,
        gamma=0.99,
        target_update_interval=5000,
        train_freq=4,
        gradient_steps=1,
        exploration_fraction=0.3,
        exploration_initial_eps=1.0,
        exploration_final_eps=0.05,
        learning_starts=5000,
        seed=42,
        verbose=1,
    )
    
    # Train in chunks and evaluate
    for chunk in range(10):
        steps = 50000
        model.learn(total_timesteps=steps, reset_num_timesteps=False)
        
        total_steps = (chunk + 1) * steps
        scores, lengths, rewards = evaluate(model, num_episodes=50)
        
        print(f"\n[After {total_steps:,} steps] "
              f"score={np.mean(scores):.2f}±{np.std(scores):.2f} "
              f"(max={max(scores)}), "
              f"length={np.mean(lengths):.0f}, "
              f"reward={np.mean(rewards):.2f}")
        
        if np.mean(scores) > 2.0:
            print("Agent is scoring consistently. Feature-based DQN works.")
            break
    
    train_env.close()
    
    final_scores, final_lengths, final_rewards = evaluate(model, num_episodes=100)
    print(f"\n{'='*50}")
    print(f"FINAL EVALUATION (100 episodes)")
    print(f"{'='*50}")
    print(f"Score:  {np.mean(final_scores):.2f} ± {np.std(final_scores):.2f} (max={max(final_scores)})")
    print(f"Length: {np.mean(final_lengths):.0f} ± {np.std(final_lengths):.0f}")
    print(f"Reward: {np.mean(final_rewards):.2f} ± {np.std(final_rewards):.2f}")
    
    if np.mean(final_scores) > 0.5:
        print("\n→ DQN CAN learn this task.")
    else:
        print("\n→ DQN CANNOT learn even with features.")


if __name__ == "__main__":
    main()