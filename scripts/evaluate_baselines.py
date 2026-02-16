"""
Evaluate baseline agents on FlappyBird.

Baselines:
1. Random agent - flaps 50% of the time
2. Biased random agent - flaps ~15% of the time (better flap rate)
3. Heuristic agent - flaps when below the gap center

Each is evaluated over multiple episodes with statistics reported.
These numbers are the floor that DQN must beat.

Usage:
    python scripts/evaluate_baselines.py
"""

import sys
sys.path.insert(0, '.')

import numpy as np
from env.flappy_env import FlappyBirdEnv


# =============================================================================
# BASELINE AGENTS
# =============================================================================

class RandomAgent:
    """Flaps with 50% probability each step."""
    
    def __init__(self, env):
        self.env = env
        self.name = "Random (50/50)"
    
    def act(self, obs, info):
        return self.env.action_space.sample()


class BiasedRandomAgent:
    """Flaps with a lower probability (closer to optimal flap rate)."""
    
    def __init__(self, env, flap_prob=0.15):
        self.env = env
        self.flap_prob = flap_prob
        self.name = f"Biased Random (flap={flap_prob:.0%})"
        self.rng = np.random.RandomState(42)
    
    def act(self, obs, info):
        return 1 if self.rng.random() < self.flap_prob else 0


class HeuristicAgent:
    """
    pipe_dy > small_threshold → don't flap.
    """
    
    def __init__(self, env, threshold=0.02):
        self.env = env
        self.threshold = threshold
        self.name = f"Heuristic (threshold={threshold})"
    
    def act(self, obs, info):
        # obs is feature-based: [bird_y_norm, bird_vel_norm, pipe_dx, pipe_dy]
        pipe_dy = obs[3]
        
        # pipe_dy > 0 → gap center is below bird (higher y) → gravity helps → don't flap
        # pipe_dy < 0 → gap center is above bird (lower y) → need to go up → flap
        if pipe_dy < -self.threshold:
            return 1  # Flap to go up toward gap
        else:
            return 0  # Coast / let gravity pull down toward gap


# =============================================================================
# EVALUATION FUNCTION
# =============================================================================

def evaluate_agent(agent, env, num_episodes=100, seed=0):
    """
    Evaluate an agent over multiple episodes.
    
    Returns:
        Dictionary with statistics
    """
    episode_rewards = []
    episode_lengths = []
    episode_scores = []
    
    for ep in range(num_episodes):
        obs, info = env.reset(seed=seed + ep)
        total_reward = 0
        steps = 0
        
        while True:
            action = agent.act(obs, info)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            steps += 1
            
            if terminated or truncated:
                break
        
        episode_rewards.append(total_reward)
        episode_lengths.append(steps)
        episode_scores.append(info.get("score", 0))
    
    return {
        "reward_mean": np.mean(episode_rewards),
        "reward_std": np.std(episode_rewards),
        "length_mean": np.mean(episode_lengths),
        "length_std": np.std(episode_lengths),
        "score_mean": np.mean(episode_scores),
        "score_std": np.std(episode_scores),
        "score_max": np.max(episode_scores),
        "score_min": np.min(episode_scores),
        "episodes": num_episodes,
        "rewards": episode_rewards,
        "lengths": episode_lengths,
        "scores": episode_scores,
    }


# =============================================================================
# MAIN
# =============================================================================

def main():
    num_episodes = 100
    
    print("=" * 60)
    print("BASELINE AGENT EVALUATION")
    print(f"({num_episodes} episodes each)")
    print("=" * 60)
    
    # Use feature-based env for all baselines
    env = FlappyBirdEnv(max_steps=1000)
    
    agents = [
        RandomAgent(env),
        BiasedRandomAgent(env, flap_prob=0.15),
        BiasedRandomAgent(env, flap_prob=0.10),
        HeuristicAgent(env, threshold=0.02),
        HeuristicAgent(env, threshold=0.05),
    ]
    
    results = {}
    
    for agent in agents:
        print(f"\nEvaluating: {agent.name}")
        print("-" * 40)
        
        stats = evaluate_agent(agent, env, num_episodes=num_episodes, seed=0)
        results[agent.name] = stats
        
        print(f"  Score:   {stats['score_mean']:.2f} ± {stats['score_std']:.2f}  "
              f"(min={stats['score_min']}, max={stats['score_max']})")
        print(f"  Reward:  {stats['reward_mean']:.2f} ± {stats['reward_std']:.2f}")
        print(f"  Length:  {stats['length_mean']:.1f} ± {stats['length_std']:.1f}")
    
    # Summary table
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'Agent':<30} {'Score':>10} {'Reward':>12} {'Length':>10}")
    print("-" * 62)
    
    for name, stats in results.items():
        print(f"{name:<30} "
              f"{stats['score_mean']:>5.2f}±{stats['score_std']:<4.2f} "
              f"{stats['reward_mean']:>6.2f}±{stats['reward_std']:<5.2f} "
              f"{stats['length_mean']:>5.1f}±{stats['length_std']:<4.1f}")
    
    print("\n" + "=" * 60)
    print("INTERPRETATION")
    print("=" * 60)
    print("• Random (50/50) is the absolute floor. DQN must beat this.")
    print("• Biased random shows the effect of flap frequency alone.")
    print("• Heuristic shows what a simple rule can achieve.")
    print("• DQN should significantly exceed the heuristic baseline.")
    print("• If DQN = random after training, something is broken.")
    print("=" * 60)
    
    env.close()
    return results


if __name__ == "__main__":
    results = main()