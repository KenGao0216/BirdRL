"""
Custom callbacks for DQN training.
callbacks are hooks that SB3 calls at various points during training. they are used for logging, saving checkpoints, evaluation etc.

Callbacks:
- EvalCallback: Periodic evaluation with logging
- CheckpointCallback: Save model at regular intervals
- TensorBoardCallback: Log extra metrics (Q-values, epsilon, etc.)
"""

import os
import sys
import numpy as np
from typing import Optional, Dict, Any

sys.path.insert(0, '.')

from stable_baselines3.common.callbacks import BaseCallback, EventCallback
from stable_baselines3.common.vec_env import DummyVecEnv


class TensorBoardCallback(BaseCallback):
    """
    Logs additional training metrics to TensorBoard.
    
    SB3 logs episode reward/length automatically, but we want:
    - Current epsilon value
    - Score (pipes passed) per episode
    - Mean Q-values (when available)
    """
    
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self._episode_scores = []
    
    def _on_step(self) -> bool:
        """Called after every environment step."""
        
        # Log epsilon (exploration rate)
        # SB3 DQN stores this in the exploration_rate attribute
        if hasattr(self.model, 'exploration_rate'):
            self.logger.record("rollout/epsilon", self.model.exploration_rate)
        
        # Collect score from info dict => this is meta data for debugging and logging, not for learning
        """
            {
                "score": xx
                "steps": xx
                "bird_y":xx
                "bird_vel":xx
                "num_pipes": xx
            }
        """
        # SB3 wraps infos in a list (one per vectorized env)
        infos = self.locals.get("infos", [])
        for info in infos:
            # SB3 stores the final info of a completed episode in "episode" key
            # when using Monitor wrapper, or we can check for terminal info
            if "score" in info:
                episode_info = info
                
                # Check if episode just ended
                # SB3 puts terminal observation in info when episode ends
                if self.locals.get("dones", [False])[0]:
                    self._episode_scores.append(info["score"])
                    self.logger.record("rollout/score", info["score"])
        
        return True  # Return False to stop training
    
    def _on_rollout_end(self) -> None:
        """Called at the end of a rollout (every 10 episodes)."""
        if self._episode_scores:
            self.logger.record("rollout/mean_score", np.mean(self._episode_scores[-100:]))
            self._episode_scores = []  # Reset for next rollout


class EvalCallback(BaseCallback):
    """
    Periodically evaluate the agent and log results.
    
    Runs N evaluation episodes (no exploration / epsilon=0)
    every `eval_freq` steps and logs:
    - Mean/std evaluation reward
    - Mean/std evaluation score
    - Mean/std episode length
    - Saves best model
    """
    
    def __init__(
        self,
        eval_env_fn,
        eval_freq: int = 10000,
        n_eval_episodes: int = 10,
        best_model_save_path: str = "checkpoints", # save the best model in checkpoints folder
        verbose: int = 1, # whether the callback prints status msgs to console during training 
    ):
        super().__init__(verbose)
        self.eval_env_fn = eval_env_fn
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.best_model_save_path = best_model_save_path
        self.best_mean_reward = -np.inf
        self.eval_env = None
    
    def _init_callback(self) -> None:
        """Create eval environment and ensure save directory exists."""
        self.eval_env = DummyVecEnv([self.eval_env_fn])
        os.makedirs(self.best_model_save_path, exist_ok=True)
    
    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq == 0:
            self._run_evaluation()
        return True
    
    def _run_evaluation(self) -> None:
        """Run evaluation episodes and log results."""
        episode_rewards = []
        episode_lengths = []
        episode_scores = []
        
        for ep in range(self.n_eval_episodes):
            obs = self.eval_env.reset()
            total_reward = 0
            steps = 0
            done = False
            
            while not done:
                # Deterministic=True means no epsilon exploration
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, done, info = self.eval_env.step(action)
                total_reward += reward[0]
                steps += 1
            
            episode_rewards.append(total_reward)
            episode_lengths.append(steps)
            
            # Extract score from final info
            if "score" in info[0]:
                episode_scores.append(info[0]["score"])
            # SB3 VecEnv may store terminal info differently
            elif "terminal_info" in info[0] and "score" in info[0]["terminal_info"]:
                episode_scores.append(info[0]["terminal_info"]["score"])
        
        mean_reward = np.mean(episode_rewards)
        mean_length = np.mean(episode_lengths)
        mean_score = np.mean(episode_scores) if episode_scores else 0
        
        # Log to TensorBoard
        self.logger.record("eval/mean_reward", mean_reward)
        self.logger.record("eval/std_reward", np.std(episode_rewards))
        self.logger.record("eval/mean_length", mean_length)
        self.logger.record("eval/mean_score", mean_score)
        self.logger.record("eval/max_score", max(episode_scores) if episode_scores else 0)
        
        if self.verbose:
            print(f"\n[Eval @ step {self.num_timesteps}] "
                  f"reward={mean_reward:.2f}±{np.std(episode_rewards):.2f}, "
                  f"length={mean_length:.1f}, "
                  f"score={mean_score:.2f}")
        
        # Save best model
        if mean_reward > self.best_mean_reward:
            self.best_mean_reward = mean_reward
            save_path = os.path.join(self.best_model_save_path, "best_model")
            self.model.save(save_path)
            if self.verbose:
                print(f"  New best model saved! (reward={mean_reward:.2f})")
    
    def _on_training_end(self) -> None:
        """Clean up eval environment."""
        if self.eval_env is not None:
            self.eval_env.close()


class CheckpointCallback(BaseCallback):
    """
    Save model checkpoint at regular intervals.
    
    Useful for:
    - Recovering from crashes
    - Comparing agent behavior at different training stages
    - Creating training progression videos
    """
    
    def __init__(
        self,
        save_freq: int = 25000,
        save_path: str = "checkpoints",
        name_prefix: str = "dqn_flappy",
        verbose: int = 1,
    ):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path
        self.name_prefix = name_prefix
    
    def _init_callback(self) -> None:
        os.makedirs(self.save_path, exist_ok=True)
    
    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq == 0:
            path = os.path.join(
                self.save_path,
                f"{self.name_prefix}_{self.num_timesteps}_steps"
            )
            self.model.save(path)
            if self.verbose:
                print(f"  [Checkpoint] Saved model to {path}")
        return True