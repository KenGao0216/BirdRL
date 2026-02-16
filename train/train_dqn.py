#training script

"""
DQN Training Script for Pixel-Based Flappy Bird

This script trains a DQN agent using Stable-Baselines3 with:
- CNN feature extraction (NatureCNN)
- Frame-stacked grayscale pixel observations
- TensorBoard logging
- Periodic evaluation
- Model checkpointing

Usage:
    python train/train_dqn.py


Monitor training:
    tensorboard --logdir logs/
"""

import argparse, os, sys, time

sys.path.insert(0, '.')

import torch, numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv

from env.wrappers import make_env_fn
from train.callbacks import TensorBoardCallback, EvalCallback, CheckpointCallback


def parse_args():
    parser = argparse.ArgumentParser(description="Train DQN on Flappy Bird")
    
    # Training duration
    parser.add_argument("--total-timesteps", type=int, default=300000,
                        help="Total training steps (default: 300000)")
    
    # Reproducibility
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    
    # Environment
    parser.add_argument("--max-steps", type=int, default=1000,
                        help="Max steps per episode before truncation (default: 1000)")
    parser.add_argument("--n-stack", type=int, default=4,
                        help="Number of frames to stack (default: 4)")
    
    # DQN hyperparameters
    parser.add_argument("--learning-rate", type=float, default=1e-4,
                        help="Learning rate (default: 1e-4)")
    parser.add_argument("--buffer-size", type=int, default=50000,
                        help="Replay buffer size (default: 50000)")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Mini-batch size for gradient updates (default: 32)")
    parser.add_argument("--gamma", type=float, default=0.99,
                        help="Discount factor (default: 0.99)")
    parser.add_argument("--target-update-interval", type=int, default=5000,
                        help="Steps between target network updates (default: 5000)")
    parser.add_argument("--train-freq", type=int, default=4,
                        help="Update the model every N steps (default: 4)")
    parser.add_argument("--gradient-steps", type=int, default=1,
                        help="Gradient steps per update (default: 1)")
    
    # Exploration
    parser.add_argument("--exploration-fraction", type=float, default=0.3,
                        help="Fraction of training for epsilon decay (default: 0.3)")
    parser.add_argument("--exploration-initial-eps", type=float, default=1.0,
                        help="Initial epsilon (default: 1.0)")
    parser.add_argument("--exploration-final-eps", type=float, default=0.05,
                        help="Final epsilon (default: 0.05)")
    
    # Learning starts
    parser.add_argument("--learning-starts", type=int, default=10000,
                        help="Steps before first gradient update (default: 10000)")
    
    # Logging and saving
    parser.add_argument("--eval-freq", type=int, default=10000,
                        help="Steps between evaluations (default: 10000)")
    parser.add_argument("--checkpoint-freq", type=int, default=25000,
                        help="Steps between checkpoints (default: 25000)")
    parser.add_argument("--log-dir", type=str, default="logs",
                        help="TensorBoard log directory (default: logs)")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints",
                        help="Model checkpoint directory (default: checkpoints)")
    
    # Resume from checkpoint
    parser.add_argument("--load-model", type=str, default=None,
                        help="Path to a saved model to resume training from (e.g., checkpoints/best_model)")

    return parser.parse_args()


def main():
    args = parse_args()
    
    # ================================================================
    # SETUP
    # ================================================================
    
    # Create directories
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    # Generate a unique run name
    run_name = f"dqn_flappy_seed{args.seed}_{int(time.time())}"
    log_path = os.path.join(args.log_dir, run_name)
    
    if args.load_model:
        print(f"Resuming from:    {args.load_model}")

    print("=" * 60)
    print("DQN TRAINING - FLAPPY BIRD")
    print("=" * 60)
    print(f"Run name:         {run_name}")
    print(f"Seed:             {args.seed}")
    print(f"Total timesteps:  {args.total_timesteps:,}")
    print(f"Device:           {'cuda' if torch.cuda.is_available() else 'cpu'}")
    print(f"Log directory:    {log_path}")
    print(f"Checkpoint dir:   {args.checkpoint_dir}")
    print()
    
    # ================================================================
    # HYPERPARAMETERS 
    # ================================================================
    
    print("Hyperparameters:")
    print(f"  Learning rate:           {args.learning_rate}")
    print(f"  Buffer size:             {args.buffer_size:,}")
    print(f"  Batch size:              {args.batch_size}")
    print(f"  Gamma (discount):        {args.gamma}")
    print(f"  Target update interval:  {args.target_update_interval:,}")
    print(f"  Train frequency:         every {args.train_freq} steps")
    print(f"  Gradient steps:          {args.gradient_steps}")
    print(f"  Exploration fraction:    {args.exploration_fraction}")
    print(f"  Epsilon:                 {args.exploration_initial_eps} → {args.exploration_final_eps}")
    print(f"  Learning starts:         {args.learning_starts:,}")
    print(f"  Eval frequency:          every {args.eval_freq:,} steps")
    print()
    
    # ================================================================
    # ENVIRONMENT
    # ================================================================
    
    print("Creating environments...")
    
    # Training environment (headless, no rendering to screen)
    train_env = DummyVecEnv([make_env_fn(
        max_steps=args.max_steps,
        n_stack=args.n_stack,
    )])
    
    print(f"  Observation space: {train_env.observation_space}")
    print(f"  Action space:      {train_env.action_space}")
    print()
    
    # ================================================================
    # CREATE OR LOAD DQN MODEL
    # ================================================================
    
    if args.load_model:
        print(f"Loading model from: {args.load_model}")
        model = DQN.load(
            args.load_model,
            env=train_env,
            seed=args.seed,
            device="auto",
            tensorboard_log=log_path,
            # Override these hyperparameters on the loaded model
            learning_rate=args.learning_rate,
            buffer_size=args.buffer_size,
            batch_size=args.batch_size,
            gamma=args.gamma,
            target_update_interval=args.target_update_interval,
            train_freq=args.train_freq,
            gradient_steps=args.gradient_steps,
            exploration_fraction=args.exploration_fraction,
            exploration_initial_eps=args.exploration_initial_eps,
            exploration_final_eps=args.exploration_final_eps,
            learning_starts=0,  # Don't wait — buffer will fill from playing
        )
        print(f"  Model loaded successfully")
        print(f"  Note: Replay buffer starts fresh (old data discarded)")
    else:
        print("Creating new DQN model...")
        model = DQN(
            policy="CnnPolicy",
            env=train_env,
            learning_rate=args.learning_rate,
            buffer_size=args.buffer_size,
            batch_size=args.batch_size,
            gamma=args.gamma,
            target_update_interval=args.target_update_interval,
            train_freq=args.train_freq,
            gradient_steps=args.gradient_steps,
            exploration_fraction=args.exploration_fraction,
            exploration_initial_eps=args.exploration_initial_eps,
            exploration_final_eps=args.exploration_final_eps,
            learning_starts=args.learning_starts,
            tensorboard_log=log_path,
            seed=args.seed,
            device="auto",
            verbose=1,
            optimize_memory_usage=False,
        )
    
    # Print model summary
    print(f"\n  Policy network:")
    print(f"  {model.policy}")
    total_params = sum(p.numel() for p in model.policy.parameters())
    print(f"\n  Total parameters: {total_params:,}")
    print()
    
    # ================================================================
    # CALLBACKS
    # ================================================================
    
    print("Setting up callbacks...")
    
    callbacks = [
        # Log extra metrics to TensorBoard
        TensorBoardCallback(verbose=0),
        
        # Periodic evaluation (separate env, deterministic policy)
        EvalCallback(
            eval_env_fn=make_env_fn(
                max_steps=args.max_steps,
                n_stack=args.n_stack,
            ),
            eval_freq=args.eval_freq,
            n_eval_episodes=10,
            best_model_save_path=args.checkpoint_dir,
            verbose=1,
        ),
        
        # Save checkpoints for training progression analysis
        CheckpointCallback(
            save_freq=args.checkpoint_freq,
            save_path=args.checkpoint_dir,
            name_prefix="dqn_flappy",
            verbose=1,
        ),
    ]
    
    print("   TensorBoard logging")
    print("   Evaluation every {:,} steps".format(args.eval_freq))
    print("   Checkpoints every {:,} steps".format(args.checkpoint_freq))
    print()
    
    # ================================================================
    # TRAIN
    # ================================================================
    
    print("=" * 60)
    print("STARTING TRAINING")
    print("=" * 60)
    print(f"Monitor with: tensorboard --logdir {args.log_dir}")
    print()
    
    start_time = time.time()
    
    try:
        model.learn(
            total_timesteps=args.total_timesteps,
            callback=callbacks,
            log_interval=10,  #Log to console every 10 episodes
            progress_bar=True,  #show progress bar
            reset_num_timesteps=args.load_model is None, #continue counting
        )
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user!")
        print("Saving current model...")
        model.save(os.path.join(args.checkpoint_dir, "interrupted_model"))
        print("Model saved.")
    
    elapsed = time.time() - start_time
    
    # ================================================================
    # SAVE FINAL MODEL
    # ================================================================
    
    final_path = os.path.join(args.checkpoint_dir, "final_model")
    model.save(final_path)
    
    print()
    print("=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"  Total time:     {elapsed/60:.1f} minutes")
    print(f"  Steps/second:   {args.total_timesteps/elapsed:.1f}")
    print(f"  Final model:    {final_path}")
    print(f"  Best model:     {os.path.join(args.checkpoint_dir, 'best_model')}")
    print(f"  TensorBoard:    tensorboard --logdir {args.log_dir}")
    print("=" * 60)
    
    # Cleanup
    train_env.close()


if __name__ == "__main__":
    main()