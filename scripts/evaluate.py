"""
Evaluate trained DQN agent

Modes:
1.Headless evaluation:Run N episodes, report statistics
2.Real-time playback:Watch agent play in a Pygame window

Usage:
    # Headless evaluation
    python scripts/evaluate.py --model checkpoints/best_model --episodes 100

    # Watch the agent play (fps = 30)
    python scripts/evaluate.py --model checkpoints/best_model --render

    # Compare a checkpoint from mid-training
    python scripts/evaluate.py --model checkpoints/dqn_flappy_100000_steps --episodes 50
"""

import argparse
import sys
import time

sys.path.insert(0, '.')

import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv

from env.flappy_env import FlappyBirdEnv
from env.wrappers import PixelObservationWrapper, FrameStackWrapper


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate trained DQN agent")
    
    parser.add_argument("--model", type=str, required=True,
                        help="Path to saved model (e.g., checkpoints/best_model)")
    parser.add_argument("--episodes", type=int, default=100,
                        help="Number of evaluation episodes (default: 100)")
    parser.add_argument("--render", action="store_true",
                        help="Render agent playing in real time")
    parser.add_argument("--fps", type=int, default=30,
                        help="Playback FPS when rendering (default: 30)")
    parser.add_argument("--seed", type=int, default=0,
                        help="Starting seed for evaluation (default: 0)")
    parser.add_argument("--n-stack", type=int, default=4,
                        help="Frame stack size (must match training, default: 4)")
    parser.add_argument("--max-steps", type=int, default=2000,
                        help="Max steps per episode (default: 2000)")
    parser.add_argument("--deterministic", action="store_true", default=True,
                        help="Use deterministic actions (default: True)")
    parser.add_argument("--stochastic", action="store_true",
                        help="Use stochastic actions (with epsilon)")
    
    return parser.parse_args()


def create_eval_env(render: bool, n_stack: int, max_steps: int, fps: int):
    """
    Create an evaluation environment.
    
    For rendering: We need rgb_array (for pixel observations fed to model) and human display
    - Using render_mode="rgb_array" for the model's observations
    - Manually blitting to a pygame window for human viewing
    
    For headless: use rgb_array.
    """
    if render:
        #For real-time viewing, use rgb_array for the model and create a separate display window
        base_env = FlappyBirdEnv(render_mode="rgb_array", max_steps=max_steps)
    else:
        base_env = FlappyBirdEnv(render_mode="rgb_array", max_steps=max_steps)
    
    pixel_env = PixelObservationWrapper(base_env)
    stacked_env = FrameStackWrapper(pixel_env, n_stack=n_stack)
    
    return stacked_env, base_env


def evaluate_headless(model, args):
    """Run headless evaluation and report statistics."""
    
    print("=" * 60)
    print("HEADLESS EVALUATION")
    print(f"Model: {args.model}")
    print(f"Episodes: {args.episodes}")
    print(f"Deterministic: {not args.stochastic}")
    print("=" * 60)
    
    env, base_env = create_eval_env(
        render=False, n_stack=args.n_stack,
        max_steps=args.max_steps, fps=args.fps
    )
    
    deterministic = not args.stochastic
    
    episode_rewards = []
    episode_lengths = []
    episode_scores = []
    
    for ep in range(args.episodes):
        obs, info = env.reset(seed=args.seed + ep)
        total_reward = 0
        steps = 0
        
        while True:
            # Model expects VecEnv format: add batch dimension
            action, _ = model.predict(
                obs[np.newaxis, ...],  # (4,84,84) → (1,4,84,84)
                deterministic=deterministic
            )
            action = action.item()  # Extract scalar from array
            
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            steps += 1
            
            if terminated or truncated:
                break
        
        episode_rewards.append(total_reward)
        episode_lengths.append(steps)
        episode_scores.append(info.get("score", 0))
        
        if (ep + 1) % 10 == 0:
            print(f"  Episodes {ep+1}/{args.episodes}: "
                  f"mean_score={np.mean(episode_scores):.2f}")
    
    # Report results
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"  Episodes:      {args.episodes}")
    print(f"  Reward:        {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
    print(f"  Score:         {np.mean(episode_scores):.2f} ± {np.std(episode_scores):.2f}")
    print(f"  Score range:   [{min(episode_scores)}, {max(episode_scores)}]")
    print(f"  Length:        {np.mean(episode_lengths):.1f} ± {np.std(episode_lengths):.1f}")
    print()
    
    print("COMPARISON WITH BASELINES:")
    print(f"  Random agent:     score=0.00, length=38.0,  reward=2.70")
    print(f"  Biased random:    score=0.01, length=75.9,  reward=6.50")
    print(f"  Heuristic:        score=0.34, length=127.9, reward=12.03")
    print(f"  THIS AGENT:       score={np.mean(episode_scores):.2f}, "
          f"length={np.mean(episode_lengths):.1f}, "
          f"reward={np.mean(episode_rewards):.2f}")
    
    if np.mean(episode_scores) > 0.34:
        print("\n  Agent BEATS the heuristic baseline.")
    elif np.mean(episode_rewards) > 2.70:
        print("\n  Agent beats random but not heuristic yet.")
    else:
        print("\n  Agent is at or below random baseline. More training needed.")
    
    print("=" * 60)
    
    env.close()
    return episode_rewards, episode_lengths, episode_scores


def evaluate_rendered(model, args):
    """Watch agent play in real time."""
    
    import pygame
    
    print("=" * 60)
    print("REAL-TIME PLAYBACK")
    print(f"Model: {args.model}")
    print(f"FPS: {args.fps}")
    print(f"Press Ctrl+C or close window to stop")
    print("=" * 60)
    
    env, base_env = create_eval_env(
        render=True, n_stack=args.n_stack,
        max_steps=args.max_steps, fps=args.fps
    )
    
    deterministic = not args.stochastic
    
    #Create display window for viewing
    pygame.init()
    screen = pygame.display.set_mode(
        (base_env.SCREEN_WIDTH, base_env.SCREEN_HEIGHT)
    )
    pygame.display.set_caption(f"Flappy Bird DQN - Evaluation ({args.fps} FPS)")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("Arial", 18, bold=True)
    
    episode = 0
    running = True
    
    while running and episode < args.episodes:
        obs, info = env.reset(seed=args.seed + episode)
        total_reward = 0
        steps = 0
        episode += 1
        
        print(f"\nEpisode {episode}:")
        
        while running:
            #handle pygame events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    break
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                        break
            
            if not running:
                break
            
            #Get action from model
            action, _ = model.predict(
                obs[np.newaxis, ...],
                deterministic=deterministic
            )
            action = action.item()
            
            # Step environment
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            steps += 1
            
            # Get the RGB frame from the base environment for display
            frame = base_env.render()
            if frame is not None:
                # Convert numpy array to pygame surface
                # frame is (H, W, 3) RGB — pygame needs (W, H, 3) for surfarray
                surf = pygame.surfarray.make_surface(frame.transpose(1, 0, 2))
                screen.blit(surf, (0, 0))
                
                # Overlay info
                action_str = "FLAP" if action == 1 else "----"
                info_text = f"Score: {info.get('score', 0)}  Steps: {steps}  Action: {action_str}"
                text_surf = font.render(info_text, True, (255, 255, 255))
                shadow_surf = font.render(info_text, True, (0, 0, 0))
                screen.blit(shadow_surf, (12, 12))
                screen.blit(text_surf, (10, 10))
                
                pygame.display.flip()
                clock.tick(args.fps)
            
            if terminated or truncated:
                print(f"  score={info.get('score', 0)}, steps={steps}, "
                      f"reward={total_reward:.1f}, "
                      f"{'DIED' if terminated else 'TRUNCATED'}")
                
                time.sleep(0.5)
                break
    
    pygame.quit()
    env.close()
    print("\nPlayback ended.")


def main():
    args = parse_args()
    
    #load the trained model
    print(f"Loading model from: {args.model}")
    try:
        model = DQN.load(args.model)
        print("  Model loaded successfully")
    except Exception as e:
        print(f"  Failed to load model: {e}")
        sys.exit(1)
    
    if args.render:
        evaluate_rendered(model, args)
    else:
        evaluate_headless(model, args)


if __name__ == "__main__":
    main()