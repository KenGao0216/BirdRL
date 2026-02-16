"""
Record video of a trained DQN agent playing Flappy Bird.

Records RGB frames from the environment and saves as MP4.

Usage:
    # Record best model, 3 episodes
    python scripts/record_video.py --model checkpoints/best_model --episodes 3

    # Record a specific checkpoint
    python scripts/record_video.py --model checkpoints/dqn_flappy_100000_steps --episodes 1

    # Custom output path and FPS
    python scripts/record_video.py --model checkpoints/best_model --output videos/my_run.mp4 --fps 30

    # Record with colorful rendering (for presentation)
    python scripts/record_video.py --model checkpoints/best_model --colorful
"""

import argparse
import sys
import os

sys.path.insert(0, '.')

import numpy as np
import imageio
from stable_baselines3 import DQN

from env.flappy_env import FlappyBirdEnv
from env.wrappers import PixelObservationWrapper, FrameStackWrapper


def parse_args():
    parser = argparse.ArgumentParser(description="Record agent gameplay video")
    
    parser.add_argument("--model", type=str, required=True,
                        help="Path to saved model")
    parser.add_argument("--episodes", type=int, default=3,
                        help="Number of episodes to record (default: 3)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output video path (default: auto-generated)")
    parser.add_argument("--fps", type=int, default=30,
                        help="Video FPS (default: 30)")
    parser.add_argument("--seed", type=int, default=0,
                        help="Starting seed (default: 0)")
    parser.add_argument("--n-stack", type=int, default=4,
                        help="Frame stack size (must match training, default: 4)")
    parser.add_argument("--max-steps", type=int, default=2000,
                        help="Max steps per episode (default: 2000)")
    parser.add_argument("--colorful", action="store_true",
                        help="Use colorful rendering instead of high-contrast")
    
    return parser.parse_args()


def create_recording_env(n_stack: int, max_steps: int):
    """
    Create environment for recording.
    Returns both the wrapped env (for model input) and the base env (for frame capture).
    """
    base_env = FlappyBirdEnv(render_mode="rgb_array", max_steps=max_steps)
    pixel_env = PixelObservationWrapper(base_env)
    stacked_env = FrameStackWrapper(pixel_env, n_stack=n_stack)
    return stacked_env, base_env


def add_text_to_frame(frame, text, position=(10, 20)):
    """
    Add text overlay to a frame using simple pixel drawing.
    Uses OpenCV if available, otherwise skips text.
    """
    try:
        import cv2
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.5
        thickness = 1
        
        # Shadow
        cv2.putText(frame_bgr, text, (position[0]+1, position[1]+1),
                    font, scale, (0, 0, 0), thickness + 1)
        # Text
        cv2.putText(frame_bgr, text, position,
                    font, scale, (255, 255, 255), thickness)
        
        return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    except ImportError:
        return frame


def record_episodes(model, args):
    """Record episodes and save as video."""
    
    env, base_env = create_recording_env(
        n_stack=args.n_stack,
        max_steps=args.max_steps,
    )
    
    # Generate output path if not specified
    if args.output is None:
        model_name = os.path.basename(args.model)
        os.makedirs("videos", exist_ok=True)
        args.output = f"videos/{model_name}_{args.episodes}ep.mp4"
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else ".", exist_ok=True)
    
    print(f"Recording to: {args.output}")
    print(f"FPS: {args.fps}")
    print(f"Episodes: {args.episodes}")
    print()
    
    all_frames = []
    episode_stats = []
    
    for ep in range(args.episodes):
        obs, info = env.reset(seed=args.seed + ep)
        total_reward = 0
        steps = 0
        episode_frames = []
        
        while True:
            # Get action from model
            action, _ = model.predict(
                obs[np.newaxis, ...],
                deterministic=True
            )
            action = action.item()
            
            # Step
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            steps += 1
            
            # Capture RGB frame from base env
            frame = base_env.render()
            if frame is not None:
                # Add overlay text
                score = info.get("score", 0)
                action_str = "FLAP" if action == 1 else ""
                overlay = f"Ep:{ep+1}  Score:{score}  Step:{steps}  {action_str}"
                frame = add_text_to_frame(frame, overlay)
                episode_frames.append(frame)
            
            if terminated or truncated:
                # Add a few freeze frames at the end so you can see the death
                for _ in range(args.fps // 2):  # Half second freeze
                    if frame is not None:
                        end_text = f"Ep:{ep+1}  FINAL Score:{info.get('score', 0)}  Steps:{steps}"
                        end_frame = add_text_to_frame(base_env.render() or frame, end_text)
                        episode_frames.append(end_frame)
                break
        
        all_frames.extend(episode_frames)
        
        stat = {
            "episode": ep + 1,
            "score": info.get("score", 0),
            "steps": steps,
            "reward": total_reward,
            "frames": len(episode_frames),
        }
        episode_stats.append(stat)
        
        print(f"  Episode {ep+1}: score={stat['score']}, "
              f"steps={steps}, reward={total_reward:.1f}, "
              f"frames={len(episode_frames)}")
    
    # Write video
    print(f"\nWriting {len(all_frames)} frames to {args.output}...")
    
    writer = imageio.get_writer(
        args.output,
        fps=args.fps,
        quality=8,  # 0-10, higher = better quality, larger file
    )
    
    for frame in all_frames:
        writer.append_data(frame)
    
    writer.close()
    
    # Summary
    file_size = os.path.getsize(args.output) / (1024 * 1024)
    duration = len(all_frames) / args.fps
    
    print(f"\n{'='*50}")
    print(f"VIDEO RECORDED")
    print(f"{'='*50}")
    print(f"  File:      {args.output}")
    print(f"  Size:      {file_size:.1f} MB")
    print(f"  Duration:  {duration:.1f} seconds")
    print(f"  Frames:    {len(all_frames)}")
    print(f"  FPS:       {args.fps}")
    print()
    print(f"  Episode Summary:")
    for stat in episode_stats:
        print(f"    Ep {stat['episode']}: score={stat['score']}, "
              f"steps={stat['steps']}, reward={stat['reward']:.1f}")
    
    mean_score = np.mean([s["score"] for s in episode_stats])
    mean_steps = np.mean([s["steps"] for s in episode_stats])
    print(f"\n  Mean score:  {mean_score:.2f}")
    print(f"  Mean steps:  {mean_steps:.0f}")
    print(f"{'='*50}")
    
    env.close()


def main():
    args = parse_args()
    
    # Load model
    print(f"Loading model: {args.model}")
    try:
        model = DQN.load(args.model)
        print("  Model loaded")
    except Exception as e:
        print(f"  Failed: {e}")
        sys.exit(1)
    
    print()
    record_episodes(model, args)


if __name__ == "__main__":
    main()