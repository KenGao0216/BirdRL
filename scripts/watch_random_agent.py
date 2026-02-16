"""
Watch an agent play Flappy Bird.

This is a visual sanity check:
-Does the bird fall with gravity?
-Does flapping move the bird up?
-Do pipes scroll left?
-Does collision detection work?
-Does the score increment?
"""

import sys
sys.path.insert(0, '.')

from env.flappy_env import FlappyBirdEnv


def main():
    env = FlappyBirdEnv(render_mode="human")
    
    num_episodes = 5
    
    for episode in range(num_episodes):
        obs, info = env.reset(seed=episode)
        total_reward = 0
        steps = 0
        flap_count = 0
        
        while True:
            action = env.action_space.sample()
            
            if action == 1:
                flap_count += 1
            
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            steps += 1
            
            # Print every step: action, bird position, velocity
            action_str = "FLAP" if action == 1 else "----"
            print(f"  Step {steps:3d}: {action_str}  "
                  f"bird_y={info['bird_y']:6.1f}  "
                  f"vel={info['bird_vel']:+5.1f}  "
                  f"reward={reward:+.1f}")
            
            env.render()
            
            if terminated or truncated:
                flap_pct = (flap_count / steps) * 100
                print(f"\nEpisode {episode + 1}: "
                      f"score={info['score']}, "
                      f"steps={steps}, "
                      f"reward={total_reward:.1f}, "
                      f"flaps={flap_count}/{steps} ({flap_pct:.0f}%), "
                      f"{'DIED' if terminated else 'TRUNCATED'}\n")
                break
    
    env.close()
    print("\nDone. Close the window or press Ctrl+C.")

if __name__ == "__main__":
    main()


