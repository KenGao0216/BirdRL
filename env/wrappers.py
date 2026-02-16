#frame stacking, preprocessing 

"""
Observation wrappers for FlappyBirdEnv.

These wrappers transform the base environment's observations into suitable format for CNN-based agents.

Wrapper stack for pixel-based DQN:
    FlappyBirdEnv(render_mode="rgb_array")
    -> PixelObservationWrapper (grayscale, resize, normalize)
    -> FrameStackWrapper (stack 4 frames) 
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import cv2
from typing import Optional, Tuple, Dict, Any


class PixelObservationWrapper(gym.ObservationWrapper):
    """
    Converts the environment to use pixel-based observations.
    
    Pipeline:
        1. Calls env.render() to get RGB frame (H, W, 3)
        2. Converts to grayscale (H, W)
        3. Resizes to (84, 84)
        4. Adds channel dimension → (1, 84, 84) [channels-first for PyTorch]
        5. Returns as uint8 [0, 255]
    
    The wrapped environment must have render_mode="rgb_array".
    
    Observation space: Box(0, 255, shape=(1, 84, 84), dtype=uint8)
    """
    
    def __init__(
        self,
        env: gym.Env,
        width: int = 84,
        height: int = 84,
    ):
        """
        Args:
            env: Base environment (must have render_mode="rgb_array")
            width: Target observation width
            height: Target observation height
        """
        super().__init__(env)
        
        self.width = width
        self.height = height
        
        # Verify the base environment can produce pixel frames
        assert env.render_mode == "rgb_array", (
            f"PixelObservationWrapper requires render_mode='rgb_array', "
            f"got '{env.render_mode}'"
        )
        
        # Override observation space for pixel observations
        # Shape: (1, height, width) — channels-first for PyTorch/SB3
        # dtype: uint8 — saves memory in replay buffer
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(1, self.height, self.width),
            dtype=np.uint8,
        )
    
    def observation(self, obs: np.ndarray) -> np.ndarray:
        """
        Transform the base observation into a pixel observation.
        
        This method is called automatically by gym.ObservationWrapper
        on every reset() and step(). The `obs` argument is the original
        feature-based observation, which is ignored — use the rendered
        frame instead.
        
        Args:
            obs: Original observation from base env (ignored)
        
        Returns:
            Pixel observation of shape (1, 84, 84), dtype uint8
        """
        return self._get_pixel_observation()
    
    def _get_pixel_observation(self) -> np.ndarray:
        """
        Capture and preprocess the current frame.
        
        Returns:
            Preprocessed frame of shape (1, 84, 84), dtype uint8
        """
        # 1. Get RGB frame from renderer
        frame = self.env.render()
        
        assert frame is not None, (
            "render() returned None. Is render_mode='rgb_array'?"
        )
        # frame shape: (SCREEN_HEIGHT, SCREEN_WIDTH, 3), dtype uint8
        
        # 2. Convert RGB to grayscale
        # cv2.cvtColor expects (H, W, 3) uint8 — which is what we have
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        # gray shape: (SCREEN_HEIGHT, SCREEN_WIDTH), dtype uint8
        
        # 3. Resize to target dimensions
        # cv2.INTER_AREA is best for downscaling (anti-aliased)
        resized = cv2.resize(
            gray,
            (self.width, self.height),
            interpolation=cv2.INTER_AREA,
        )
        # resized shape: (84, 84), dtype uint8
        
        # 4. Add channel dimension (channels-first for PyTorch)
        # (84, 84) → (1, 84, 84)
        pixel_obs = resized[np.newaxis, :, :]
        
        # 5. Ensure correct dtype
        pixel_obs = pixel_obs.astype(np.uint8)
        
        return pixel_obs

class FrameStackWrapper(gym.Wrapper):
    """
    Stack the last N observations along the channel dimension.
    
    Converts observations from shape (1, H, W) to (N, H, W) by
    maintaining a buffer of the last N frames.
    
    On reset, the initial frame is repeated N times.
    
    This is equivalent to SB3's VecFrameStack but works with
    single (non-vectorized) environments.
    """
    
    def __init__(self, env: gym.Env, n_stack: int = 4):
        """
        Args:
            env: Wrapped environment (must produce observations with a channel dim)
            n_stack: Number of frames to stack
        """
        super().__init__(env)
        
        self.n_stack = n_stack
        
        # Get the shape of a single observation from the wrapped env
        # Expected: (1, 84, 84)
        old_space = env.observation_space
        assert len(old_space.shape) == 3, (
            f"Expected 3D observation (C, H, W), got shape {old_space.shape}"
        )
        
        single_channels = old_space.shape[0]  # Should be 1
        height = old_space.shape[1]
        width = old_space.shape[2]
        
        # New observation space: (n_stack * channels, H, W)
        # For us: (4 * 1, 84, 84) = (4, 84, 84)
        self.observation_space = spaces.Box(
            low=np.repeat(old_space.low, n_stack, axis=0),
            high=np.repeat(old_space.high, n_stack, axis=0),
            dtype=old_space.dtype,
        )
        
        # Frame buffer: stores the last n_stack frames
        # Initialize as zeros; filled properly on reset()
        self.frames = np.zeros(
            (n_stack * single_channels, height, width),
            dtype=old_space.dtype,
        )
    
    def reset(self, **kwargs) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reset environment and fill frame stack with initial observation.
        
        The first frame is repeated n_stack times so the agent
        doesn't see a "blank history."
        """
        obs, info = self.env.reset(**kwargs)
        
        # Fill entire buffer with the first frame (repeated)
        # obs shape: (1, 84, 84)
        for i in range(self.n_stack):
            self.frames[i * obs.shape[0]:(i + 1) * obs.shape[0]] = obs
        
        return self.frames.copy(), info #.copy is important, prevents mutation (for replay buffer safety)
    
    def step(self, action) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Take a step and update the frame stack.
        
        Shifts the buffer: drops oldest frame, appends newest.
        """
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # Shift frames left: drop oldest, append newest
        # For single-channel obs (1, 84, 84):
        #   frames[0] = old frame t-3  → dropped
        #   frames[1] = old frame t-2  → becomes frames[0]
        #   frames[2] = old frame t-1  → becomes frames[1]
        #   frames[3] = old frame t    → becomes frames[2]
        #   new obs                    → becomes frames[3]
        channels = obs.shape[0]  # 1 for grayscale
        self.frames[:-channels] = self.frames[channels:]  # Shift left
        self.frames[-channels:] = obs                      # Insert newest
        
        return self.frames.copy(), reward, terminated, truncated, info #.copy is important


def make_pixel_env(
    render_mode: str = "rgb_array",
    max_steps: int = 1000,
    n_stack: int = 4,
    **kwargs,
) -> gym.Env:
    """
    Factory function to create a fully-wrapped FlappyBird environment.
    
    Wrapper stack:
        FlappyBirdEnv(render_mode="rgb_array")
        → PixelObservationWrapper (grayscale 84x84)
        → FrameStackWrapper (stack n_stack frames)
    
    Args:
        render_mode: Must be "rgb_array" for pixel observations
        max_steps: Maximum steps before truncation
        n_stack: Number of frames to stack (default: 4)
    
    Returns:
        Fully wrapped environment with shape (n_stack, 84, 84)
    """
    from env.flappy_env import FlappyBirdEnv
    
    base_env = FlappyBirdEnv(render_mode=render_mode, max_steps=max_steps, **kwargs)
    pixel_env = PixelObservationWrapper(base_env)
    stacked_env = FrameStackWrapper(pixel_env, n_stack=n_stack)
    
    return stacked_env

def make_env_fn(max_steps: int = 1000, n_stack: int = 4):
    """
    Returns a callable that creates a wrapped environment.
    
    SB3's DummyVecEnv requires a list of callables (env factories),
    not pre-built environments. This function provides that.
    
    Usage:
        from stable_baselines3.common.vec_env import DummyVecEnv
        env = DummyVecEnv([make_env_fn(max_steps=1000)])
    
    Args:
        max_steps: Maximum steps before truncation
        n_stack: Number of frames to stack
    
    Returns:
        Callable that creates a fully-wrapped environment
    """
    def _init():
        return make_pixel_env(
            render_mode="rgb_array",
            max_steps=max_steps,
            n_stack=n_stack,
        )
    return _init