#main env

from doctest import FAIL_FAST
import gymnasium as gym 
from gymnasium import spaces
import numpy as np 
from typing import Optional, Tuple, Dict, Any
import pygame

class FlappyBirdEnv(gym.Env):
    """
    Observation Space (Feature-Based):
        Box(4,) containing:
        - bird_y: Normalized bird y position [0, 1]
        - bird_vel: Normalized bird velocity [-1, 1]  
        - pipe_dx: Normalized horizontal distance to next pipe [0, 1]
        - pipe_dy: Normalized vertical distance to gap center [-1, 1]

    Reward:
        - +1.0 for passing a pipe
        - +0.1 for each step survived
        - -1.0 for dying (hitting obstacle or boundary)

    Episode Termination:
        - Bird collides with pipe
        - Bird goes above screen (y < 0)
        - Bird falls below screen (y > SCREEN_HEIGHT)
    
    Episode Truncation:
        - Episode exceeds max_steps (default: 1000)    
    """
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps":30}

    #physics 
    SCREEN_WIDTH=288
    SCREEN_HEIGHT=512
    BIRD_X=50
    BIRD_SIZE=20 #hitbox
    GRAVITY=0.5
    FLAP_VELOCITY=-8
    MAX_VELOCITY=10
    PIPE_SPEED=3
    PIPE_WIDTH=52
    PIPE_GAP=150
    PIPE_SPACING=200
    # High-contrast colors for CNN visibility
    COLOR_SKY = (0, 0, 0)            
    COLOR_BIRD = (255, 255, 255)     
    COLOR_PIPE = (128, 128, 128)     
    COLOR_PIPE_BORDER = (128, 128, 128)  
    COLOR_GROUND = (64, 64, 64)      
    COLOR_TEXT = (255, 255, 255)     
    COLOR_TEXT_SHADOW = (128, 128, 128)

    def __init__(self, render_mode: Optional[str] = None, max_steps: int=1000):
        #initialize env: 
        #render_mode: human for pygame window, rgb_array for pixel array, None for no rendering
        #max_steps: max steps before truncation
        super().__init__()
        self.render_mode = render_mode
        self.max_steps = max_steps

        self.action_space = spaces.Discrete(2)

        self.observation_space = spaces.Box(
            low=np.array([0.0, -1.0, 0.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32
        )

        self.screen = None
        self.clock = None

        self.bird_y: float = 0.0
        self.bird_vel: float = 0.0
        self.pipes: list = []
        self.score: int = 0
        self.steps: int = 0
        self.pipes_passed: set = set()
        self.pipe_id_counter: int = 0

    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        reset env to initial state 

        seed: random seeed for reproducibility 
        options: not unused yet
        returns observation: initial observation 
        returns info: additional info dict 
        """
        super().reset(seed=seed)
        self.bird_y = self.SCREEN_HEIGHT / 2
        self.bird_vel = 0.0
        self.pipes = []
        self._spawn_pipe(self.SCREEN_WIDTH+100) #first pipe appears at right edge
        self.score = 0
        self.steps = 0
        self.pipes_passed = set()
        self.pipe_id_counter = 0
        observation = self._get_observation()
        info = self._get_info()

        return observation, info
    

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
            execute one env step 
            action: 0 (do nothing) or 1 (flap)

            returns: 
                observation: new observation after action 
                reward: reward for this step
                terminated
                truncated
                info
        """

        assert self.action_space.contains(action), f"Invalid action: {action}"

        self.steps+=1
        reward = 0.0
        terminated = False
        truncated = False

        if action == 1:
            self.bird_vel = self.FLAP_VELOCITY

        self.bird_vel +=self.GRAVITY
        self.bird_vel = np.clip(self.bird_vel, -self.MAX_VELOCITY, self.MAX_VELOCITY)

        self.bird_y+=self.bird_vel

        for pipe in self.pipes: #update all pipes
            pipe["x"]-=self.PIPE_SPEED
        
        self.pipes = [p for p in self.pipes if p["x"] + self.PIPE_WIDTH > 0] #filter out pipes that have gone off screeen

        #spawn new pipe if needed
        if len(self.pipes) == 0 or self.pipes[-1]["x"] < self.SCREEN_WIDTH - self.PIPE_SPACING: 
            self._spawn_pipe(self.SCREEN_WIDTH)

        if self._check_collision(): #terminated state
            terminated = True
            reward = -1.0
        else: #checking for passing pipes + updating score and reward
            for pipe in self.pipes:
                pipe_right = pipe["x"] + self.PIPE_WIDTH
                bird_right = self.BIRD_X + self.BIRD_SIZE

                if bird_right > pipe_right and pipe["id"] not in self.pipes_passed:
                    self.pipes_passed.add(pipe["id"])
                    self.score += 1
                    reward += 5.0

            reward+=0.02 #small reward for surviving 


        if self.steps >= self.max_steps: #truncated
            truncated = True

        observation = self._get_observation()
        info = self._get_info()

        return observation, reward, terminated, truncated, info

    def _spawn_pipe(self, x:float) -> None: 
        #spawn new pipe, gap is randomized 

        min_gap_y = self.PIPE_GAP // 2 + 50
        max_gap_y = self.SCREEN_HEIGHT - self.PIPE_GAP // 2 - 50
        gap_y = self.np_random.integers(min_gap_y, max_gap_y)
        self.pipes.append({"x": x, "gap_y": gap_y, "id": self.pipe_id_counter})
        self.pipe_id_counter += 1
    
    def _check_collision(self) -> bool: 
        #return true if collision detected, false otherwise 

        bird_top = self.bird_y
        bird_bottom = self.bird_y + self.BIRD_SIZE
        bird_left = self.BIRD_X
        bird_right = self.BIRD_X + self.BIRD_SIZE

        if bird_top < 0 or bird_bottom > self.SCREEN_HEIGHT: 
            return True
        
        for pipe in self.pipes: 
            pipe_left = pipe["x"]
            pipe_right = pipe["x"] + self.PIPE_WIDTH
            gap_top = pipe["gap_y"] - self.PIPE_GAP // 2
            gap_bottom = pipe["gap_y"] + self.PIPE_GAP // 2

            if bird_right > pipe_left and bird_left < pipe_right: 
                if bird_top < gap_top or bird_bottom > gap_bottom: 
                    return True

        return False
    
    def _get_observation(self) -> np.ndarray: 
        #compute current observation vector
        #return 4 elem array, normalized

        next_pipe = None #find next pipe
        for pipe in self.pipes: 
            if pipe["x"] + self.PIPE_WIDTH > self.BIRD_X: 
                next_pipe = pipe
                break
        
        if next_pipe is None:  #next_pipe should not be none but backup/default case
            pipe_dx = 1.0
            pipe_dy = 0.0
        else: 
            pipe_dx = (next_pipe["x"] - self.BIRD_X) / self.SCREEN_WIDTH
            pipe_dx = np.clip(pipe_dx, 0.0, 1.0)

            gap_center = next_pipe["gap_y"]
            bird_center = self.bird_y + self.BIRD_SIZE / 2
            pipe_dy = (gap_center - bird_center) / (self.SCREEN_HEIGHT / 2)
            pipe_dy = np.clip(pipe_dy, -1.0, 1.0)

        bird_y_norm = self.bird_y / self.SCREEN_HEIGHT
        bird_y_norm = np.clip(bird_y_norm, 0.0, 1.0)

        bird_vel_norm = self.bird_vel / self.MAX_VELOCITY
        bird_vel_norm = np.clip(bird_vel_norm, -1.0, 1.0)

        return np.array([bird_y_norm, bird_vel_norm, pipe_dx, pipe_dy], dtype=np.float32)

    def _get_info(self) -> Dict[str, Any]:
        #return additional info about current state for debugging, logging etc
        return {
            "score": self.score, 
            "steps": self.steps, 
            "bird_y": self.bird_y, 
            "bird_vel": self.bird_vel, 
            "num_pipes": len(self.pipes)
        }
    
    def render(self) -> Optional[np.ndarray]:
        """
        Render the current state.
        
        Returns:
            RGB array (H, W, 3) if render_mode is "rgb_array", None for "human"
        """
        if self.render_mode is None:
            return None
        
        return self._render_frame()
    
    def _render_frame(self) -> Optional[np.ndarray]:
        """
        Internal rendering logic. Creates the visual representation.
        
        - For "human" mode: displays on pygame window
        - For "rgb_array" mode: returns pixel array (no window needed)
        
        Returns:
            np.ndarray of shape (SCREEN_HEIGHT, SCREEN_WIDTH, 3) for rgb_array,
            None for human mode
        """
        # Lazy initialization of pygame
        if self.screen is None:
            pygame.init()
            
            if self.render_mode == "human":
                pygame.display.set_caption("Flappy Bird RL")
                self.screen = pygame.display.set_mode(
                    (self.SCREEN_WIDTH, self.SCREEN_HEIGHT)
                )
            else:
                # For rgb_array: use an off-screen surface (no window)
                self.screen = pygame.Surface(
                    (self.SCREEN_WIDTH, self.SCREEN_HEIGHT)
                )
            
            self.clock = pygame.time.Clock()
            self.font = pygame.font.SysFont("Arial", 24, bold=True)
        
        # === DRAW EVERYTHING ===
        
        # Sky background
        self.screen.fill(self.COLOR_SKY)
        
        # Ground (bottom strip)
        ground_height = 20
        pygame.draw.rect(
            self.screen,
            self.COLOR_GROUND,
            (0, self.SCREEN_HEIGHT - ground_height, self.SCREEN_WIDTH, ground_height)
        )
        
        # Pipes
        for pipe in self.pipes:
            pipe_x = int(pipe["x"])
            gap_y = int(pipe["gap_y"])
            gap_half = self.PIPE_GAP // 2
            
            # Top pipe (from top of screen to gap start)
            top_pipe_rect = pygame.Rect(
                pipe_x, 0,
                self.PIPE_WIDTH, gap_y - gap_half
            )
            pygame.draw.rect(self.screen, self.COLOR_PIPE, top_pipe_rect)
            pygame.draw.rect(self.screen, self.COLOR_PIPE_BORDER, top_pipe_rect, 3)
            
            # Top pipe cap (wider lip at bottom of top pipe)
            cap_overhang = 4
            top_cap_rect = pygame.Rect(
                pipe_x - cap_overhang,
                gap_y - gap_half - 20,
                self.PIPE_WIDTH + cap_overhang * 2,
                20
            )
            pygame.draw.rect(self.screen, self.COLOR_PIPE, top_cap_rect)
            pygame.draw.rect(self.screen, self.COLOR_PIPE_BORDER, top_cap_rect, 3)
            
            # Bottom pipe (from gap end to bottom of screen)
            bottom_pipe_rect = pygame.Rect(
                pipe_x, gap_y + gap_half,
                self.PIPE_WIDTH, self.SCREEN_HEIGHT - (gap_y + gap_half)
            )
            pygame.draw.rect(self.screen, self.COLOR_PIPE, bottom_pipe_rect)
            pygame.draw.rect(self.screen, self.COLOR_PIPE_BORDER, bottom_pipe_rect, 3)
            
            # Bottom pipe cap (wider lip at top of bottom pipe)
            bottom_cap_rect = pygame.Rect(
                pipe_x - cap_overhang,
                gap_y + gap_half,
                self.PIPE_WIDTH + cap_overhang * 2,
                20
            )
            pygame.draw.rect(self.screen, self.COLOR_PIPE, bottom_cap_rect)
            pygame.draw.rect(self.screen, self.COLOR_PIPE_BORDER, bottom_cap_rect, 3)
        
        # Bird (yellow circle with outline)
        bird_center_x = self.BIRD_X + self.BIRD_SIZE // 2
        bird_center_y = int(self.bird_y) + self.BIRD_SIZE // 2
        bird_radius = self.BIRD_SIZE // 2

        pygame.draw.circle(
            self.screen, self.COLOR_BIRD,
            (bird_center_x, bird_center_y), bird_radius
        )
        
        # Score text (with shadow for readability)
        score_text = self.font.render(f"Score: {self.score}", True, self.COLOR_TEXT_SHADOW)
        self.screen.blit(score_text, (12, 12))
        score_text = self.font.render(f"Score: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))
        
        # === OUTPUT ===
        
        if self.render_mode == "human":
            # Process pygame events (REQUIRED to prevent window freezing)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.close()
                    return None
            
            pygame.display.flip()
            self.clock.tick(self.metadata["render_fps"])  # Cap at 30 FPS
            return None
        
        elif self.render_mode == "rgb_array":
            # Convert pygame surface to numpy array
            # pygame surfarray gives (W, H, 3), we need (H, W, 3)
            frame = pygame.surfarray.array3d(self.screen)
            frame = np.transpose(frame, (1, 0, 2))  # (W,H,3) -> (H,W,3)
            return frame
    
    def close(self) -> None:
        """Clean up pygame resources."""
        if self.screen is not None:
            pygame.quit()
            self.screen = None
            self.clock = None
            self.font = None
        