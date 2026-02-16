"""
Run all tests: 
pytest tests/test_env_reset_step.py -v

For more detail:
pytest tests/test_env_reset_step.py -v --tb=short


Run a specific test class:
pytest tests/test_env_reset_step.py::TestPhysics -v

Run a specific test
pytest tests/test_env_reset_step.py::TestPhysics::test_gravity_pulls_bird_down -v
"""


"""
Unit Tests for FlappyBirdEnv

These tests verify the environment follows the Gymnasium API correctly and that the game physics work as expected
"""

import pytest
import numpy as np
import sys
sys.path.insert(0, '.')

from env.flappy_env import FlappyBirdEnv


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def env():
    """Create a fresh environment for each test."""
    env = FlappyBirdEnv()
    yield env
    env.close()


@pytest.fixture
def env_short():
    """Environment with short max_steps for truncation testing."""
    env = FlappyBirdEnv(max_steps=50)
    yield env
    env.close()


# =============================================================================
# SPACE TESTS
# =============================================================================

class TestSpaces:
    """Tests for observation and action space definitions."""
    
    def test_action_space_is_discrete(self, env):
        """Action space should be Discrete(2)."""
        from gymnasium.spaces import Discrete
        assert isinstance(env.action_space, Discrete)
        assert env.action_space.n == 2
    
    def test_observation_space_is_box(self, env):
        """Observation space should be Box with correct shape."""
        from gymnasium.spaces import Box
        assert isinstance(env.observation_space, Box)
        assert env.observation_space.shape == (4,)
        assert env.observation_space.dtype == np.float32
    
    def test_observation_space_bounds(self, env):
        """Observation space should have correct bounds."""
        # bird_y: [0, 1], bird_vel: [-1, 1], pipe_dx: [0, 1], pipe_dy: [-1, 1]
        expected_low = np.array([0.0, -1.0, 0.0, -1.0], dtype=np.float32)
        expected_high = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)
        
        np.testing.assert_array_equal(env.observation_space.low, expected_low)
        np.testing.assert_array_equal(env.observation_space.high, expected_high)
    
    def test_action_sample_is_valid(self, env):
        """Sampled actions should be in action space."""
        for _ in range(100):
            action = env.action_space.sample()
            assert env.action_space.contains(action)
            assert action in [0, 1]


# =============================================================================
# RESET TESTS
# =============================================================================

class TestReset:
    """Tests for the reset() method."""
    
    def test_reset_returns_tuple(self, env):
        """Reset should return (observation, info) tuple."""
        result = env.reset()
        assert isinstance(result, tuple)
        assert len(result) == 2
    
    def test_reset_observation_type(self, env):
        """Reset observation should be numpy array with correct dtype."""
        obs, info = env.reset()
        assert isinstance(obs, np.ndarray)
        assert obs.dtype == np.float32
        assert obs.shape == (4,)
    
    def test_reset_observation_in_space(self, env):
        """Reset observation should be within observation space."""
        for seed in range(10):
            obs, _ = env.reset(seed=seed)
            assert env.observation_space.contains(obs), \
                f"Observation {obs} not in space for seed {seed}"
    
    def test_reset_info_is_dict(self, env):
        """Reset info should be a dictionary."""
        _, info = env.reset()
        assert isinstance(info, dict)
    
    def test_reset_info_contains_expected_keys(self, env):
        """Reset info should contain expected keys."""
        _, info = env.reset()
        expected_keys = {"score", "steps", "bird_y", "bird_vel", "num_pipes"}
        assert expected_keys.issubset(info.keys())
    
    def test_reset_initial_state(self, env):
        """Reset should initialize bird at center with zero velocity."""
        _, info = env.reset()
        
        # Bird should be near center
        assert abs(info["bird_y"] - env.SCREEN_HEIGHT / 2) < 1.0
        
        # Velocity should be zero
        assert info["bird_vel"] == 0.0
        
        # Score should be zero
        assert info["score"] == 0
        
        # Steps should be zero
        assert info["steps"] == 0
    
    def test_reset_deterministic_with_seed(self, env):
        """Same seed should produce same initial observation."""
        obs1, _ = env.reset(seed=42)
        obs2, _ = env.reset(seed=42)
        
        np.testing.assert_array_equal(obs1, obs2)
    
    def test_reset_different_seeds_different_pipes(self, env):
        """Different seeds should (usually) produce different pipe positions."""
        obs1, _ = env.reset(seed=42)
        obs2, _ = env.reset(seed=123)
        
        # pipe_dy (index 3) should differ due to different gap positions
        # Note: This could theoretically be the same, but very unlikely
        # We just check they're not always identical
        observations = [env.reset(seed=i)[0] for i in range(10)]
        pipe_dys = [obs[3] for obs in observations]
        
        # At least 2 different values among 10 seeds
        assert len(set(pipe_dys)) >= 2, "Pipe positions should vary with seed"


# =============================================================================
# STEP TESTS
# =============================================================================

class TestStep:
    """Tests for the step() method."""
    
    def test_step_returns_5_tuple(self, env):
        """Step should return (obs, reward, terminated, truncated, info)."""
        env.reset(seed=42)
        result = env.step(0)
        
        assert isinstance(result, tuple)
        assert len(result) == 5
    
    def test_step_return_types(self, env):
        """Step returns should have correct types."""
        env.reset(seed=42)
        obs, reward, terminated, truncated, info = env.step(0)
        
        assert isinstance(obs, np.ndarray)
        assert isinstance(reward, (int, float))
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)
    
    def test_step_observation_in_space(self, env):
        """Step observations should always be in observation space."""
        env.reset(seed=42)
        
        for _ in range(100):
            action = env.action_space.sample()
            obs, _, terminated, truncated, _ = env.step(action)
            
            assert env.observation_space.contains(obs), \
                f"Observation {obs} not in space"
            
            if terminated or truncated:
                env.reset(seed=42)
    
    def test_step_invalid_action_raises(self, env):
        """Invalid action should raise an error."""
        env.reset(seed=42)
        
        with pytest.raises(AssertionError):
            env.step(2)  # Only 0 and 1 are valid
        
        with pytest.raises(AssertionError):
            env.step(-1)
    
    def test_step_increments_step_counter(self, env):
        """Each step should increment the step counter."""
        env.reset(seed=42)
        
        for i in range(10):
            _, _, terminated, _, info = env.step(0)
            if terminated:
                break
            assert info["steps"] == i + 1


# =============================================================================
# PHYSICS TESTS
# =============================================================================

class TestPhysics:
    """Tests for game physics."""
    
    def test_gravity_pulls_bird_down(self, env):
        """Bird should fall when not flapping."""
        env.reset(seed=42)
        initial_y = env.bird_y
        
        # Take several steps without flapping
        for _ in range(5):
            _, _, terminated, _, _ = env.step(0)
            if terminated:
                break
        
        # Bird should have moved down (higher y value)
        assert env.bird_y > initial_y, "Bird should fall due to gravity"
    
    def test_flap_moves_bird_up(self, env):
        """Flapping should give upward velocity."""
        env.reset(seed=42)
        
        # Let bird fall a bit first
        for _ in range(3):
            env.step(0)
        
        y_before_flap = env.bird_y
        env.step(1)  # Flap
        
        # Velocity should be negative (upward)
        assert env.bird_vel < 0, "Flap should give negative (upward) velocity"
    
    def test_velocity_is_clamped(self, env):
        """Velocity should be clamped to MAX_VELOCITY."""
        env.reset(seed=42)
        
        # Let bird fall for many steps
        for _ in range(100):
            _, _, terminated, _, _ = env.step(0)
            if terminated:
                break
            assert abs(env.bird_vel) <= env.MAX_VELOCITY + 0.01
    
    def test_pipes_move_left(self, env):
        """Pipes should move left each step."""
        env.reset(seed=42)
        
        initial_pipe_x = env.pipes[0]["x"]
        
        # Take a step (without dying)
        env.step(1)  # Flap to stay alive
        
        # Pipe should have moved left
        assert env.pipes[0]["x"] < initial_pipe_x
        assert env.pipes[0]["x"] == initial_pipe_x - env.PIPE_SPEED


# =============================================================================
# TERMINATION TESTS
# =============================================================================

class TestTermination:
    """Tests for episode termination conditions."""
    
    def test_dies_from_falling(self, env):
        """Bird should die from falling off screen."""
        env.reset(seed=42)
        
        terminated = False
        steps = 0
        
        while not terminated and steps < 500:
            _, _, terminated, _, info = env.step(0)  # Never flap
            steps += 1
        
        assert terminated, "Bird should die from falling"
        assert info["bird_y"] >= env.SCREEN_HEIGHT - env.BIRD_SIZE
    
    def test_dies_from_ceiling(self, env):
        """Bird should die from hitting ceiling."""
        env.reset(seed=42)
        
        terminated = False
        steps = 0
        
        while not terminated and steps < 500:
            _, _, terminated, _, info = env.step(1)  # Always flap
            steps += 1
        
        assert terminated, "Bird should die from ceiling or pipe"
        # Could die from ceiling (y < 0) or pipe, both are valid
    
    def test_truncation_at_max_steps(self, env_short):
        """Episode should truncate at max_steps."""
        env_short.reset(seed=42)
        
        # Use a strategy that might keep bird alive
        steps = 0
        truncated = False
        terminated = False
        
        while not (terminated or truncated) and steps < 100:
            # Alternate flapping to try to stay alive
            action = 1 if steps % 6 == 0 else 0
            _, _, terminated, truncated, _ = env_short.step(action)
            steps += 1
        
        # Either truncated at 50 or died before
        if truncated:
            assert steps == 50, f"Should truncate at max_steps=50, got {steps}"
    
    def test_terminated_and_truncated_mutually_exclusive(self, env_short):
        """terminated and truncated should not both be True."""
        env_short.reset(seed=42)
        
        for _ in range(100):
            _, _, terminated, truncated, _ = env_short.step(env_short.action_space.sample())
            
            # They should not both be True at the same time
            assert not (terminated and truncated), \
                "terminated and truncated should be mutually exclusive"
            
            if terminated or truncated:
                break


# =============================================================================
# REWARD TESTS
# =============================================================================

class TestReward:
    """Tests for reward function."""
    
    def test_survival_reward(self, env):
        """Should get small positive reward for surviving."""
        env.reset(seed=42)
        
        # Flap to stay alive
        _, reward, terminated, _, _ = env.step(1)
        
        if not terminated:
            assert reward == pytest.approx(0.1, abs=0.01), \
                f"Survival reward should be 0.1, got {reward}"
    
    def test_death_penalty(self, env):
        """Should get negative reward for dying."""
        env.reset(seed=42)
        
        # Let bird die
        reward = 0
        for _ in range(500):
            _, reward, terminated, _, _ = env.step(0)
            if terminated:
                break
        
        assert reward == pytest.approx(-1.0, abs=0.01), \
            f"Death penalty should be -1.0, got {reward}"
    
    def test_reward_is_float(self, env):
        """Reward should always be a float."""
        env.reset(seed=42)
        
        for _ in range(50):
            _, reward, terminated, truncated, _ = env.step(env.action_space.sample())
            assert isinstance(reward, (int, float))
            
            if terminated or truncated:
                env.reset(seed=42)


# =============================================================================
# DETERMINISM TESTS
# =============================================================================

class TestDeterminism:
    """Tests for reproducibility."""
    
    def test_same_seed_same_trajectory(self, env):
        """Same seed and actions should produce identical trajectories."""
        actions = [0, 1, 0, 0, 1, 1, 0, 1, 0, 0]
        
        # First run
        obs1, _ = env.reset(seed=42)
        trajectory1 = [obs1.copy()]
        rewards1 = []
        
        for action in actions:
            obs, reward, terminated, truncated, _ = env.step(action)
            trajectory1.append(obs.copy())
            rewards1.append(reward)
            if terminated or truncated:
                break
        
        # Second run with same seed
        obs2, _ = env.reset(seed=42)
        trajectory2 = [obs2.copy()]
        rewards2 = []
        
        for action in actions:
            obs, reward, terminated, truncated, _ = env.step(action)
            trajectory2.append(obs.copy())
            rewards2.append(reward)
            if terminated or truncated:
                break
        
        # Compare
        assert len(trajectory1) == len(trajectory2)
        for i, (o1, o2) in enumerate(zip(trajectory1, trajectory2)):
            np.testing.assert_array_almost_equal(o1, o2, decimal=5,
                err_msg=f"Observations differ at step {i}")
        
        assert rewards1 == rewards2
    
    def test_different_seeds_different_trajectories(self, env):
        """Different seeds should produce different trajectories."""
        actions = [0] * 20  # Same actions
        
        obs1, _ = env.reset(seed=42)
        for action in actions:
            obs1, _, terminated, _, _ = env.step(action)
            if terminated:
                break
        
        obs2, _ = env.reset(seed=123)
        for action in actions:
            obs2, _, terminated, _, _ = env.step(action)
            if terminated:
                break
        
        # Observations should differ (due to different pipe positions)
        assert not np.allclose(obs1, obs2), \
            "Different seeds should produce different observations"


# =============================================================================
# EDGE CASE TESTS
# =============================================================================

class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""
    
    def test_many_resets(self, env):
        """Environment should handle many resets without issues."""
        for i in range(100):
            obs, info = env.reset(seed=i)
            assert env.observation_space.contains(obs)
    
    def test_step_after_terminated(self, env):
        """Behavior after termination (should ideally reset or warn)."""
        env.reset(seed=42)
        
        # Run until terminated
        for _ in range(500):
            _, _, terminated, _, _ = env.step(0)
            if terminated:
                break
        
        # Gymnasium doesn't strictly require error on step after done,
        # but the state should be consistent
        # Just verify no crash
        obs, _, _, _, _ = env.step(0)
        assert obs is not None
    
    def test_observation_normalization_bounds(self, env):
        """Observations should stay within normalized bounds even at extremes."""
        env.reset(seed=42)
        
        # Run many episodes collecting observations
        all_obs = []
        
        for episode in range(10):
            env.reset(seed=episode)
            for _ in range(100):
                obs, _, terminated, truncated, _ = env.step(env.action_space.sample())
                all_obs.append(obs)
                if terminated or truncated:
                    break
        
        all_obs = np.array(all_obs)
        
        # Check bounds
        assert np.all(all_obs >= env.observation_space.low - 0.01), \
            f"Observations below lower bound: {all_obs.min(axis=0)}"
        assert np.all(all_obs <= env.observation_space.high + 0.01), \
            f"Observations above upper bound: {all_obs.max(axis=0)}"


# =============================================================================
# GYMNASIUM API COMPLIANCE TESTS
# =============================================================================

class TestGymnasiumCompliance:
    """Tests for Gymnasium API compliance."""
    
    def test_has_required_attributes(self, env):
        """Environment should have all required Gymnasium attributes."""
        assert hasattr(env, 'action_space')
        assert hasattr(env, 'observation_space')
        assert hasattr(env, 'reset')
        assert hasattr(env, 'step')
        assert hasattr(env, 'render')
        assert hasattr(env, 'close')
    
    def test_metadata_exists(self, env):
        """Environment should have metadata dict."""
        assert hasattr(env, 'metadata')
        assert isinstance(env.metadata, dict)
        assert 'render_modes' in env.metadata
    
    def test_render_mode_in_metadata(self, env):
        """Render mode should be listed in metadata."""
        render_modes = env.metadata.get('render_modes', [])
        assert 'human' in render_modes or 'rgb_array' in render_modes


# =============================================================================
# RUN TESTS
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])