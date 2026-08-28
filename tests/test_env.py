import numpy as np
import pytest
from gymnasium import Env, spaces

from tetris_env import tetris_env


@pytest.fixture(scope="module")
def env():
    e = tetris_env(gb_path="roms/tetris.gb", window="null", log_level="ERROR")
    yield e
    e.close()


class TestEnvGymnasiumAPI:
    """Verifies the environment conforms to Gymnasium Env contract."""

    def test_is_gymnasium_env(self, env):
        assert isinstance(env, Env)

    def test_action_space(self, env):
        assert isinstance(env.action_space, spaces.Discrete)
        assert env.action_space.n == len(env.valid_actions)

    def test_observation_space(self, env):
        assert isinstance(env.observation_space, spaces.Box)
        assert env.observation_space.shape == (18, 10)
        assert env.observation_space.dtype == np.uint8
        assert env.observation_space.low.min() == 0
        assert env.observation_space.high.max() == 8

    def test_reset_returns_obs_and_info(self, env):
        obs, info = env.reset(seed=42)
        assert isinstance(obs, np.ndarray)
        assert obs.shape == (18, 10)
        assert obs.dtype == np.uint8
        assert obs.max() <= 8
        assert isinstance(info, dict)

    def test_reset_different_seeds(self, env):
        obs1, _ = env.reset(seed=42)
        obs2, _ = env.reset(seed=999)
        assert not np.array_equal(obs1, obs2)

    def test_step_returns_5_tuple(self, env):
        env.reset(seed=42)
        result = env.step(0)
        assert len(result) == 5
        obs, reward, terminated, truncated, info = result
        assert isinstance(obs, np.ndarray)
        assert obs.shape == (18, 10)
        assert isinstance(reward, (float, int, np.floating))
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)
        assert truncated is False  # we never truncate

    def test_step_all_actions(self, env):
        env.reset(seed=42)
        for action in range(env.action_space.n):
            obs, reward, terminated, truncated, info = env.step(action)
            assert obs.shape == (18, 10)
            env.render()
            if terminated:
                break

    def test_render_returns_obs(self, env):
        env.reset(seed=42)
        obs = env.render()
        assert isinstance(obs, np.ndarray)
        assert obs.shape == (18, 10)
        assert obs.dtype == np.uint8

    def test_close(self, env):
        env.close()
        # Re-create since the fixture scope yields once
        # Verifying close didn't crash is enough

    def test_get_game_score_returns_int(self, env):
        score = env.get_game_score()
        assert isinstance(score, int)


class TestGameOver:
    def test_game_over_starts_false(self, env):
        env.reset(seed=42)
        assert not env.pyboy.game_wrapper.game_over()


class TestObservationValues:
    def test_observation_compressed(self, env):
        env.reset(seed=42)
        obs = env.render()
        assert obs.dtype == np.uint8
        assert obs.min() >= 0 and obs.max() <= 8, "Observation values must be 0-8"
