import numpy as np
import pytest
from pyboy.utils import WindowEvent

from tetris_env import FEATURE_DIM, TETROMINOES, action_names, one_hot, parse_action, tetris_env, tetromino_to_id


@pytest.fixture(scope="module")
def env():
    """Integration env requiring ROM — skipped on CI without ROM."""
    if not __import__("pathlib").Path("roms/tetris.gb").exists():
        pytest.skip("roms/tetris.gb not found (CI without ROM)")
    e = tetris_env(gb_path="roms/tetris.gb", window="null", log_level="ERROR")
    yield e
    e.close()


@pytest.fixture(scope="module")
def helper_env():
    """Lightweight env for pure helper tests — no ROM/PyBoy required.

    Board helpers (get_column_height, get_holes_count, etc.) only use
    the board array, not emulator state. Construct via __new__ to avoid
    PyBoy init, allowing CI without ROM on Ubuntu.
    """
    e = tetris_env.__new__(tetris_env)
    # minimal attrs used by helpers (none currently, but safe defaults)
    e.output_shape = (18, 10)
    e.obs_mode = "enriched"
    e._prev_next_idx = -1
    e._current_idx = -1
    yield e


class TestParseAction:
    def test_left(self):
        assert parse_action("left") == WindowEvent.PRESS_ARROW_LEFT
        assert parse_action("LEFT") == WindowEvent.PRESS_ARROW_LEFT

    def test_right(self):
        assert parse_action("right") == WindowEvent.PRESS_ARROW_RIGHT

    def test_down(self):
        assert parse_action("down") == WindowEvent.PRESS_ARROW_DOWN

    def test_up(self):
        # UP removed in 6-action env (was no-op in GB Tetris) — now invalid
        with pytest.raises(ValueError, match="Invalid action"):
            parse_action("up")

    def test_a(self):
        assert parse_action("a") == WindowEvent.PRESS_BUTTON_A

    def test_b(self):
        assert parse_action("b") == WindowEvent.PRESS_BUTTON_B

    def test_pass(self):
        # PASS removed in 5-action env (let pieces fall centrally without moving) — now invalid
        with pytest.raises(ValueError, match="Invalid action"):
            parse_action("pass")

    def test_start(self):
        assert parse_action("start") == WindowEvent.PRESS_BUTTON_START

    def test_invalid(self):
        with pytest.raises(ValueError, match="Invalid action"):
            parse_action("invalid")

    def test_whitespace(self):
        assert parse_action("  left  ") == WindowEvent.PRESS_ARROW_LEFT


class TestActionNames:
    def test_all_valid_actions_mapped(self, env):
        for action in env.valid_actions:
            assert action in action_names, f"Action {action} missing from action_names"


class TestBoardHelpers:
    """Deterministic pure-function tests (no emulator state needed)."""

    @staticmethod
    def _empty_board():
        return np.zeros((18, 10), dtype=np.uint8)

    @staticmethod
    def _full_board():
        return np.ones((18, 10), dtype=np.uint8)

    def test_get_aggregate_height_empty(self, helper_env):
        assert helper_env.get_aggregate_height(self._empty_board()) == 0

    def test_get_aggregate_height_partial(self, helper_env):
        b = self._empty_board()
        b[15:, 0] = 1  # 3 blocks at bottom of column 0
        assert helper_env.get_aggregate_height(b) == 3

    def test_get_complete_lines_none(self, helper_env):
        assert helper_env.get_complete_lines(self._empty_board()) == 0
        assert helper_env.get_complete_lines(self._full_board()) == 18

    def test_get_complete_lines_one(self, helper_env):
        b = self._empty_board()
        b[17, :] = 1
        assert helper_env.get_complete_lines(b) == 1

    def test_get_holes_count_none(self, helper_env):
        assert helper_env.get_holes_count(self._empty_board()) == 0
        assert helper_env.get_holes_count(self._full_board()) == 0

    def test_get_holes_count_with_holes(self, helper_env):
        b = self._empty_board()
        b[14, 0] = 1  # block
        b[17, 0] = 1  # block below with empty in between = holes at rows 15,16
        assert helper_env.get_holes_count(b) == 2

    def test_get_holes_count_compressed_tiles(self, helper_env):
        """Ubuntu phase-3 fix: values 2-8 must count as blocks (was ==1)."""
        b = self._empty_board()
        b[14, 0] = 2  # compressed tile !=1
        b[17, 0] = 5
        assert helper_env.get_holes_count(b) == 2

    def test_get_bumpiness_empty(self, helper_env):
        assert helper_env.get_bumpiness(self._empty_board()) == 0

    def test_get_column_height_empty(self, helper_env):
        assert helper_env.get_column_height(np.zeros(18), 18) == 0

    def test_get_column_height_with_block(self, helper_env):
        col = np.zeros(18)
        col[14] = 1  # first block at row 14 → height = 18 - 14 = 4
        assert helper_env.get_column_height(col, 18) == 4

    def test_get_column_height_compressed_tile(self, helper_env):
        col = np.zeros(18, dtype=np.uint8)
        col[14] = 8  # max compressed value
        assert helper_env.get_column_height(col, 18) == 4

    def test_get_total_score_returns_number(self, helper_env):
        score = helper_env.get_total_score(self._empty_board())
        assert isinstance(score, int | float | np.number)

    def test_get_total_score_accepts_dict(self, helper_env):
        obs = {"board": self._empty_board(), "next": one_hot(0), "current": one_hot(1)}
        assert helper_env.get_total_score(obs) == helper_env.get_total_score(self._empty_board())


class TestRobustObs:
    """ROM-less tests for next/current/features helpers."""

    def test_tetromino_ids(self):
        assert [tetromino_to_id(t) for t in TETROMINOES] == list(range(7))
        assert tetromino_to_id("x") == -1
        assert tetromino_to_id(None) == -1

    def test_one_hot(self):
        assert one_hot(3).tolist() == [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
        assert one_hot(-1).sum() == 0.0

    def test_features_empty(self, helper_env):
        f = helper_env.get_features(np.zeros((18, 10), dtype=np.uint8))
        assert f.shape == (FEATURE_DIM,)
        assert f.dtype == np.float32
        assert (f == 0).all()

    def test_features_single_column(self, helper_env):
        b = np.zeros((18, 10), dtype=np.uint8)
        b[15:, 0] = 1
        f = helper_env.get_features(b)
        assert f.shape == (FEATURE_DIM,)
        assert f[0] == pytest.approx(3 / 18)
        assert (f >= 0).all() and (f <= 1).all()

    def test_build_obs_enriched_romless(self, helper_env):
        b = np.zeros((18, 10), dtype=np.uint8)
        obs = helper_env._build_obs(b)
        assert set(obs) == {"board", "next", "current", "features"}
        assert obs["board"].shape == (18, 10)
        assert obs["next"].shape == (7,)

    def test_vectorized_stats_parity(self, helper_env):
        """_board_stats must match legacy scalar helpers on varied boards."""
        rng = np.random.default_rng(7)
        for _ in range(50):
            density = rng.random()
            b = ((rng.random((18, 10)) < density) * rng.integers(0, 9, size=(18, 10))).astype(np.uint8)
            heights, holes, bump, lines = helper_env._board_stats(b)
            assert list(heights) == [helper_env.get_column_height(b[:, c], 18) for c in range(10)]
            assert holes == helper_env.get_holes_count(b)
            assert bump == helper_env.get_bumpiness(b)
            assert lines == helper_env.get_complete_lines(b)
            f = helper_env.get_features(b)
            assert f.shape == (14,) and f.dtype == np.float32
            assert (f >= 0).all() and (f <= 1).all()

    def test_obs_mode_guard(self):
        with pytest.raises(ValueError, match="Invalid obs_mode"):
            tetris_env(gb_path="roms/tetris.gb", window="null", obs_mode="bad", log_level="ERROR")


class TestTetrisEnvGuards:
    """Ubuntu portability guards from phases 3-4."""

    def test_rom_guard(self):
        with pytest.raises(FileNotFoundError, match="ROM not found"):
            tetris_env(gb_path="roms/missing.gb", window="null", log_level="ERROR")

    def test_init_state_guard(self):
        if not __import__("pathlib").Path("roms/tetris.gb").exists():
            pytest.skip("requires ROM")
        with pytest.raises(FileNotFoundError, match="Init state not found"):
            tetris_env(gb_path="roms/tetris.gb", init_state="states/missing.state", window="null")

    def test_window_guard(self):
        with pytest.raises(ValueError, match="Invalid window"):
            tetris_env(gb_path="roms/tetris.gb", window="bad", log_level="ERROR")

    def test_headless_alias(self):
        if not __import__("pathlib").Path("roms/tetris.gb").exists():
            pytest.skip("requires ROM")
        e = tetris_env(gb_path="roms/tetris.gb", window="headless", log_level="ERROR")
        assert e.window == "null"
        e.close()

    def test_close_idempotent(self):
        if not __import__("pathlib").Path("roms/tetris.gb").exists():
            pytest.skip("requires ROM")
        e = tetris_env(gb_path="roms/tetris.gb", window="null", log_level="ERROR")
        e.close()
        e.close()  # should not raise
