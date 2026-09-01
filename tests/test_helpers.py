import numpy as np
import pytest
from pyboy.utils import WindowEvent

from tetris_env import action_names, parse_action, tetris_env


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
        assert parse_action("pass") == WindowEvent.PASS

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
