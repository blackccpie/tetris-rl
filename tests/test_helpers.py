import numpy as np
import pytest
from pyboy.utils import WindowEvent

from tetris_env import action_names, parse_action, tetris_env


# Create one env for all pure-function tests (the helper methods
# only read `self` for config values, not for state)
@pytest.fixture(scope="module")
def env():
    e = tetris_env(gb_path="roms/tetris.gb", window="null", log_level="ERROR")
    yield e
    e.close()


class TestParseAction:
    def test_left(self):
        assert parse_action("left") == WindowEvent.PRESS_ARROW_LEFT
        assert parse_action("LEFT") == WindowEvent.PRESS_ARROW_LEFT

    def test_right(self):
        assert parse_action("right") == WindowEvent.PRESS_ARROW_RIGHT

    def test_down(self):
        assert parse_action("down") == WindowEvent.PRESS_ARROW_DOWN

    def test_up(self):
        assert parse_action("up") == WindowEvent.PRESS_ARROW_UP

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

    def test_get_aggregate_height_empty(self, env):
        assert env.get_aggregate_height(self._empty_board()) == 0

    def test_get_aggregate_height_partial(self, env):
        b = self._empty_board()
        b[15:, 0] = 1  # 3 blocks at bottom of column 0
        assert env.get_aggregate_height(b) == 3

    def test_get_complete_lines_none(self, env):
        assert env.get_complete_lines(self._empty_board()) == 0
        assert env.get_complete_lines(self._full_board()) == 18

    def test_get_complete_lines_one(self, env):
        b = self._empty_board()
        b[17, :] = 1
        assert env.get_complete_lines(b) == 1

    def test_get_holes_count_none(self, env):
        assert env.get_holes_count(self._empty_board()) == 0
        assert env.get_holes_count(self._full_board()) == 0

    def test_get_holes_count_with_holes(self, env):
        b = self._empty_board()
        b[14, 0] = 1  # block
        b[17, 0] = 1  # block below with empty in between = holes at rows 15,16
        assert env.get_holes_count(b) == 2

    def test_get_bumpiness_empty(self, env):
        assert env.get_bumpiness(self._empty_board()) == 0

    def test_get_column_height_empty(self, env):
        assert env.get_column_height(np.zeros(18), 18) == 0

    def test_get_column_height_with_block(self, env):
        col = np.zeros(18)
        col[14] = 1  # first block at row 14 → height = 18 - 14 = 4
        assert env.get_column_height(col, 18) == 4

    def test_get_total_score_returns_number(self, env):
        score = env.get_total_score(self._empty_board())
        assert isinstance(score, (int, float, np.number))
