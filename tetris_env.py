# The MIT License

# Copyright (c) 2025 Albert Murienne

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

import logging
import time
from pathlib import Path

import numpy as np
from gymnasium import Env, spaces
from pyboy import PyBoy
from pyboy.utils import WindowEvent

action_names = {
    WindowEvent.PRESS_ARROW_LEFT: "LEFT",
    WindowEvent.PRESS_ARROW_RIGHT: "RIGHT",
    WindowEvent.PRESS_ARROW_DOWN: "DOWN",
    WindowEvent.PRESS_BUTTON_A: "A",
    WindowEvent.PRESS_BUTTON_B: "B",
    WindowEvent.PRESS_BUTTON_START: "START",
}


def parse_action(s: str) -> int:
    action = s.strip().upper()
    if action == "LEFT":
        return WindowEvent.PRESS_ARROW_LEFT
    elif action == "RIGHT":
        return WindowEvent.PRESS_ARROW_RIGHT
    elif action == "DOWN":
        return WindowEvent.PRESS_ARROW_DOWN
    elif action == "A":
        return WindowEvent.PRESS_BUTTON_A
    elif action == "B":
        return WindowEvent.PRESS_BUTTON_B
    elif action == "START":
        return WindowEvent.PRESS_BUTTON_START
    else:
        raise ValueError(f"Invalid action: {action}")


# Tetromino IDs for next/current-piece observation (GB Tetris NEXT box).
# PyBoy `game_wrapper.next_tetromino()` returns one of these single chars.
TETROMINOES = ("I", "O", "T", "S", "Z", "J", "L")
TETROMINO_TO_ID = {t: i for i, t in enumerate(TETROMINOES)}
FEATURE_DIM = 14  # 10 normalized heights + holes + bump + max_well + n_wells


def tetromino_to_id(t: str | None) -> int:
    """Map next_tetromino() char to 0-6; unknown/None → -1 (caller uses zeros)."""
    if t is None:
        return -1
    return TETROMINO_TO_ID.get(str(t).upper(), -1)


def one_hot(idx: int, n: int = 7) -> np.ndarray:
    """One-hot float32 vector; idx<0 → zeros (unknown piece, robust)."""
    v = np.zeros(n, dtype=np.float32)
    if 0 <= idx < n:
        v[idx] = 1.0
    return v


class tetris_env(Env):  # noqa: N801
    """
    Defines an environment for managing the game state, the agent's actions, and the
    reward system for the Tetris game.
    """

    def __init__(
        self,
        gb_path: str = "",
        init_state: str = "",
        speedup: int = 1,
        action_freq: int = 8,
        window: str = "SDL2",
        log_level: str = "ERROR",
        shaped_alpha: float = 0.1,
        shaped_weights: tuple[float, float, float, float] | None = None,
        obs_mode: str = "enriched",
        max_steps: int = 10000,
        noop_max: int = 30,
    ) -> None:
        """
        Initialize the Tetris environment.

        Args:
            gb_path (str): Path to the Game Boy ROM file.
            init_state (str): Path to the initial state file.
            speedup (int): Speed multiplier for the emulator.
            action_freq (int): Frequency of actions in emulator ticks (8 = finer
                lateral movement before lock than legacy 24/12; more LEFT/RIGHT per drop).
            window (str): Window backend for PyBoy (e.g., "SDL2").
            log_level (str): Logging level (e.g., "ERROR", "DEBUG").
            shaped_alpha (float): PBRS coefficient for dense shaped delta.
                0.0 = pure score (legacy), 0.1-0.3 recommended (hybrid).
                Dense term is `alpha * (shaped - prev_shaped)` where shaped
                is `get_total_score()` (height/lines/holes/bump).
            shaped_weights (tuple): Optional override for
                (height, lines, holes, bump) weights. Defaults to
                (-0.5, 0.75, -0.8, -0.5) — stronger bump (-0.5 vs -0.35) punishes
                stacking first pieces centrally without spreading.
            obs_mode (str): "enriched" (Dict board+next+current+features, robust)
                or "board" (legacy Box 18×10 only, for ablation).
            max_steps (int): Episode truncation after N steps (0=disabled).
                Prevents infinite stall loops; 10000 default.
            noop_max (int): Random no-op ticks on reset in [0, noop_max] to
                desync piece RNG across parallel envs (domain randomization).
        """
        super().__init__()

        self.gb_path = gb_path
        self.init_state = init_state
        self.speedup = speedup
        self.action_freq = action_freq
        if window not in ("null", "SDL2", "headless"):
            raise ValueError(f"Invalid window '{window}': expected 'null', 'SDL2' or 'headless'")
        # "headless" is an alias for "null" (unlimited speed, no window)
        self.window = "null" if window == "headless" else window
        if not logging.getLogger().hasHandlers():
            logging.basicConfig(level=log_level.upper())
        else:
            logging.getLogger().setLevel(log_level.upper())
        if self.gb_path and not Path(self.gb_path).exists():
            raise FileNotFoundError(
                f"ROM not found: {self.gb_path} — place a legal dump at roms/tetris.gb "
                "(see README.md). If you use a different path, pass gb_path explicitly."
            )
        if self.init_state and not Path(self.init_state).exists():
            raise FileNotFoundError(f"Init state not found: {self.init_state}")
        self._closed = False
        self.shaped_alpha = float(shaped_alpha)
        # (height, completion, holes, bumpiness) — Dellacherie/Bertsekas
        # stronger bump -0.5 vs -0.35 punishes stacking first pieces in spawn column
        # without spreading; PASS removed to force lateral/rotation.
        self.shaped_weights = shaped_weights if shaped_weights is not None else (-0.5, 0.75, -0.8, -0.5)
        self.prev_shaped: float = 0.0
        if obs_mode not in ("board", "enriched"):
            raise ValueError(f"Invalid obs_mode '{obs_mode}': expected 'board' or 'enriched'")
        self.obs_mode = obs_mode
        self.max_steps = int(max_steps)
        self.noop_max = int(noop_max)
        self.step_count = 0
        # piece tracking: next_tetromino() gives upcoming piece; current is
        # inferred as previous next on change (no direct current API in PyBoy).
        self._prev_next_idx = -1
        self._current_idx = -1

        # 5 actions — UP and PASS removed (UP no-op, PASS let pieces fall centrally
        # without moving — caused “stack first pieces without moving”)
        # indices: 0 LEFT, 1 RIGHT, 2 DOWN, 3 A (rotate), 4 B (rotate)
        self.valid_actions = [
            WindowEvent.PRESS_ARROW_LEFT,
            WindowEvent.PRESS_ARROW_RIGHT,
            WindowEvent.PRESS_ARROW_DOWN,
            WindowEvent.PRESS_BUTTON_A,
            WindowEvent.PRESS_BUTTON_B,
        ]

        self.release_arrow = [
            WindowEvent.RELEASE_ARROW_LEFT,
            WindowEvent.RELEASE_ARROW_RIGHT,
            WindowEvent.RELEASE_ARROW_DOWN,
        ]

        self.release_button = [
            WindowEvent.RELEASE_BUTTON_A,
            WindowEvent.RELEASE_BUTTON_B,
        ]

        self.output_shape = (18, 10)
        self.board = np.zeros(self.output_shape)

        self.action_space = spaces.Discrete(len(self.valid_actions))
        if self.obs_mode == "board":
            self.observation_space = spaces.Box(low=0, high=8, shape=self.output_shape, dtype=np.uint8)
        else:
            self.observation_space = spaces.Dict(
                {
                    "board": spaces.Box(low=0, high=8, shape=self.output_shape, dtype=np.uint8),
                    "next": spaces.Box(low=0.0, high=1.0, shape=(7,), dtype=np.float32),
                    "current": spaces.Box(low=0.0, high=1.0, shape=(7,), dtype=np.float32),
                    "features": spaces.Box(low=0.0, high=1.0, shape=(FEATURE_DIM,), dtype=np.float32),
                }
            )

        self.current_score = 0
        self.prev_shaped = 0.0

        self.pyboy = PyBoy(
            gamerom=self.gb_path,
            log_level="INFO",
            no_input=False,
            window=self.window,
        )

        # Patch frame_limiter to avoid "Exception ignored in PyBoyWindowPlugin.frame_limiter"
        # when CTRL+C is delivered during the SDL sleep. The Cython plugin reports
        # KeyboardInterrupt as ignored if it propagates out of time.sleep. Catch it
        # inside and set quitting so the outer loop can exit cleanly.
        try:
            from pyboy.plugins.base_plugin import PyBoyWindowPlugin

            _orig_frame_limiter = PyBoyWindowPlugin.frame_limiter

            def _safe_frame_limiter(self, speed):  # type: ignore[no-untyped-def]
                try:
                    return _orig_frame_limiter(self, speed)
                except KeyboardInterrupt:
                    try:
                        self.pyboy.quitting = True
                    except Exception:
                        pass
                    return False

            PyBoyWindowPlugin.frame_limiter = _safe_frame_limiter  # type: ignore[method-assign,assignment]
        except Exception:
            pass

        self.pyboy.set_emulation_speed(0 if self.window == "null" else self.speedup)

        self.pyboy.game_wrapper.start_game()
        self._tile_mapping = self.pyboy.game_wrapper.mapping_compressed
        self.reset()

    def reset(self, seed: int | None = None) -> tuple[np.ndarray | dict, dict]:
        """
        Reset the environment to its initial state.

        Args:
            seed (int, optional): Seed for random number generation.

        Returns:
            tuple: Observation (board Box or enriched Dict) and info dict.
        """
        super().reset(seed=seed)

        if self.init_state != "":
            with open(self.init_state, "rb") as f:
                self.pyboy.load_state(f)

        if seed is not None:
            for _ in range(seed % 60):
                self.pyboy.tick()

        # domain randomization: random no-op ticks desync piece RNG across envs
        if getattr(self, "noop_max", 0) > 0:
            try:
                n_noop = int(self.np_random.integers(0, self.noop_max + 1))
            except Exception:
                n_noop = 0
            for _ in range(n_noop):
                self.pyboy.tick()
                if getattr(self.pyboy, "quitting", False):
                    raise KeyboardInterrupt

        self.step_count = 0
        board = self.render_board()
        # (re)sync piece tracking after state load
        next_idx = self._read_next_idx()
        self._prev_next_idx = next_idx
        self._current_idx = -1  # unknown until first next-change; zeros one-hot
        observation = self._build_obs(board)
        self.current_score = self.get_game_score()
        self.prev_shaped = self.get_total_score(board)
        self.board = board
        return observation, {}

    def step(self, action: int) -> tuple[np.ndarray | dict, float, bool, bool, dict]:
        """
        Perform an action in the environment.

        Args:
            action (int): Index of the action to perform.

        Returns:
            tuple: Observation, reward, terminated flag, truncated flag, and additional info.
                   Reward is hybrid PBRS: score delta + alpha * shaped delta
                   (alpha=0 → legacy pure score). Shaped is get_total_score().
                   Truncated when max_steps>0 and step_count reaches it.
        """
        self.do_input(action)
        self.step_count += 1
        board = self.render_board()
        self._update_pieces()
        observation = self._build_obs(board)

        game_score = self.get_game_score()
        shaped = self.get_total_score(board)
        shaped_delta = shaped - self.prev_shaped
        score_delta = game_score - self.current_score

        # hybrid PBRS — keeps optimal policy (Ng et al. 1999) while densifying
        reward = score_delta + self.shaped_alpha * shaped_delta

        # update trackers before early return so next reset is consistent
        self.current_score = game_score
        self.prev_shaped = shaped
        self.board = board

        if self.pyboy.game_wrapper.game_over():
            # keep -100 terminal penalty but include shaped term for PBRS
            # (if alpha=0, this is exactly legacy -100)
            return observation, -100 + self.shaped_alpha * shaped_delta, True, False, {}

        if getattr(self, "max_steps", 0) > 0 and self.step_count >= self.max_steps:
            return observation, reward, False, True, {"truncated": "max_steps"}

        logging.debug(f"Game Score: {game_score} Shaped: {shaped:.2f} Reward: {reward:.2f}")

        return observation, reward, False, False, {}

    def render_board(self) -> np.ndarray:
        """Read 18×10 compressed-tile board from the emulator."""
        mapping = self._tile_mapping
        ga = self.pyboy.game_area()
        flat = ga.flatten().astype(np.int64)
        flat[flat >= len(mapping)] = 0
        compressed = mapping[flat].reshape(ga.shape)
        self.observation = compressed.astype(np.uint8)
        return self.observation

    def render(self) -> np.ndarray | dict:
        """
        Render the current observation (board Box or enriched Dict).

        Returns:
            Board array in "board" mode, Dict in "enriched" mode.
        """
        board = self.render_board()
        return self._build_obs(board)

    def _read_next_idx(self) -> int:
        """Read next_tetromino() → 0-6; -1 on unknown/error (ROM-less safe)."""
        try:
            gw = getattr(getattr(self, "pyboy", None), "game_wrapper", None)
            if gw is None or not hasattr(gw, "next_tetromino"):
                return -1
            return tetromino_to_id(gw.next_tetromino())
        except Exception:
            return -1

    def _update_pieces(self) -> None:
        """Track current as previous next on change (no direct current API)."""
        nxt = self._read_next_idx()
        if nxt < 0:
            return
        prev = getattr(self, "_prev_next_idx", -1)
        if prev < 0:
            self._prev_next_idx = nxt
            return
        if nxt != prev:
            # next shifted → previous next is now falling
            self._current_idx = prev
            self._prev_next_idx = nxt

    def _build_obs(self, board: np.ndarray) -> np.ndarray | dict:
        """Wrap board into Box or enriched Dict observation."""
        if getattr(self, "obs_mode", "board") == "board":
            return board
        return {
            "board": board,
            "next": one_hot(getattr(self, "_prev_next_idx", -1)),
            "current": one_hot(getattr(self, "_current_idx", -1)),
            "features": self.get_features(board),
        }

    def get_features(self, board: np.ndarray) -> np.ndarray:
        """14-dim robust features: 10 heights/18 + holes/20 + bump/40 + max_well/10 + n_wells/10."""
        board = board if not isinstance(board, dict) else board.get("board", np.zeros(self.output_shape))
        h = board.shape[0]
        heights = [self.get_column_height(board[:, c], h) for c in range(board.shape[1])]
        holes = self.get_holes_count(board)
        bump = self.get_bumpiness(board)
        depths = []
        for c in range(len(heights)):
            left = heights[c - 1] if c > 0 else heights[c]
            right = heights[c + 1] if c < len(heights) - 1 else heights[c]
            ref = min(left, right)
            depths.append(max(0, ref - heights[c]))
        max_well = max(depths) if depths else 0
        n_wells = sum(1 for d in depths if d >= 2)
        feats = [min(x / 18.0, 1.0) for x in heights]
        feats += [min(holes / 20.0, 1.0), min(bump / 40.0, 1.0), min(max_well / 10.0, 1.0), min(n_wells / 10.0, 1.0)]
        return np.asarray(feats, dtype=np.float32)

    def get_total_score(self, observation: np.ndarray | dict) -> float:
        """
        Calculate the total score based on the current observation.

        Args:
            observation: Board array or enriched Dict (uses ["board"]).

        Returns:
            int: Total score.
        """
        board = observation["board"] if isinstance(observation, dict) else observation
        height_score = self.get_aggregate_height(board)
        completion_score = self.get_complete_lines(board)
        holes_score = self.get_holes_count(board)
        bumpiness_score = self.get_bumpiness(board)
        logging.debug(f"Height Score: {height_score}")
        logging.debug(f"Bumpiness Score: {bumpiness_score}")
        logging.debug(f"Completion Score: {completion_score}")
        logging.debug(f"Holes Score: {holes_score}")

        scores = [height_score, completion_score, holes_score, bumpiness_score]

        weights = getattr(self, "shaped_weights", (-0.5, 0.75, -0.35, -0.2))
        return float(np.sum(np.multiply(weights, scores)))

    def get_game_score(self) -> int:
        """
        Get the current score from the emulator.

        Returns:
            int: Current score.
        """
        return self.pyboy.game_wrapper.score

    def get_bumpiness(self, board: np.ndarray) -> int:
        """
        Calculate the bumpiness of the board, i.e., the variation of its column heights.
        It is computed by summing up the absolute differences between the heights of
        adjacent columns.

        Args:
            board (numpy.ndarray): Current state of the game board.

        Returns:
            int: Bumpiness score.
        """
        column_heights = [self.get_column_height(board[:, col], board.shape[0]) for col in range(board.shape[1])]

        bumpiness = sum(abs(column_heights[i] - column_heights[i + 1]) for i in range(len(column_heights) - 1))

        logging.debug(f"Column Heights: {column_heights}")
        logging.debug(f"Bumpiness: {bumpiness}")
        return bumpiness

    def get_complete_lines(self, board: np.ndarray) -> int:
        """
        Count the number of complete lines in the board.

        Args:
            board (numpy.ndarray): Current state of the game board.

        Returns:
            int: Number of complete lines.
        """
        return np.sum(np.all(board, axis=1))

    def get_aggregate_height(self, board: np.ndarray) -> int:
        """
        Calculate the aggregate height of the board based on the first valid block
        in each column when going from the top.

        Args:
            board (numpy.ndarray): Current state of the game board.

        Returns:
            int: Aggregate height.
        """
        aggregate_height = sum(self.get_column_height(board[:, col], board.shape[0]) for col in range(board.shape[1]))

        logging.debug(f"Aggregate Height: {aggregate_height}")
        return aggregate_height

    def get_holes_count(self, board: np.ndarray) -> int:
        """
        Count the number of holes in the board.

        A hole is defined as an empty space such that there is at least one tile
        in the same column above it.

        Args:
            board (numpy.ndarray): Current state of the game board.

        Returns:
            int: Number of holes.
        """
        holes = 0
        for col in range(board.shape[1]):
            column = board[:, col]
            block_found = False
            for row in range(board.shape[0]):
                if column[row] != 0:
                    block_found = True
                elif block_found and column[row] == 0:
                    holes += 1
        return holes

    def tick(self) -> None:
        """
        Advance the emulator by one tick.
        """
        self.pyboy.tick()
        if getattr(self.pyboy, "quitting", False):
            raise KeyboardInterrupt

    def do_input(self, action_idx: int) -> None:
        """
        Perform an input action in the emulator.

        Args:
            action_idx (int): Index of the action (0-4) within valid_actions
                (0 LEFT, 1 RIGHT, 2 DOWN, 3 A, 4 B).
        """
        press_event = self.valid_actions[action_idx]
        self.pyboy.send_input(press_event)
        for i in range(self.action_freq):
            # hold 4 ticks then release (freq 8 → 4 ticks idle; freq 12 → 4 ticks idle)
            release_tick = min(4, self.action_freq - 1)
            if i == release_tick:
                if action_idx < 3:
                    self.pyboy.send_input(self.release_arrow[action_idx])
                elif action_idx < 5:
                    self.pyboy.send_input(self.release_button[action_idx - 3])
                elif press_event == WindowEvent.PRESS_BUTTON_START:
                    self.pyboy.send_input(WindowEvent.RELEASE_BUTTON_START)
            self.pyboy.tick()
            if getattr(self.pyboy, "quitting", False):
                raise KeyboardInterrupt
        logging.debug(f"Action: {action_names[press_event]}")

    def save_state(self, dest: str = "") -> None:
        """
        Save the current state of the emulator to a file.

        Args:
            dest (str): Destination file path. Defaults to a timestamped filename.
        """
        if dest == "":
            dest = time.strftime("%Y%m%d-%H%M%S.save")

        with open(dest, "wb") as f:
            self.pyboy.save_state(f)

    def load_state(self, src: str) -> None:
        """
        Load a saved state into the emulator.

        Args:
            src (str): Source file path of the saved state.
        """
        with open(src, "rb") as f:
            self.pyboy.load_state(f)

    def get_column_height(self, column: np.ndarray, board_height: int) -> int:
        """
        Calculate the height of a column based on the first valid block from the top.

        Any non-zero compressed tile (1-8) counts as a block; 0 is empty.
        Used by bumpiness/aggregate_height (reward itself uses game_wrapper.score).

        Args:
            column (numpy.ndarray): A single column of the board.
            board_height (int): Total height of the board.

        Returns:
            int: Height of the column.
        """
        for row in range(board_height):
            if column[row] != 0:
                return board_height - row
        return 0

    def close(self) -> None:
        """
        Clean up resources. Stops the PyBoy emulator. Idempotent.
        """
        if getattr(self, "_closed", False):
            return
        self._closed = True
        try:
            self.pyboy.stop()
        except Exception:
            pass
