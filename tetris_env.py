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
    WindowEvent.PRESS_ARROW_UP: "UP",
    WindowEvent.PRESS_BUTTON_A: "A",
    WindowEvent.PRESS_BUTTON_B: "B",
    WindowEvent.PASS: "PASS",
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
    elif action == "UP":
        return WindowEvent.PRESS_ARROW_UP
    elif action == "A":
        return WindowEvent.PRESS_BUTTON_A
    elif action == "B":
        return WindowEvent.PRESS_BUTTON_B
    elif action == "PASS":
        return WindowEvent.PASS
    elif action == "START":
        return WindowEvent.PRESS_BUTTON_START
    else:
        raise ValueError(f"Invalid action: {action}")


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
        action_freq: int = 24,
        window: str = "SDL2",
        log_level: str = "ERROR",
    ) -> None:
        """
        Initialize the Tetris environment.

        Args:
            gb_path (str): Path to the Game Boy ROM file.
            init_state (str): Path to the initial state file.
            speedup (int): Speed multiplier for the emulator.
            action_freq (int): Frequency of actions in emulator ticks.
            window (str): Window backend for PyBoy (e.g., "SDL2").
            log_level (str): Logging level (e.g., "ERROR", "DEBUG").
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

        self.valid_actions = [
            WindowEvent.PRESS_ARROW_LEFT,
            WindowEvent.PRESS_ARROW_RIGHT,
            WindowEvent.PRESS_ARROW_DOWN,
            WindowEvent.PRESS_ARROW_UP,
            WindowEvent.PRESS_BUTTON_A,
            WindowEvent.PRESS_BUTTON_B,
            WindowEvent.PASS,
        ]

        self.release_arrow = [
            WindowEvent.RELEASE_ARROW_LEFT,
            WindowEvent.RELEASE_ARROW_RIGHT,
            WindowEvent.RELEASE_ARROW_DOWN,
            WindowEvent.RELEASE_ARROW_UP,
        ]

        self.release_button = [
            WindowEvent.RELEASE_BUTTON_A,
            WindowEvent.RELEASE_BUTTON_B,
        ]

        self.output_shape = (18, 10)
        self.board = np.zeros(self.output_shape)

        self.action_space = spaces.Discrete(len(self.valid_actions))
        self.observation_space = spaces.Box(low=0, high=8, shape=self.output_shape, dtype=np.uint8)

        self.current_score = 0

        self.pyboy = PyBoy(
            gamerom=self.gb_path,
            log_level="INFO",
            no_input=False,
            window=self.window,
        )

        self.pyboy.set_emulation_speed(0 if self.window == "null" else self.speedup)

        self.pyboy.game_wrapper.start_game()
        self._tile_mapping = self.pyboy.game_wrapper.mapping_compressed
        self.reset()

    def reset(self, seed: int | None = None) -> tuple[np.ndarray, dict]:
        """
        Reset the environment to its initial state.

        Args:
            seed (int, optional): Seed for random number generation.

        Returns:
            tuple: Observation of the board and an empty dictionary.
        """
        super().reset(seed=seed)

        if self.init_state != "":
            with open(self.init_state, "rb") as f:
                self.pyboy.load_state(f)

        if seed is not None:
            for _ in range(seed % 60):
                self.pyboy.tick()

        observation = self.render()
        self.current_score = self.get_game_score()
        self.board = observation
        return observation, {}

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        """
        Perform an action in the environment.

        Args:
            action (int): Index of the action to perform.

        Returns:
            tuple: Observation, reward, terminated flag, truncated flag, and additional info.
        """
        self.do_input(action)
        observation = self.render()

        if self.pyboy.game_wrapper.game_over():
            return observation, -100, True, False, {}

        game_score = self.get_game_score()
        reward = game_score - self.current_score
        self.current_score = game_score
        self.board = observation

        logging.debug(f"Game Score: {game_score}")
        logging.debug(f"Reward: {reward}")

        return observation, reward, False, False, {}

    def render(self) -> np.ndarray:
        """
        Render the current state of the game board.

        Returns:
            numpy.ndarray: 2D binary array representing the game board.
        """
        mapping = self._tile_mapping
        ga = self.pyboy.game_area()
        flat = ga.flatten().astype(np.int64)
        flat[flat >= len(mapping)] = 0
        compressed = mapping[flat].reshape(ga.shape)
        self.observation = compressed.astype(np.uint8)
        return self.observation

    def get_total_score(self, observation: np.ndarray) -> float:
        """
        Calculate the total score based on the current observation.

        Args:
            observation (numpy.ndarray): Current state of the game board.

        Returns:
            int: Total score.
        """
        height_score = self.get_aggregate_height(observation)
        completion_score = self.get_complete_lines(observation)
        holes_score = self.get_holes_count(observation)
        bumpiness_score = self.get_bumpiness(observation)
        logging.debug(f"Height Score: {height_score}")
        logging.debug(f"Bumpiness Score: {bumpiness_score}")
        logging.debug(f"Completion Score: {completion_score}")
        logging.debug(f"Holes Score: {holes_score}")

        scores = [height_score, completion_score, holes_score, bumpiness_score]

        weights = [-0.5, 0.75, -0.35, -0.2]
        return np.sum(np.multiply(weights, scores))

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

    def do_input(self, action_idx: int) -> None:
        """
        Perform an input action in the emulator.

        Args:
            action_idx (int): Index of the action (0-6) within valid_actions.
        """
        press_event = self.valid_actions[action_idx]
        self.pyboy.send_input(press_event)
        for i in range(self.action_freq):
            if i == 8:
                if action_idx < 4:
                    self.pyboy.send_input(self.release_arrow[action_idx])
                elif action_idx < 6:
                    self.pyboy.send_input(self.release_button[action_idx - 4])
                elif press_event == WindowEvent.PRESS_BUTTON_START:
                    self.pyboy.send_input(WindowEvent.RELEASE_BUTTON_START)
            self.pyboy.tick()
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
