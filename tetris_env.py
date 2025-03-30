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

import time
import logging
from typing import List, Tuple, Dict, Optional

import numpy as np
from pyboy import PyBoy
from gymnasium import Env, spaces
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
        raise ValueError("Invalid action: {}".format(action))

class tetris_env(Env):
    """
    Defines an environment for managing the game state, the agent's actions, and the
    reward system for the Tetris game.
    """

    def __init__(self, gb_path: str = "", init_state: str = "", speedup: int = 1, action_freq: int = 24, window: str = "SDL2", log_level: str = "ERROR") -> None:
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
        self.gb_path = gb_path
        self.init_state = init_state
        self.speedup = speedup
        self.action_freq = action_freq
        self.window = window
        logging.basicConfig(level=log_level.upper())

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

        self.sprite_tiles = [i for i in range(120, 140)]
        self.output_shape = (18, 10)
        self.board = np.zeros(self.output_shape)

        # must be set in Env subclasses 
        self.action_space = spaces.Discrete(len(self.valid_actions))
        self.observation_space = spaces.Box(
            low=0, high=1, shape=self.output_shape, dtype=np.uint8
        )

        self.current_score = 0

        self.pyboy = PyBoy(
            gamerom=self.gb_path,
            log_level="INFO",
            no_input=True,
            window=self.window,
        )

        self.pyboy.set_emulation_speed(0 if self.window == "null" else self.speedup)
        self.reset()

    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, Dict]:
        """
        Reset the environment to its initial state.

        Args:
            seed (int, optional): Seed for random number generation.

        Returns:
            tuple: Observation of the board and an empty dictionary.
        """
        self.seed = seed

        # Load the initial state
        if self.init_state != "":
            with open(self.init_state, "rb") as f:
                self.pyboy.load_state(f)

        observation = self.render()
        self.current_score = self.get_total_score(observation)
        self.board = observation
        return observation, {}
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Perform an action in the environment.

        Args:
            action (int): Index of the action to perform.

        Returns:
            tuple: Observation, reward, done flag, truncated flag, and additional info.
        """
        self.do_input(self.valid_actions[action])
        observation = self.render()
        if observation[0].sum() >= len(observation[0]):
            # Game over
            return observation, -100, True, False, {}
        
        # Set reward equal to difference between current and previous score
        total_score = self.get_total_score(observation)
        reward = total_score - self.current_score
        self.current_score = total_score
        self.board = observation

        logging.debug("Total Score: {}".format(total_score))
        logging.debug("Reward: {}".format(reward))

        return observation, reward, False, False, {}
    
    def render(self) -> np.ndarray:
        """
        Render the current state of the game board.

        Returns:
            numpy.ndarray: 2D array representing the game board.
        """
        # Render the sprite map on the backgound
        background = np.asarray(self.pyboy.tilemap_background[2:12, 0:18])
        self.observation = np.where(background == 47, 0, 1)

        # Find all tile indexes for the current tetromino
        sprite_indexes = self.pyboy.get_sprite_by_tile_identifier(self.sprite_tiles, on_screen=False)
        for sprite_tiles in sprite_indexes:
            for sprite_idx in sprite_tiles:
                sprite = self.pyboy.get_sprite(sprite_idx)
                tile_x = (sprite.x // 8) - 2
                tile_y = sprite.y // 8
                if tile_x < self.output_shape[1] and tile_y < self.output_shape[0]:
                    self.observation[tile_y, tile_x] = 1
        logging.debug("Board State:\n{}".format(self.observation))
        return self.observation

    def get_total_score(self, observation: np.ndarray) -> int:
        """
        Calculate the total score based on the current observation.

        Args:
            observation (numpy.ndarray): Current state of the game board.

        Returns:
            int: Total score.
        """

        # see: https://codemyroad.wordpress.com/2013/04/14/tetris-ai-the-near-perfect-player/

        #game_score = self.get_game_score()
        height_score = self.get_aggregate_height(observation)
        completion_score = self.get_complete_lines(observation)
        holes_score = self.get_holes_count(observation)
        bumpiness_score = self.get_bumpiness(observation)
        logging.debug("Height Score: {}".format(height_score))
        logging.debug("Bumpiness Score: {}".format(bumpiness_score))
        logging.debug("Completion Score: {}".format(completion_score))
        logging.debug("Holes Score: {}".format(holes_score))

        scores = [
            height_score,
            completion_score,
            holes_score,
            bumpiness_score
        ]
        
        # now compute and return a weighed sum of the scores
        weights = [-0.5, 0.75, -0.35, -0.2]
        return np.sum(np.multiply(weights, scores))

    def get_game_score(self) -> int:
        """
        Get the current score from the emulator's memory.

        Returns:
            int: Current score.
        """
        return self.pyboy.memory[0xC0A0]
    
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
        column_heights = [
            self.get_column_height(board[:, col], board.shape[0])
            for col in range(board.shape[1])
        ]

        # Calculate bumpiness as the sum of absolute differences between adjacent columns
        bumpiness = sum(
            abs(column_heights[i] - column_heights[i + 1])
            for i in range(len(column_heights) - 1)
        )

        logging.debug("Column Heights: {}".format(column_heights))
        logging.debug("Bumpiness: {}".format(bumpiness))
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
        aggregate_height = sum(
            self.get_column_height(board[:, col], board.shape[0])
            for col in range(board.shape[1])
        )

        logging.debug("Aggregate Height: {}".format(aggregate_height))
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
        for col in range(board.shape[1]):  # Iterate over each column
            column = board[:, col]
            block_found = False
            for row in range(board.shape[0]):  # Iterate over each row in the column
                if column[row] == 1:
                    block_found = True  # A block is found above
                elif block_found and column[row] == 0:
                    holes += 1  # Count the empty space as a hole
        return holes
    
    def tick(self) -> None:
        """
        Advance the emulator by one tick.
        """
        self.pyboy.tick()
    
    def do_input(self, action: int) -> None:
        """
        Perform an input action in the emulator.

        Args:
            action (int): Action to perform.
        """
        # Press and release the button to simulate human input
        self.pyboy.send_input(action)
        for i in range(self.action_freq):
            if i == 8:
                if action < 4:
                    self.pyboy.send_input(self.release_arrow[action])
                if action > 3 and action < 6: 
                    self.pyboy.send_input(self.release_button[action - 4])
                if action == WindowEvent.PRESS_BUTTON_START:
                    self.pyboy.send_input(WindowEvent.RELEASE_BUTTON_START)
            self.pyboy.tick()
        logging.debug("Action: {}".format(action_names[action]))

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

        Args:
            column (numpy.ndarray): A single column of the board.
            board_height (int): Total height of the board.

        Returns:
            int: Height of the column.
        """
        for row in range(board_height):
            if column[row] == 1:
                return board_height - row  # Height is from the bottom
        return 0  # If no blocks are found, height is 0

    ############## OLD SCORING ############## 

    def is_hole(self, board, x, y):
        """
        Check if a given coordinate is a hole
        """
        if board[x][y] == 1:
            return False
        for adj in self.get_adjacent(board, x, y):
            if board[adj[0]][adj[1]] == 0:
                return False
        return True

    def get_adjacent(self, board: np.ndarray, x: int, y: int) -> List[Tuple[int, int]]:
        """
        Get all adjacent coordinates for a given coordinate.

        Args:
            board (numpy.ndarray): Current state of the game board.
            x (int): Row index.
            y (int): Column index.

        Returns:
            list: List of tuples representing adjacent coordinates.
        """
        adjacent = []
        shape = board.shape
        if x > 0:
            adjacent.append((x - 1, y))
        if x < shape[0] - 1:
            adjacent.append((x + 1, y))
        if y > 0:
            adjacent.append((x, y - 1))
        if y < shape[1] - 1:
            adjacent.append((x, y + 1))
        return adjacent

    def get_max_height(self, board: np.ndarray) -> int:
        """
        Get the maximum height of the blocks on the board.

        Args:
            board (numpy.ndarray): Current state of the game board.

        Returns:
            int: Maximum height.
        """
        return np.max(np.sum(board, axis=0))

    def get_total_score_old(self, observation: np.ndarray) -> int:
        """
        Calculate the total score based on the current observation.

        Args:
            observation (numpy.ndarray): Current state of the game board.

        Returns:
            int: Total score.
        """
        score = self.get_score()
        logging.debug("Score: {}".format(score))

        #board_reward = self.get_board_score(observation)
        #placement_reward = self.get_placement_score(observation)
        #surface_score = self.get_surface_area(observation) * -1
        #print("Board Reward: {}".format(board_reward))
        #print("Placement Reward: {}".format(placement_reward))
        #print("Surface Score: {}".format(surface_score))

        scores = [
            score,
            #board_reward,
            #placement_reward,
            #surface_score,
        ]
        return np.sum(scores)

    def get_placement_score(self, board: np.ndarray) -> int:
        """
        Calculate the placement score based on the difference between the current
        and previous board states.

        Args:
            board (numpy.ndarray): Current state of the game board.

        Returns:
            int: Placement score.
        """
        score = 0
        height = self.get_max_height(board)
        for i in range(len(board)):
            diff = np.sum(board[i] - self.board[i])
            score += diff * i
        return score
    
    def get_surface_area(self, board: np.ndarray) -> int:
        """
        Calculate the surface area of the blocks on the board.

        Args:
            board (numpy.ndarray): Current state of the game board.

        Returns:
            int: Surface area.
        """
        area = 0
        for i in range(len(board)):
            for j in range(len(board[i])):
                if board[i][j] == 1:
                    adj = self.get_adjacent(board, i, j)
                    for a in adj:
                        if board[a[0]][a[1]] == 0:
                            area += 1
        return area
    
    def get_board_score(self, board: np.ndarray) -> int:
        """
        Calculate the score of the board based on holes, stack height, and completion.

        Args:
            board (numpy.ndarray): Current state of the game board.

        Returns:
            int: Board score.
        """
        #n = len(board)
        #score_vector = [i / n for i in range(n)]
        #for i in range(len(board)):
        #    current_row = np.sum(board[i]) / len(board[i])
        #    score += current_row * score_vector[i]
        hole_score = self.get_holes_count(board) * -1
        
        height = self.get_max_height(board)
        stack_score = height * -1

        completion_score = 0
        for i in range(len(board)):
            completion = np.sum(board[i]) / len(board[i])
            completion *= i / len(board)
            completion_score += completion

        print("Holes: {}".format(hole_score))
        print("Stack: {}".format(stack_score))
        print("Completion: {}".format(completion_score))
        return hole_score + stack_score
        #return hole_score + stack_score + completion_score