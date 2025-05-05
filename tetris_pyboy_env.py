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

import gymnasium as gym
from gymnasium import Env, spaces
import numpy as np
from pyboy import PyBoy

actions = ['','a', 'b', 'left', 'right', 'up', 'down', 'start', 'select']

matrix_shape = (16, 20)
game_area_observation_space = spaces.Box(low=0, high=255, shape=matrix_shape, dtype=np.uint8)

class tetris_env(Env):

    def __init__(self, pyboy, debug=False):
        super().__init__()
        self.pyboy = pyboy
        self._fitness=0
        self._previous_fitness=0
        self.debug = debug

        if not self.debug:
            self.pyboy.set_emulation_speed(0)

        self.action_space = spaces.Discrete(len(actions))
        self.observation_space = game_area_observation_space

        self.pyboy.game_wrapper.start_game()

    def step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))

        # Move the agent
        if action == 0:
            pass
        else:
            self.pyboy.button(actions[action])

        # Consider disabling renderer when not needed to improve speed:
        # self.pyboy.tick(1, False)
        self.pyboy.tick(1)

        done = self.pyboy.game_wrapper.game_over

        self._calculate_fitness()
        reward=self._fitness-self._previous_fitness

        observation=self.pyboy.game_area()
        info = {}
        truncated = False

        return observation, reward, done, truncated, info

    def _calculate_fitness(self):
        self._previous_fitness=self._fitness

        # NOTE: Only some game wrappers will provide a score
        # If not, you'll have to investigate how to score the game yourself
        self._fitness=self.pyboy.game_wrapper.score

    def reset(self, **kwargs):
        self.pyboy.game_wrapper.reset_game()
        self._fitness=0
        self._previous_fitness=0

        observation=self.pyboy.game_area()
        info = {}
        return observation, info

    def render(self, mode='human'):
        pass

    def close(self):
        self.pyboy.stop()