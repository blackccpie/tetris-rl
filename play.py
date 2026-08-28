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

import numpy as np
from stable_baselines3 import PPO

from tetris_env import tetris_env

if __name__ == "__main__":
    runs = 4
    init_state = "states/init.state"

    env = tetris_env(
        gb_path="roms/tetris.gb", action_freq=24, speedup=1, init_state=init_state, log_level="INFO", window="SDL2"
    )

    model = PPO.load("models/tetris_ppo_model", env=env)

    for _ in range(runs):
        seed = np.random.randint(0, 100000)
        obs, _ = env.reset(seed=seed)
        terminated = False
        steps = 0

        while not terminated:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, _, _ = env.step(action)
            env.render()
            steps += 1
        print(f"Seed: {seed}, Steps: {steps}, Score: {env.get_game_score()}")
