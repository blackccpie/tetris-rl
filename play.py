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

import argparse
import os
import warnings

import numpy as np
from stable_baselines3 import PPO

from tetris_env import tetris_env


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rom", type=str, default="roms/tetris.gb", help="Path to ROM.")
    parser.add_argument("--init", type=str, default="states/init.state", help="Path to init state.")
    parser.add_argument("--model", type=str, default="models/tetris_ppo_model", help="Model path.")
    parser.add_argument("--speedup", type=int, default=1, help="Emulator speedup (SDL2 only).")
    parser.add_argument("--freq", type=int, default=24, help="Action frequency.")
    parser.add_argument("--runs", type=int, default=4, help="Number of runs.")
    parser.add_argument("--window", type=str, default="SDL2", choices=["null", "SDL2", "headless"])
    parser.add_argument("--shaped-alpha", type=float, default=0.1, help="PBRS alpha (must match train, 0.0 legacy).")
    parser.add_argument("--log-level", type=str, default="INFO")
    parser.add_argument("--deterministic", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    return parser.parse_args()


def resolve_window(window: str) -> str:
    if window == "SDL2" and not os.environ.get("DISPLAY") and not os.environ.get("WAYLAND_DISPLAY"):
        warnings.warn("No DISPLAY/WAYLAND_DISPLAY; SDL2 may fail. Use --window null or xvfb-run -a")
    return window


if __name__ == "__main__":
    args = parse_args()
    window = resolve_window(args.window)
    device = args.device if args.device != "auto" else "auto"

    env = tetris_env(
        gb_path=args.rom,
        action_freq=args.freq,
        speedup=args.speedup,
        init_state=args.init,
        log_level=args.log_level,
        window=window,
        shaped_alpha=args.shaped_alpha,
    )

    # PPO.load needs custom_objects for TetrisCNN if saved with CnnPolicy
    custom_objects = None
    try:
        from train import TetrisCNN

        custom_objects = {"TetrisCNN": TetrisCNN}
    except Exception:
        pass
    try:
        model = PPO.load(args.model, env=env, device=device, custom_objects=custom_objects)
    except Exception:
        model = PPO.load(args.model, env=env, device=device)

    try:
        for _ in range(args.runs):
            seed = np.random.randint(0, 100000)
            obs, _ = env.reset(seed=seed)
            terminated = False
            steps = 0

            while not terminated:
                # Allow clean exit when SIGINT was delivered inside frame_limiter
                # (patched to set pyboy.quitting instead of raising)
                if getattr(env.pyboy, "quitting", False):
                    raise KeyboardInterrupt
                action, _states = model.predict(obs, deterministic=args.deterministic)
                obs, reward, terminated, _, _ = env.step(action)
                env.render()
                steps += 1
                if getattr(env.pyboy, "quitting", False):
                    raise KeyboardInterrupt
            print(f"Seed: {seed}, Steps: {steps}, Score: {env.get_game_score()}", flush=True)
    except KeyboardInterrupt:
        print("\nInterrupted by user (CTRL+C) — closing emulator...", flush=True)
    finally:
        env.close()
