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
from pathlib import Path

from stable_baselines3 import PPO

from agent_trainer import agent_trainer
from tetris_env import tetris_env


def _resolve_device(requested: str) -> str:
    """Resolve --device auto/cpu/cuda with Ubuntu VRAM heuristic.

    On auto, falls back to cpu if CUDA is unavailable or GPU has <4GB VRAM
    (e.g. GTX 960 2GB on Ubuntu 24.04). Pass --device cuda to force.
    """
    if requested != "auto":
        return requested
    try:
        import torch

        if not torch.cuda.is_available():
            return "cpu"
        # Prefer pynvml if present, else torch mem query, else assume cpu-safe
        try:
            import pynvml

            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            mem = pynvml.nvmlDeviceGetMemoryInfo(handle).total
            pynvml.nvmlShutdown()
            if mem < 4 * 1024**3:
                warnings.warn(f"GPU VRAM {mem // 1024**2} MiB <4GB, using cpu (pass --device cuda to force)")
                return "cpu"
        except Exception:
            try:
                free, total = torch.cuda.mem_get_info(0)
                if total < 4 * 1024**3:
                    warnings.warn(f"GPU VRAM {total // 1024**2} MiB <4GB, using cpu (pass --device cuda to force)")
                    return "cpu"
            except Exception:
                pass
        return "auto"
    except ImportError:
        return "cpu"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rom", type=str, default="roms/tetris.gb", help="Path to the ROM file.")
    parser.add_argument("--init", type=str, default="states/init.state", help="Path to the initial state.")
    parser.add_argument("--speedup", type=int, default=5, help="Speedup factor.")
    parser.add_argument("--freq", type=int, default=24, help="Action frequency.")
    parser.add_argument("--policy", type=str, default="MlpPolicy", help="Model policy to use.")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training.")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs for training.")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor for training.")
    parser.add_argument("--steps", type=int, default=2048, help="Number of steps for the model to learn.")
    parser.add_argument("--sessions", type=int, default=40, help="Number of training sessions.")
    parser.add_argument("--runs", type=int, default=4, help="Number of runs per session.")
    parser.add_argument("--model-name", type=str, default="models/tetris_ppo_model", help="Model save path.")
    parser.add_argument(
        "--log-stdout", action=argparse.BooleanOptionalAction, default=True, help="Log session events to stdout."
    )
    parser.add_argument("--log-level", type=str, default="ERROR", help="Logging level.")
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Torch device (auto handles <4GB VRAM).",
    )
    parser.add_argument(
        "--window", type=str, default="null", choices=["null", "SDL2", "headless"], help="PyBoy window backend."
    )
    return parser.parse_args()


def train(args):
    device = _resolve_device(args.device)
    if device != args.device:
        print(f"-> device auto-resolved to '{device}' (requested '{args.device}')")
    env = tetris_env(
        gb_path=args.rom,
        action_freq=args.freq,
        speedup=args.speedup,
        init_state=args.init,
        log_level=args.log_level,
        window=args.window,
    )

    verbose = 1 if args.log_stdout else 0

    if os.path.exists(f"{args.model_name}.zip"):
        try:
            model = PPO.load(args.model_name, env=env, device=device)
            print("-> continuing training!")
        except ValueError as e:
            if "Observation spaces do not match" in str(e):
                print(f"-> saved model incompatible ({e}). Starting fresh training.")
                model = None
            else:
                raise
    else:
        model = None

    if model is None:
        model = PPO(
            policy=args.policy,
            env=env,
            verbose=verbose,
            n_steps=args.steps,
            batch_size=args.batch_size,
            n_epochs=args.epochs,
            gamma=args.gamma,
            device=device,
        )

    trainer = agent_trainer(model)
    trainer.train(sessions=args.sessions, runs_per_session=args.runs)

    Path(args.model_name).parent.mkdir(parents=True, exist_ok=True)
    model.save(args.model_name)


if __name__ == "__main__":
    train(parse_args())
