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

import torch as th
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from tqdm import tqdm

from tetris_env import tetris_env


class TetrisCNN(BaseFeaturesExtractor):
    """Small CNN for 18×10 board — NatureCNN (8×8) too large for 18×10.

    Handles both (18,10) and (1,18,10) observations; normalizes 0-8 → 0-1.
    3×3 kernels preserve spatial locality for holes/bumpiness/height.
    SB3 docs: CnnPolicy uses NatureCNN for 84×84; we use 32→64 channels
    and 128-dim output, matching donnybadamo/Mini-Tetris findings where
    CNN > MLP for board.
    """

    def __init__(self, observation_space: spaces.Box, features_dim: int = 128):
        super().__init__(observation_space, features_dim)
        # infer n_input_channels: 1 for (18,10) or (1,18,10)
        n_input_channels = 1
        self.cnn = th.nn.Sequential(
            th.nn.Conv2d(n_input_channels, 32, kernel_size=3, stride=1, padding=1),
            th.nn.ReLU(),
            th.nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            th.nn.ReLU(),
            th.nn.Flatten(),
        )
        with th.no_grad():
            sample = th.as_tensor(observation_space.sample()[None]).float()
            if sample.ndim == 3:  # (N,H,W) → (N,C,H,W)
                sample = sample.unsqueeze(1)
            # normalize like forward will
            sample = sample / 8.0
            n_flatten = self.cnn(sample).shape[1]
        self.linear = th.nn.Sequential(th.nn.Linear(n_flatten, features_dim), th.nn.ReLU())

    def forward(self, observations: th.Tensor) -> th.Tensor:
        # observations: (B,18,10) or (B,1,18,10) uint8 0-8
        if observations.ndim == 3:
            observations = observations.unsqueeze(1)
        observations = observations.float() / 8.0
        return self.linear(self.cnn(observations))


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
    parser.add_argument(
        "--policy",
        type=str,
        default="CnnPolicy",
        choices=["CnnPolicy", "MlpPolicy"],
        help="SB3 policy — CnnPolicy (TetrisCNN 18×10) recommended per 2024-25 Tetris RL (CNN>MLP).",
    )
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size (dense reward: 32).")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs.")
    parser.add_argument("--gamma", type=float, default=0.95, help="Discount (dense: 0.95, sparse: 0.99).")
    parser.add_argument("--steps", type=int, default=2048, help="n_steps per rollout.")
    parser.add_argument(
        "--sessions",
        type=int,
        default=200,
        help="Training sessions (200×4×2048≈1.6M steps, 600≈5M, 1200≈10M — ALE 10M still sparse).",
    )
    parser.add_argument("--runs", type=int, default=4, help="Runs per session.")
    parser.add_argument("--model-name", type=str, default="models/tetris_ppo_model", help="Model save path.")
    parser.add_argument("--learning-rate", type=float, default=1e-4, help="Adam LR (dense: 1e-4).")
    parser.add_argument("--clip-range", type=float, default=0.1, help="PPO clip (dense: 0.1).")
    parser.add_argument("--ent-coef", type=float, default=0.01, help="Entropy bonus (dense: 0.01).")
    parser.add_argument(
        "--shaped-alpha",
        type=float,
        default=0.1,
        help="PBRS alpha for shaped delta (0.0 legacy score-only, 0.1 hybrid).",
    )
    parser.add_argument("--n-envs", type=int, default=1, help="VecEnv count (1=Dummy, >1 Subproc).")
    parser.add_argument("--tensorboard", type=str, default=None, help="TensorBoard log dir.")
    parser.add_argument(
        "--checkpoint-freq",
        type=int,
        default=10,
        help="Save checkpoint every N sessions (0=only at end).",
    )
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
    # VecEnv for throughput — SB3 docs: SubprocVecEnv for CPU PPO
    if args.n_envs > 1:
        from stable_baselines3.common.env_util import make_vec_env
        from stable_baselines3.common.vec_env import SubprocVecEnv

        def _make_env():
            return tetris_env(
                gb_path=args.rom,
                action_freq=args.freq,
                speedup=args.speedup,
                init_state=args.init,
                log_level=args.log_level,
                window=args.window,
                shaped_alpha=args.shaped_alpha,
            )

        env = make_vec_env(_make_env, n_envs=args.n_envs, vec_env_cls=SubprocVecEnv)
        # PPO n_steps is per-env, total = n_steps * n_envs
    else:
        env = tetris_env(
            gb_path=args.rom,
            action_freq=args.freq,
            speedup=args.speedup,
            init_state=args.init,
            log_level=args.log_level,
            window=args.window,
            shaped_alpha=args.shaped_alpha,
        )

    verbose = 1 if args.log_stdout else 0

    # Policy kwargs for CnnPolicy — TetrisCNN small for 18×10 vs Nature 84×84
    policy_kwargs = None
    if args.policy == "CnnPolicy":
        policy_kwargs = dict(
            features_extractor_class=TetrisCNN,
            features_extractor_kwargs=dict(features_dim=128),
            normalize_images=False,  # we already /8 in extractor
        )

    custom_objects = {"TetrisCNN": TetrisCNN}
    if os.path.exists(f"{args.model_name}.zip"):
        try:
            model = PPO.load(args.model_name, env=env, device=device, custom_objects=custom_objects)
            # If policy mismatches requested (e.g. Mlp→Cnn), start fresh for architecture change
            loaded_policy = model.policy.__class__.__name__ if hasattr(model, "policy") else ""
            # CnnPolicy loads as ActorCriticCnnPolicy, Mlp as ActorCriticPolicy
            wants_cnn = args.policy == "CnnPolicy"
            is_cnn = "Cnn" in loaded_policy
            if wants_cnn != is_cnn:
                print(f"-> saved model {loaded_policy} != requested {args.policy}, starting fresh.")
                model = None
            else:
                print("-> continuing training!")
        except ValueError as e:
            if "Observation spaces do not match" in str(e) or "features_extractor" in str(e):
                print(f"-> saved model incompatible ({e}). Starting fresh training.")
                model = None
            else:
                raise
        except Exception as e:
            # SB3 may raise different error for policy mismatch
            if "TetrisCNN" in str(e) or "CnnPolicy" in str(e) or "MlpPolicy" in str(e):
                print(f"-> saved model policy mismatch ({e}). Starting fresh.")
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
            learning_rate=args.learning_rate,
            clip_range=args.clip_range,
            ent_coef=args.ent_coef,
            device=device,
            policy_kwargs=policy_kwargs,
            tensorboard_log=args.tensorboard,
        )

    # Resumable segmented training — saves every checkpoint_freq sessions
    # so you can run long CPU trainings as multiple short segments:
    #   uv run train.py --sessions 50  # segment 1 (50×4×2048≈400k)
    #   uv run train.py --sessions 50  # segment 2 resumes from models/tetris_ppo_model.zip
    # For a single 5M run split into 10×500k, just run 10 times or use checkpoint-freq.
    checkpoint_freq = int(args.checkpoint_freq)
    Path(args.model_name).parent.mkdir(parents=True, exist_ok=True)

    try:
        # Manual loop to checkpoint; delegate single-step to agent_trainer for tqdm consistency
        # but checkpoint every N sessions to survive CTRL+C / power loss
        for i in tqdm(range(args.sessions), desc="sessions"):
            model.learn(total_timesteps=model.n_steps * args.runs)
            # periodic checkpoint (and always on last session)
            is_checkpoint = checkpoint_freq > 0 and ((i + 1) % checkpoint_freq == 0)
            is_last = (i + 1) == args.sessions
            if is_checkpoint or is_last:
                model.save(args.model_name)
                if is_checkpoint and not is_last:
                    # also keep numbered copy for rollback: models/tetris_ppo_model_ckpt_10.zip
                    ckpt = f"{args.model_name}_ckpt_{(i + 1):04d}"
                    try:
                        model.save(ckpt)
                    except Exception:
                        pass
                    print(f"-> checkpoint saved {ckpt}.zip ({i+1}/{args.sessions})")
    except KeyboardInterrupt:
        print("\nInterrupted — saving checkpoint...")
        try:
            model.save(args.model_name)
            print(f"-> saved {args.model_name}.zip — resume with same command")
        except Exception as e:
            print(f"-> save failed: {e}")
        raise
    finally:
        # ensure final save even on normal exit (already done) and close env
        try:
            if 'model' in locals() and model is not None:
                # non-checkpoint final save already done; ensure at least once
                if not Path(f"{args.model_name}.zip").exists():
                    model.save(args.model_name)
        except Exception:
            pass
        try:
            env.close()
        except Exception:
            pass


if __name__ == "__main__":
    train(parse_args())
