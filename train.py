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


class TetrisMultiInput(BaseFeaturesExtractor):
    """Multi-input extractor for enriched Dict obs: board CNN + pieces + features.

    board (18,10) uint8 → TetrisCNN 128-dim; next(7)+current(7) → 32-dim;
    features(14) → 32-dim; concat 192 → 128-dim ReLU.
    """

    def __init__(self, observation_space, features_dim: int = 128):
        super().__init__(observation_space, features_dim)
        board_space = observation_space.spaces["board"]
        n_input_channels = 1
        self.cnn = th.nn.Sequential(
            th.nn.Conv2d(n_input_channels, 32, kernel_size=3, stride=1, padding=1),
            th.nn.ReLU(),
            th.nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            th.nn.ReLU(),
            th.nn.Flatten(),
        )
        with th.no_grad():
            sample = th.as_tensor(board_space.sample()[None]).float()
            if sample.ndim == 3:
                sample = sample.unsqueeze(1)
            n_flatten = self.cnn(sample / 8.0).shape[1]
        self.board_fc = th.nn.Sequential(th.nn.Linear(n_flatten, 128), th.nn.ReLU())
        self.piece_fc = th.nn.Sequential(th.nn.Linear(14, 32), th.nn.ReLU())
        self.feat_fc = th.nn.Sequential(th.nn.Linear(14, 32), th.nn.ReLU())
        self.out = th.nn.Sequential(th.nn.Linear(192, features_dim), th.nn.ReLU())

    def forward(self, observations) -> th.Tensor:
        board = observations["board"]
        if board.ndim == 3:
            board = board.unsqueeze(1)
        board = board.float() / 8.0
        b = self.board_fc(self.cnn(board))
        pieces = th.cat([observations["next"].float(), observations["current"].float()], dim=1)
        p = self.piece_fc(pieces)
        f = self.feat_fc(observations["features"].float())
        return self.out(th.cat([b, p, f], dim=1))


def linear_schedule(initial: float):
    """Linear LR/clip decay to ~0 over training (SB3 callable schedule)."""

    def _fn(progress_remaining: float) -> float:
        return max(initial * progress_remaining, 1e-6)

    return _fn


def _check_torch_sm_compat() -> str | None:
    """Return warning if current torch cannot run on this GPU (sm_52 vs sm_75+)."""
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        cap = torch.cuda.get_device_capability(0)
        hw = cap[0] * 10 + cap[1]  # e.g. 52
        arches = torch.cuda.get_arch_list() if hasattr(torch.cuda, "get_arch_list") else []
        # torch 2.7+ lists sm_75+ only; sm_50 covers 5.0-5.9
        # If hw <75 and 'sm_50' not in arches, incompatible
        if hw < 75 and not any(a in ("sm_50", "sm_52", "sm_60") for a in arches):
            # also sm_60 is 6.0 but still <7.5, but 2.7 drops 50/60
            if "sm_50" not in arches:
                return (
                    f"GPU sm_{cap[0]}{cap[1]} (e.g. GTX 960 sm_52) incompatible with torch {torch.__version__} "
                    f"arch {arches} (needs sm_75+). Last sm_52 wheel is 2.6.0+cu124: "
                    f"./scripts/setup_legacy_gpu.sh && uv run --no-sync train.py --device cuda ..."
                )
    except Exception:
        pass
    return None


def _resolve_device(requested: str) -> str:
    """Resolve --device auto/cpu/cuda with Ubuntu VRAM heuristic.

    On auto, falls back to cpu if CUDA is unavailable or GPU has <4GB VRAM
    (e.g. GTX 960 2GB on Ubuntu 24.04). Pass --device cuda to force.
    Also warns if torch is too new for sm_52 (torch>=2.7 drops sm_50).
    """
    # early compat warning even for --device cuda
    msg = _check_torch_sm_compat()
    if msg and requested == "cuda":
        warnings.warn(msg)
    if requested != "auto":
        return requested
    try:
        import torch

        if not torch.cuda.is_available():
            return "cpu"
        # legacy check for auto path
        compat = _check_torch_sm_compat()
        if compat:
            warnings.warn(compat + " — auto→cpu")
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
    parser.add_argument(
        "--freq", type=int, default=8, help="Action frequency (8 finer lateral before lock than legacy 24/12)."
    )
    parser.add_argument(
        "--policy",
        type=str,
        default="MultiInputPolicy",
        choices=["MultiInputPolicy", "CnnPolicy", "MlpPolicy"],
        help="SB3 policy — MultiInputPolicy (enriched Dict, default) or CnnPolicy/MlpPolicy (board only).",
    )
    parser.add_argument(
        "--obs-mode",
        type=str,
        default="enriched",
        choices=["enriched", "board"],
        help="Observation: enriched Dict (board+next+current+features) or board Box (ablation).",
    )
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size (dense reward: 32).")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs.")
    parser.add_argument("--gamma", type=float, default=0.95, help="Discount (dense: 0.95, sparse: 0.99).")
    parser.add_argument("--gae-lambda", type=float, default=0.95, help="GAE lambda (0.9-0.95 robust).")
    parser.add_argument("--vf-coef", type=float, default=0.5, help="Value loss coef (lower if value dominates).")
    parser.add_argument(
        "--clip-range-vf", type=float, default=0.2, help="Value clip (0.2 robust; 0 disables: --clip-range-vf 0)."
    )
    parser.add_argument("--max-grad-norm", type=float, default=0.5, help="Gradient clipping (SB3 default).")
    parser.add_argument("--target-kl", type=float, default=0.02, help="Early-stop PPO epochs if KL exceeds.")
    parser.add_argument(
        "--normalize-advantage",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Normalize advantages per minibatch (robust default true).",
    )
    parser.add_argument(
        "--lr-schedule",
        type=str,
        default="constant",
        choices=["linear", "constant"],
        help="LR schedule (constant recommended: linear restarts every session, spiking KL).",
    )
    parser.add_argument(
        "--clip-schedule", type=str, default="constant", choices=["linear", "constant"], help="Clip decay schedule."
    )
    parser.add_argument("--steps", type=int, default=2048, help="n_steps per rollout.")
    parser.add_argument(
        "--sessions",
        type=int,
        default=200,
        help="Training sessions (200×4×2048≈1.6M steps, 600≈5M, 1200≈10M — ALE 10M still sparse).",
    )
    parser.add_argument("--runs", type=int, default=4, help="Runs per session.")
    parser.add_argument(
        "--model-name", type=str, default="models_robust/tetris_ppo_next", help="Model save path (robust worktree)."
    )
    parser.add_argument("--learning-rate", type=float, default=1e-4, help="Adam LR (dense: 1e-4).")
    parser.add_argument("--clip-range", type=float, default=0.1, help="PPO clip (dense: 0.1).")
    parser.add_argument(
        "--ent-coef",
        type=float,
        default=0.05,
        help="Entropy bonus (0.05 keeps peaked policies exploring; 0.03 early, 0.01 collapsed).",
    )
    parser.add_argument(
        "--shaped-alpha",
        type=float,
        default=0.1,
        help="PBRS alpha for shaped delta (0.0 legacy score-only, 0.1 hybrid).",
    )
    parser.add_argument("--n-envs", type=int, default=1, help="VecEnv count (1=Dummy, >1 Subproc).")
    parser.add_argument("--tensorboard", type=str, default=None, help="TensorBoard log dir.")
    parser.add_argument("--seed", type=int, default=0, help="RNG seed (0=randomized noops only).")
    parser.add_argument("--max-steps", type=int, default=10000, help="Env truncation steps (0=disabled).")
    parser.add_argument("--noop-max", type=int, default=30, help="Random no-op ticks on reset [0, N].")
    parser.add_argument("--eval-freq", type=int, default=0, help="Eval every N timesteps (0=disabled).")
    parser.add_argument("--eval-episodes", type=int, default=5, help="Eval episodes per EvalCallback.")
    parser.add_argument(
        "--best-model-save",
        type=str,
        default=None,
        help="Best-model dir for EvalCallback (default <model-name>_best).",
    )
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
    def _make_single_env():
        return tetris_env(
            gb_path=args.rom,
            action_freq=args.freq,
            speedup=args.speedup,
            init_state=args.init,
            log_level=args.log_level,
            window=args.window,
            shaped_alpha=args.shaped_alpha,
            obs_mode=args.obs_mode,
            max_steps=args.max_steps,
            noop_max=args.noop_max,
        )

    if args.n_envs > 1:
        from stable_baselines3.common.env_util import make_vec_env
        from stable_baselines3.common.vec_env import SubprocVecEnv

        env = make_vec_env(_make_single_env, n_envs=args.n_envs, vec_env_cls=SubprocVecEnv)
        # PPO n_steps is per-env, total = n_steps * n_envs
    else:
        env = _make_single_env()

    verbose = 1 if args.log_stdout else 0

    # Policy kwargs — TetrisCNN for board, TetrisMultiInput for enriched Dict
    policy_kwargs = None
    if args.policy == "CnnPolicy":
        policy_kwargs = dict(
            features_extractor_class=TetrisCNN,
            features_extractor_kwargs=dict(features_dim=128),
            normalize_images=False,  # we already /8 in extractor
        )
    elif args.policy == "MultiInputPolicy":
        policy_kwargs = dict(
            features_extractor_class=TetrisMultiInput,
            features_extractor_kwargs=dict(features_dim=128),
            normalize_images=False,
        )
    if args.obs_mode == "enriched" and args.policy != "MultiInputPolicy":
        print(f"-> obs_mode enriched needs MultiInputPolicy, got {args.policy}; switching to MultiInputPolicy.")
        args.policy = "MultiInputPolicy"
        policy_kwargs = dict(
            features_extractor_class=TetrisMultiInput,
            features_extractor_kwargs=dict(features_dim=128),
            normalize_images=False,
        )
    if args.obs_mode == "board" and args.policy == "MultiInputPolicy":
        print("-> obs_mode board with MultiInputPolicy is invalid; switching to CnnPolicy.")
        args.policy = "CnnPolicy"
        policy_kwargs = dict(
            features_extractor_class=TetrisCNN,
            features_extractor_kwargs=dict(features_dim=128),
            normalize_images=False,
        )

    # PPO loss schedules — linear decay stabilizes late training
    lr = linear_schedule(args.learning_rate) if args.lr_schedule == "linear" else args.learning_rate
    clip = linear_schedule(args.clip_range) if args.clip_schedule == "linear" else args.clip_range
    clip_vf = None if args.clip_range_vf == 0 else args.clip_range_vf
    target_kl = None if args.target_kl == 0 else args.target_kl
    tb_log = args.tensorboard
    if tb_log:
        try:
            import tensorboard  # noqa: F401

            Path(tb_log).mkdir(parents=True, exist_ok=True)
        except ImportError:
            print("-> tensorboard not installed, disabling tb logging.")
            tb_log = None

    custom_objects = {"TetrisCNN": TetrisCNN, "TetrisMultiInput": TetrisMultiInput}
    resumed = False
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
            elif hasattr(model, "action_space") and model.action_space.n != env.action_space.n:
                print(
                    f"-> saved model action_space {model.action_space.n} != env {env.action_space.n} "
                    f"(5 actions LEFT/RIGHT/DOWN/A/B; UP/PASS removed). Starting fresh — archive old "
                    f"with: mv {args.model_name}.zip {args.model_name}_legacy{model.action_space.n}.zip"
                )
                model = None
            else:
                print("-> continuing training!")
                resumed = True
        except ValueError as e:
            msg = str(e)
            if (
                "Observation spaces do not match" in msg
                or "features_extractor" in msg
                or "action_space" in msg
                or "Discrete" in msg
            ):
                print(f"-> saved model incompatible ({e}). Starting fresh training.")
                model = None
            else:
                raise
        except Exception as e:
            msg = str(e)
            # SB3 may raise different error for policy/action mismatch
            if (
                "TetrisCNN" in msg
                or "TetrisMultiInput" in msg
                or "CnnPolicy" in msg
                or "MlpPolicy" in msg
                or "MultiInputPolicy" in msg
                or "action_space" in msg
                or "Discrete" in msg
                or "Dict" in msg
            ):
                print(f"-> saved model policy/obs/action mismatch ({e}). Starting fresh.")
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
            gae_lambda=args.gae_lambda,
            learning_rate=lr,
            clip_range=clip,
            clip_range_vf=clip_vf,
            normalize_advantage=args.normalize_advantage,
            ent_coef=args.ent_coef,
            vf_coef=args.vf_coef,
            max_grad_norm=args.max_grad_norm,
            target_kl=target_kl,
            device=device,
            policy_kwargs=policy_kwargs,
            tensorboard_log=tb_log,
            seed=args.seed if args.seed != 0 else None,
        )
    elif resumed:
        # PPO.load keeps stored hyperparams — re-apply CLI values so segmented
        # runs can tune without restarting (optimizer state is preserved).
        from stable_baselines3.common.utils import FloatSchedule

        model.ent_coef = args.ent_coef
        model.vf_coef = args.vf_coef
        model.gamma = args.gamma
        model.gae_lambda = args.gae_lambda
        model.max_grad_norm = args.max_grad_norm
        model.target_kl = target_kl
        model.normalize_advantage = args.normalize_advantage
        model.clip_range = FloatSchedule(clip)  # float or schedule both accepted
        model.learning_rate = lr
        model._setup_lr_schedule()
        print(
            f"-> applied CLI hyperparams to resumed model "
            f"(ent={args.ent_coef} vf={args.vf_coef} lr={args.learning_rate}/{args.lr_schedule} "
            f"clip={args.clip_range} kl={args.target_kl} gamma={args.gamma} gae={args.gae_lambda})"
        )

    # Optional EvalCallback — best-model tracking for robustness
    eval_callback = None
    if int(args.eval_freq) > 0:
        from stable_baselines3.common.callbacks import EvalCallback
        from stable_baselines3.common.vec_env import DummyVecEnv

        best_dir = args.best_model_save or f"{args.model_name}_best"
        Path(best_dir).mkdir(parents=True, exist_ok=True)
        eval_env = DummyVecEnv([_make_single_env])
        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path=best_dir,
            eval_freq=int(args.eval_freq),
            n_eval_episodes=int(args.eval_episodes),
            deterministic=True,
            verbose=1,
        )
        print(f"-> EvalCallback every {args.eval_freq} steps, best → {best_dir}/best_model.zip")

    # Resumable segmented training — saves every checkpoint_freq sessions
    # so you can run long CPU trainings as multiple short segments:
    #   uv run train.py --sessions 50  # segment 1 (50×4×2048≈400k)
    #   uv run train.py --sessions 50  # segment 2 resumes from models_robust/tetris_ppo_next.zip
    # For a single 5M run split into 10×500k, just run 10 times or use checkpoint-freq.
    checkpoint_freq = int(args.checkpoint_freq)
    Path(args.model_name).parent.mkdir(parents=True, exist_ok=True)

    try:
        # Manual loop to checkpoint; delegate single-step to agent_trainer for tqdm consistency
        # but checkpoint every N sessions to survive CTRL+C / power loss.
        # reset_num_timesteps=False: the zip counter accumulates across sessions and
        # invocations (default True resets it every learn() call, hiding progress).
        # NOTE: with reset=False, a linear LR schedule would collapse to ~0 after the
        # first sessions (each call starts at X/(done+X)) — use constant (default).
        if args.lr_schedule == "linear" and (args.sessions > 1 or resumed):
            print("-> warning: linear LR restarts every session; prefer --lr-schedule constant for segments.")
        for i in tqdm(range(args.sessions), desc="sessions"):
            model.learn(
                total_timesteps=model.n_steps * args.runs,
                callback=eval_callback,
                reset_num_timesteps=False,
            )
            # periodic checkpoint (and always on last session)
            is_checkpoint = checkpoint_freq > 0 and ((i + 1) % checkpoint_freq == 0)
            is_last = (i + 1) == args.sessions
            if is_checkpoint or is_last:
                model.save(args.model_name)
                if is_checkpoint and not is_last:
                    # also keep numbered copy for rollback: models_robust/tetris_ppo_next_ckpt_0010.zip
                    ckpt = f"{args.model_name}_ckpt_{(i + 1):04d}"
                    try:
                        model.save(ckpt)
                    except Exception:
                        pass
                    print(f"-> checkpoint saved {ckpt}.zip ({i + 1}/{args.sessions})")
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
            if "model" in locals() and model is not None:
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
