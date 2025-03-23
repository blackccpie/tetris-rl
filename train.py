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

import asyncio
import argparse

from stable_baselines3 import PPO

from stable_baselines3.common.logger import Logger, make_output_format

from agent_trainer import agent_trainer
from tetris_env import tetris_env

def parse_args():
    # Parse the command line arguments
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
    parser.add_argument("--log-stdout", type=bool, default=True, help="Log session events to stdout.")
    parser.add_argument("--log-level", type=str, default="ERROR", help="Logging level.")
    return parser.parse_args()

async def train(args):
    
    # configure the environment, model, and trainer
    env = tetris_env(gb_path=args.rom, action_freq=args.freq, speedup=args.speedup, init_state=args.init, log_level=args.log_level, window="null")
    model = PPO(policy=args.policy, env=env, verbose=1, n_steps=args.steps, batch_size=args.batch_size, n_epochs=args.epochs, gamma=args.gamma)
    trainer = agent_trainer(model)

    # Set logging outputs
    output_formats = []
    if args.log_stdout:
        output_formats.append(make_output_format("stdout", "sessions"))
    model.set_logger(Logger(None, output_formats=output_formats))

    await trainer.train(sessions=args.sessions, runs_per_session=args.runs)

    model.save('models/tetris_ppo_model')

if __name__ == "__main__":
    asyncio.run(train(parse_args()))
