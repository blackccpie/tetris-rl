# tetris-rl

➖➖➖➖➖➖➖🟦🟦➖  
➖➖➖➖➖➖➖🟦➖➖  
➖➖➖➖➖➖➖🟦➖➖  
➖➖➖➖➖➖➖➖➖➖  
➖➖🟩🟩➖🟧➖➖➖➖  
➖🟩🟩🟧🟧🟧➖➖➖➖  
➖🟪🟪🟪🟦🟦🟦➖➖➖  
🟨🟨🟪🟪🟥🟥🟦➖🟨🟨  
🟨🟨🟪🟪🟪🟥🟥➖🟨🟨  
⬜️⬜️⬜️⬜️⬜️⬜️⬜️⬜️⬜️⬜️  
⬜️⬜️⬜️⬜️⬜️⬜️⬜️⬜️⬜️⬜️  

Playing around with Tetris and Reinforcement Learning

## Inspirations

The initial code is mainly inspired by this project:
* [blog post](https://rotational.io/blog/reinforcement-learning-automation-and-tetris)  
* [source code](https://github.com/pdeziel/ai-tetris)

However some of the metrics I used are rather taken from this project:
* [blog post](https://codemyroad.wordpress.com/2013/04/14/tetris-ai-the-near-perfect-player)
* [source code](https://github.com/LeeYiyuan/tetrisai)

## Requirements

- Python 3.12 (`pyproject.toml:8`, `.python-version`)
- `uv` — see [uv docs](https://docs.astral.sh/uv/)
- ROM: `roms/tetris.gb` (not tracked; provide your own legal dump)

## Setup

```bash
# System deps
# Ubuntu 24.04 (Noble) — runtime only needs SDL2:
sudo apt update && sudo apt install -y libsdl2-2.0-0
# only needed to build PyBoy from source:
# sudo apt install -y libsdl2-dev

# macOS (brew):
# brew install sdl2

# Install Python deps
uv sync --extra dev          # or: uv sync (runtime only)
# If network is slow or GPU has <4GB VRAM, use CPU-only torch:
# uv sync --no-install-package torch && uv pip install torch --index-url https://download.pytorch.org/whl/cpu

# Provide ROM and create output dirs
mkdir -p roms models
# copy your dump → roms/tetris.gb
ls -lh roms/tetris.gb states/init.state  # init.state is PyBoy 2.7.0 save (already in repo)
```

`states/init.state` was generated with PyBoy 2.7.0 by pressing START after ROM load (`tetris_env.py:139`); regenerate if PyBoy version changes.

## Usage

```bash
uv run pytest tests/ -v          # headless, uses window="null"
uv run ruff check . && uv run ruff format --check .

uv run train.py --help           # trains PPO; saves to models/tetris_ppo_model.zip
uv run train.py --device cpu     # recommended on Ubuntu with <4GB VRAM (e.g. GTX 960 2GB)

uv run play.py --help            # loads model and plays with SDL2 window
# headless / CI (no DISPLAY):
xvfb-run -a uv run play.py --window null
# or: SDL_VIDEODRIVER=dummy uv run train.py --window null
```

### Window backends (`tetris_env.py:137`)

- `window="null"` → `set_emulation_speed(0)` unlimited (training, CI)
- `window="SDL2"` → visible window, respects `--speedup` (play); requires `DISPLAY` (`:1` on Ubuntu desktop / X11)

## References

- https://github.com/Baekalfen/PyBoy/wiki/Installation  
- https://github.com/Baekalfen/PyBoy/wiki/Example-Tetris
