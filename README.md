# Reinforcement learning demos

Runnable Rainbow DQN and Stable-Baselines3 examples with training, model saving/loading,
and evaluation on separate episodes:

| Demo | Environment | Actions | Agent |
| --- | --- | --- | --- |
| Atari | `ALE/Pong-v5` | Discrete joystick/button choices | PPO with a CNN |
| Rainbow Atari | `ALE/Pong-v5` | Discrete joystick/button choices | Rainbow DQN with a CNN |
| Continuous control | `Pendulum-v1` | Continuous torque in `[-2, 2]` | SAC with an MLP |

Atari games normally have discrete action spaces. Pendulum demonstrates true
continuous control without implying that Pong accepts continuous torque.
**Rainbow DQN** is implemented in `rainbow.py` using PyTorch, since standard
Stable-Baselines3 does not implement Rainbow. SAC handles continuous actions.

## Rainbow DQN

```bash
.venv/bin/python rainbow.py --steps 1000000
.venv/bin/python rainbow.py --load runs/rainbow_pong.pt --episodes 3
# Append --render to watch evaluation on a desktop machine.
```

This educational implementation combines Double DQN, dueling value/advantage
streams, proportional prioritized replay with importance weights, three-step
returns, C51 distributional learning (51 atoms on [-10, 10]), and factorized
NoisyNet exploration. Evaluation disables noise and reports unclipped rewards
over full games; training clips rewards. Truncated episodes bootstrap from the
final observation, while terminal episodes do not.

Defaults run on CPU with a 10,000-transition replay buffer (roughly 540 MiB for
stored observations, plus model and runtime memory). Replay sampling scans the
buffer rather than using a sum tree, keeping the demo readable. These reduced
settings are not an Atari benchmark reproduction. Full learning needs much
longer runs. Checkpoints contain model weights, not optimizer/replay state, and
`--load` evaluates rather than resumes training.

```bash
.venv/bin/python -m unittest -v test_rainbow
.venv/bin/python rainbow.py --steps 80 --learning-starts 32 --batch-size 8 --buffer-size 100 --episodes 1
```

The short command must print a positive gradient-update count, save the model,
and complete a game. It checks execution, not learned skill.

## Install

Use Python 3.10–3.12 (validated with Python 3.12):

```bash
cd /workspace/RL-demo
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

For a smaller CPU-only PyTorch installation, run this before installing the
requirements: `.venv/bin/python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu`.

Modern ALE wheels include Atari ROMs; this version does not require the old
AutoROM download step. No API keys or external services are needed. Installation
requires access to PyPI and its package download hosts.

## Train

```bash
# Discrete Atari: image observations, grayscale/resize, four stacked frames.
.venv/bin/python demo.py atari --steps 1000000

# Continuous control: observations are angle cosine/sine and angular velocity.
.venv/bin/python demo.py continuous --steps 50000
```

Both commands run headlessly on CPU, save under `runs/`, and report evaluation
return. Atari can require millions of steps and considerable time to learn;
these settings are demonstrations, not benchmark-tuned guarantees of good play.
PPO rounds training up to a complete rollout (128 steps here). Pendulum's reward
is negative; values closer to zero indicate better control.

## Load and watch

```bash
.venv/bin/python demo.py atari --load runs/atari_ppo.zip --episodes 3
.venv/bin/python demo.py continuous --load runs/continuous_sac.zip --episodes 3
# On a machine with a desktop display, append --render to watch evaluation.
```

Use the matching task when loading a model. Rendering requires a display and is
optional; training and evaluation work without one.

## Quick functional checks

```bash
.venv/bin/python demo.py continuous --steps 1100 --episodes 1
.venv/bin/python demo.py atari --steps 128 --episodes 1
```

These exercise environment creation, learning, saving, and complete-episode
evaluation. They validate execution, not learned playing skill.
