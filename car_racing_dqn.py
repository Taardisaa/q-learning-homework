"""
Extra Credit 2: DQN with convolutional layers for Car Racing (pixel-based).

Uses frame stacking (4 grayscale frames) and a classic Atari-style CNN.
Environment: CarRacing-v3 with discrete actions (5 actions).
"""

import gymnasium as gym
import math
import random
import argparse
import os
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count
from tqdm import tqdm

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if device.type == "cuda":
    torch.backends.cudnn.benchmark = True  # auto-tune conv kernels for fixed input size


# ── Environment ──────────────────────────────────────────────────────

def make_env():
    env = gym.make("CarRacing-v3", continuous=False)
    return env


# ── Frame preprocessing ─────────────────────────────────────────────

def preprocess_frame(frame: np.ndarray) -> np.ndarray:
    """Convert RGB 96x96x3 frame to grayscale 84x84, normalized to [0,1]."""
    # Grayscale via luminance weights
    gray = np.dot(frame[:, :, :3], [0.2989, 0.5870, 0.1140])
    # Crop bottom 12 rows (status bar) -> 84x96, then resize to 84x84
    gray = gray[:84, 6:90]  # crop to 84x84 directly
    return gray.astype(np.float32) / 255.0


class FrameStack:
    """Maintains a stack of the last N preprocessed frames."""
    def __init__(self, n_frames: int = 4):
        self.n_frames = n_frames
        self.frames = deque(maxlen=n_frames)

    def reset(self, frame: np.ndarray):
        processed = preprocess_frame(frame)
        for _ in range(self.n_frames):
            self.frames.append(processed)
        return self._get_state()

    def step(self, frame: np.ndarray):
        self.frames.append(preprocess_frame(frame))
        return self._get_state()

    def _get_state(self) -> np.ndarray:
        # Returns shape (n_frames, 84, 84)
        return np.array(self.frames, dtype=np.float32)


# ── Replay Memory ────────────────────────────────────────────────────

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))


class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


# ── CNN DQN ──────────────────────────────────────────────────────────

class CNNDQN(nn.Module):
    """Classic Atari-style CNN DQN: conv layers + fully connected layers."""
    def __init__(self, n_frames: int, n_actions: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(n_frames, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )
        # Compute flattened size: input 4x84x84
        # After conv1: 32 x 20 x 20
        # After conv2: 64 x 9 x 9
        # After conv3: 64 x 7 x 7 = 3136
        self.fc = nn.Sequential(
            nn.Linear(64 * 7 * 7, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions),
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


# ── Hyperparameters ──────────────────────────────────────────────────

BATCH_SIZE = 128
GAMMA = 0.99
EPS_START = 1.0
EPS_END = 0.05
EPS_DECAY = 50000       # decay over steps (not episodes)
TAU = 0.005             # soft target update rate
LR = 1e-4
N_FRAMES = 4
MEMORY_SIZE = 50000
OPTIMIZE_EVERY = 4      # optimize every N env steps
PLOT_EVERY = 10         # plot every N episodes

BASE_DIR = os.path.dirname(__file__)
RESULTS_DIR = os.path.join(BASE_DIR, "results", "car_racing")
os.makedirs(RESULTS_DIR, exist_ok=True)
CHECKPOINT_PATH = os.path.join(RESULTS_DIR, "cnn_dqn.pt")

# ── Args ─────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser()
parser.add_argument("--load", type=str, default=None, metavar="PATH",
                    help="Path to a checkpoint to resume from")
parser.add_argument("--episodes", type=int, default=None,
                    help="Number of episodes to train (default: 500 GPU, 30 CPU)")
args = parser.parse_args()

# ── Setup ────────────────────────────────────────────────────────────

env = make_env()
n_actions = env.action_space.n  # 5 discrete actions
frame_stack = FrameStack(N_FRAMES)

policy_net = CNNDQN(N_FRAMES, n_actions).to(device)
target_net = CNNDQN(N_FRAMES, n_actions).to(device)

optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
memory = ReplayMemory(MEMORY_SIZE)
steps_done = 0
episode_durations = []
episode_rewards = []

if args.load:
    ckpt = torch.load(args.load, map_location=device, weights_only=True)
    policy_net.load_state_dict(ckpt["policy_net"])
    target_net.load_state_dict(ckpt["target_net"])
    optimizer.load_state_dict(ckpt["optimizer"])
    steps_done = ckpt["steps_done"]
    print(f"Loaded checkpoint from {args.load} (steps_done={steps_done})")
elif os.path.exists(CHECKPOINT_PATH):
    ckpt = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=True)
    policy_net.load_state_dict(ckpt["policy_net"])
    target_net.load_state_dict(ckpt["target_net"])
    optimizer.load_state_dict(ckpt["optimizer"])
    steps_done = ckpt["steps_done"]
    episode_durations = ckpt.get("episode_durations", [])
    episode_rewards = ckpt.get("episode_rewards", [])
    print(f"Loaded checkpoint from {CHECKPOINT_PATH} (steps_done={steps_done})")
    print(f"Skipping training — checkpoint already exists. Delete it to retrain.")
else:
    target_net.load_state_dict(policy_net.state_dict())


# ── Action selection ─────────────────────────────────────────────────

def select_action(state):
    global steps_done
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1.0 * steps_done / EPS_DECAY)
    steps_done += 1
    if random.random() > eps_threshold:
        with torch.no_grad():
            return policy_net(state).argmax(dim=1).view(1, 1)
    else:
        return torch.tensor([[env.action_space.sample()]], device=device, dtype=torch.long)


# ── Random baseline evaluation ───────────────────────────────────────

def evaluate_random(num_episodes=50):
    """Run random policy to get baseline duration and reward."""
    env_eval = make_env()
    durations, rewards = [], []
    for _ in range(num_episodes):
        obs, _ = env_eval.reset()
        ep_reward = 0.0
        for t in count():
            action = env_eval.action_space.sample()
            obs, reward, terminated, truncated, _ = env_eval.step(action)
            ep_reward += reward
            if terminated or truncated:
                durations.append(t + 1)
                rewards.append(ep_reward)
                break
    env_eval.close()
    return np.mean(durations), np.mean(rewards)


# ── Plotting ─────────────────────────────────────────────────────────

def plot_training_live():
    """Quick live plot during training (reuses figure 1)."""
    plt.figure(1)
    plt.clf()
    rew = np.array(episode_rewards, dtype=np.float32)
    plt.plot(rew, alpha=0.3, color='steelblue', linewidth=0.8)
    w = 20
    if len(rew) >= w:
        smooth = np.convolve(rew, np.ones(w) / w, mode='valid')
        plt.plot(np.arange(w - 1, len(rew)), smooth, color='navy', linewidth=1.5)
    plt.title('CNN DQN – Car Racing (training)')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.pause(0.001)


def plot_final_chart(ep_rewards, random_reward, save_path):
    """Generate publication-quality final chart."""
    matplotlib.rcParams.update({'font.size': 11})
    rew = np.array(ep_rewards, dtype=np.float64)
    n = len(rew)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
    fig.suptitle('CNN DQN — Car Racing (CarRacing-v3, 5 discrete actions)',
                 fontsize=14, fontweight='bold')

    # ── Left panel: reward over training with smoothing ──
    ax1.plot(rew, alpha=0.15, color='#90CAF9', linewidth=0.6, label='Per episode')
    for w, color, lw, alpha in [(20, '#42A5F5', 1.2, 0.5),
                                 (50, '#1565C0', 2.0, 0.9)]:
        if n >= w:
            smooth = np.convolve(rew, np.ones(w) / w, mode='valid')
            ax1.plot(np.arange(w - 1, n), smooth,
                     color=color, linewidth=lw, alpha=alpha,
                     label=f'{w}-episode moving avg')

    ax1.axhline(random_reward, color='#E53935', linestyle='--', linewidth=1.5,
                label=f'Random policy ({random_reward:.1f})')
    ax1.axhline(0, color='gray', linestyle=':', linewidth=0.8, alpha=0.5)
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Total Reward')
    ax1.set_title('Total Reward Per Episode')
    ax1.legend(loc='upper left', fontsize=9, framealpha=0.9)
    ax1.grid(True, alpha=0.25)

    # ── Right panel: reward distribution (first 100 vs last 100) ──
    first_n = min(100, n // 3)
    last_n = min(100, n // 3)

    first_chunk = rew[:first_n]
    last_chunk = rew[-last_n:]

    bins = np.linspace(min(rew.min(), random_reward - 20),
                       max(rew.max(), 50), 35)

    ax2.hist(first_chunk, bins=bins, alpha=0.5, color='#EF9A9A', edgecolor='#E57373',
             label=f'First {first_n} episodes (μ={first_chunk.mean():.1f})')
    ax2.hist(last_chunk, bins=bins, alpha=0.6, color='#81C784', edgecolor='#66BB6A',
             label=f'Last {last_n} episodes (μ={last_chunk.mean():.1f})')
    ax2.axvline(random_reward, color='#E53935', linestyle='--', linewidth=1.5,
                label=f'Random policy ({random_reward:.1f})')
    ax2.axvline(0, color='gray', linestyle=':', linewidth=0.8, alpha=0.5)
    ax2.set_xlabel('Total Reward')
    ax2.set_ylabel('Count')
    ax2.set_title('Reward Distribution: Early vs Late Training')
    ax2.legend(loc='upper left', fontsize=9, framealpha=0.9)
    ax2.grid(True, alpha=0.25, axis='y')

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Chart saved to {save_path}")
    return fig


# ── Optimize ─────────────────────────────────────────────────────────

def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))

    non_final_mask = torch.tensor(
        tuple(map(lambda s: s is not None, batch.next_state)),
        device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])

    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    state_action_values = policy_net(state_batch).gather(1, action_batch)

    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    with torch.no_grad():
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1).values

    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()


# ── Training loop ────────────────────────────────────────────────────

skip_training = os.path.exists(CHECKPOINT_PATH) and not args.load

if args.episodes:
    num_episodes = args.episodes
elif torch.cuda.is_available():
    print("Using GPU for training.")
    num_episodes = 500
else:
    print("Using CPU for training.")
    num_episodes = 30

if not skip_training:
    for i_episode in tqdm(range(num_episodes), desc="Training"):
        obs, info = env.reset()
        state = frame_stack.reset(obs)
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)

        ep_reward = 0.0
        for t in count():
            action = select_action(state)
            next_obs, reward, terminated, truncated, info = env.step(action.item())
            ep_reward += reward
            reward_tensor = torch.tensor([reward], device=device, dtype=torch.float32)

            done = terminated or truncated
            if done:
                next_state = None
            else:
                next_state = frame_stack.step(next_obs)
                next_state = torch.tensor(next_state, dtype=torch.float32, device=device).unsqueeze(0)

            memory.push(state, action, next_state, reward_tensor)
            state = next_state

            # Optimize and update target net every N steps
            if t % OPTIMIZE_EVERY == 0:
                optimize_model()

                target_net_state_dict = target_net.state_dict()
                policy_net_state_dict = policy_net.state_dict()
                for key in policy_net_state_dict:
                    target_net_state_dict[key] = policy_net_state_dict[key] * TAU + \
                                                 target_net_state_dict[key] * (1 - TAU)
                target_net.load_state_dict(target_net_state_dict)

            if done:
                episode_durations.append(t + 1)
                episode_rewards.append(ep_reward)
                if i_episode % PLOT_EVERY == 0:
                    plot_training_live()
                break

    print('Training complete')

    # Save checkpoint with training metrics
    torch.save({
        "policy_net": policy_net.state_dict(),
        "target_net": target_net.state_dict(),
        "optimizer": optimizer.state_dict(),
        "steps_done": steps_done,
        "episode_durations": episode_durations,
        "episode_rewards": episode_rewards,
    }, CHECKPOINT_PATH)
    print(f"Checkpoint saved to {CHECKPOINT_PATH}")

# Save final chart (always regenerated)
if episode_rewards:
    print("Evaluating random baseline (50 episodes)...")
    rand_dur, rand_rew = evaluate_random(num_episodes=50)
    print(f"  Random baseline => duration: {rand_dur:.0f}, reward: {rand_rew:.1f}")
    chart_path = os.path.join(RESULTS_DIR, "cnn_dqn.png")
    plot_final_chart(episode_rewards, rand_rew, chart_path)
else:
    print("No training metrics available — cannot generate charts.")

env.close()
print(f"Done! All results saved to {RESULTS_DIR}")
plt.ioff()
plt.show()
