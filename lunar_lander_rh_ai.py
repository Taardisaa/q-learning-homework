"""
Part 2: DQN for Lunar Lander (LunarLander-v3)

Trains two DQN variants on the same environment and compares them:
  1. MLP DQN  – learns from the 8-dim state vector (default observation)
  2. CNN DQN  – learns from pixel observations (rgb_array render mode)

Results (charts + checkpoints) are saved to results/lunar_lander/.
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

BASE_DIR = os.path.dirname(__file__)
RESULTS_DIR = os.path.join(BASE_DIR, "results", "lunar_lander")
os.makedirs(RESULTS_DIR, exist_ok=True)


# ── Environments ─────────────────────────────────────────────────────

def make_env_state():
    """Lunar Lander with state-vector observations."""
    return gym.make("LunarLander-v3", continuous=False, gravity=-10.0,
                    enable_wind=False, wind_power=15.0, turbulence_power=1.5)


def make_env_pixel():
    """Lunar Lander with pixel observations."""
    return gym.make("LunarLander-v3", continuous=False, gravity=-10.0,
                    enable_wind=False, wind_power=15.0, turbulence_power=1.5,
                    render_mode="rgb_array")


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


# ── MLP DQN ──────────────────────────────────────────────────────────

class MLPDQN(nn.Module):
    def __init__(self, n_observations, n_actions):
        super().__init__()
        self.fc1 = nn.Linear(n_observations, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, n_actions)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


# ── CNN DQN ──────────────────────────────────────────────────────────

class CNNDQN(nn.Module):
    """Atari-style CNN DQN operating on stacked grayscale frames."""
    def __init__(self, n_frames, n_actions):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(n_frames, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )
        # Input 4x84x84 -> conv1: 32x20x20 -> conv2: 64x9x9 -> conv3: 64x7x7 = 3136
        self.fc = nn.Sequential(
            nn.Linear(64 * 7 * 7, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions),
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


# ── Frame preprocessing (for CNN) ───────────────────────────────────

def preprocess_frame(frame: np.ndarray) -> np.ndarray:
    """Convert RGB pixel frame to 84x84 grayscale, normalized to [0,1]."""
    gray = np.dot(frame[:, :, :3], [0.2989, 0.5870, 0.1140]).astype(np.float32)
    # Resize via torch interpolate (fast, no opencv needed)
    t = torch.from_numpy(gray).unsqueeze(0).unsqueeze(0)  # 1x1xHxW
    t = F.interpolate(t, size=(84, 84), mode='bilinear', align_corners=False)
    return t.squeeze().numpy() / 255.0


class FrameStack:
    def __init__(self, n_frames=4):
        self.n_frames = n_frames
        self.frames = deque(maxlen=n_frames)

    def reset(self, frame):
        processed = preprocess_frame(frame)
        for _ in range(self.n_frames):
            self.frames.append(processed)
        return np.array(self.frames, dtype=np.float32)

    def step(self, frame):
        self.frames.append(preprocess_frame(frame))
        return np.array(self.frames, dtype=np.float32)


# ── Hyperparameters ──────────────────────────────────────────────────

BATCH_SIZE = 128
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.01
EPS_DECAY = 2500
TAU = 0.005
LR = 3e-4
N_FRAMES = 4


# ── Generic training function ────────────────────────────────────────

def train_dqn(label, make_env_fn, build_net_fn, state_transform_fn,
              num_episodes, checkpoint_name):
    """
    Train a DQN agent and return (episode_durations, episode_rewards).

    Parameters
    ----------
    label : str               – name for progress bar / prints
    make_env_fn : callable    – returns a gymnasium env
    build_net_fn : callable   – returns (policy_net, target_net) on device
    state_transform_fn : callable(env, obs, reset) – converts raw obs to tensor
    num_episodes : int
    checkpoint_name : str     – filename for the checkpoint
    """
    env = make_env_fn()
    policy_net, target_net = build_net_fn()
    optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
    memory = ReplayMemory(100000)
    steps_done = 0

    # Skip training if checkpoint already exists
    ckpt_path = os.path.join(RESULTS_DIR, checkpoint_name)
    if os.path.exists(ckpt_path):
        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=True)
        policy_net.load_state_dict(checkpoint["policy_net"])
        target_net.load_state_dict(checkpoint["target_net"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        steps_done = checkpoint["steps_done"]
        print(f"  Loaded checkpoint from {ckpt_path} (steps_done={steps_done})")
        print(f"  Skipping training — checkpoint already exists. Delete it to retrain.")
        env.close()
        return [], []

    ep_durations = []
    ep_rewards = []

    def select_action(state):
        nonlocal steps_done
        eps = EPS_END + (EPS_START - EPS_END) * math.exp(-1.0 * steps_done / EPS_DECAY)
        steps_done += 1
        if random.random() > eps:
            with torch.no_grad():
                return policy_net(state).argmax(dim=1).view(1, 1)
        return torch.tensor([[env.action_space.sample()]], device=device, dtype=torch.long)

    def optimize():
        if len(memory) < BATCH_SIZE:
            return
        transitions = memory.sample(BATCH_SIZE)
        batch = Transition(*zip(*transitions))
        non_final_mask = torch.tensor(
            tuple(s is not None for s in batch.next_state),
            device=device, dtype=torch.bool)
        non_final_next = torch.cat([s for s in batch.next_state if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        q_vals = policy_net(state_batch).gather(1, action_batch)
        next_vals = torch.zeros(BATCH_SIZE, device=device)
        with torch.no_grad():
            next_vals[non_final_mask] = target_net(non_final_next).max(1).values
        expected = (next_vals * GAMMA) + reward_batch

        loss = F.smooth_l1_loss(q_vals, expected.unsqueeze(1))
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
        optimizer.step()

    for ep in tqdm(range(num_episodes), desc=label):
        obs, info = env.reset()
        state = state_transform_fn(env, obs, reset=True)
        ep_reward = 0.0

        for t in count():
            action = select_action(state)
            next_obs, reward, terminated, truncated, info = env.step(action.item())
            ep_reward += reward
            reward_t = torch.tensor([reward], device=device, dtype=torch.float32)

            done = terminated or truncated
            if done:
                next_state = None
            else:
                next_state = state_transform_fn(env, next_obs, reset=False)

            memory.push(state, action, next_state, reward_t)
            state = next_state
            optimize()

            # Soft update target network
            tgt_sd = target_net.state_dict()
            pol_sd = policy_net.state_dict()
            for key in pol_sd:
                tgt_sd[key] = pol_sd[key] * TAU + tgt_sd[key] * (1 - TAU)
            target_net.load_state_dict(tgt_sd)

            if done:
                ep_durations.append(t + 1)
                ep_rewards.append(ep_reward)
                break

    env.close()

    # Save checkpoint
    ckpt_path = os.path.join(RESULTS_DIR, checkpoint_name)
    torch.save({
        "policy_net": policy_net.state_dict(),
        "target_net": target_net.state_dict(),
        "optimizer": optimizer.state_dict(),
        "steps_done": steps_done,
    }, ckpt_path)
    print(f"  Checkpoint saved to {ckpt_path}")

    return ep_durations, ep_rewards


# ── Plotting ─────────────────────────────────────────────────────────

def plot_single(durations, rewards, title, filename):
    """Save a 2-panel (duration + reward) chart for one agent."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(title)

    dur_t = torch.tensor(durations, dtype=torch.float)
    ax1.set_xlabel('Episode'); ax1.set_ylabel('Duration')
    ax1.plot(dur_t.numpy(), color='green', alpha=0.6, label='raw')
    if len(dur_t) >= 20:
        m = dur_t.unfold(0, 20, 1).mean(1).view(-1)
        m = torch.cat((torch.zeros(19), m))
        ax1.plot(m.numpy(), color='red', label='20-ep avg')
    ax1.legend()

    rew_t = torch.tensor(rewards, dtype=torch.float)
    ax2.set_xlabel('Episode'); ax2.set_ylabel('Total Reward')
    ax2.plot(rew_t.numpy(), color='green', alpha=0.6, label='raw')
    if len(rew_t) >= 20:
        m = rew_t.unfold(0, 20, 1).mean(1).view(-1)
        m = torch.cat((torch.zeros(19), m))
        ax2.plot(m.numpy(), color='red', label='20-ep avg')
    ax2.legend()

    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  Chart saved to {path}")
    plt.close(fig)


def plot_comparison(all_results):
    """Save a combined reward comparison chart for all agents."""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_title("Lunar Lander: MLP DQN vs CNN DQN – Reward")
    ax.set_xlabel("Episode"); ax.set_ylabel("Total Reward (20-ep avg)")

    for label, _, rewards in all_results:
        rew_t = torch.tensor(rewards, dtype=torch.float)
        if len(rew_t) >= 20:
            m = rew_t.unfold(0, 20, 1).mean(1).view(-1)
            ax.plot(m.numpy(), label=label)
        else:
            ax.plot(rew_t.numpy(), label=label)
    ax.legend()
    ax.grid(True)

    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "comparison.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  Comparison chart saved to {path}")
    plt.close(fig)


# ── Main ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=None,
                        help="Override episode count (default: 600 GPU, 50 CPU)")
    args = parser.parse_args()

    if args.episodes:
        num_episodes = args.episodes
    elif torch.cuda.is_available():
        num_episodes = 600
    else:
        num_episodes = 50

    all_results = []  # list of (label, durations, rewards)

    # ── 1. MLP DQN (state-vector observations) ──────────────────────
    print("\n" + "=" * 60)
    print("Training MLP DQN (state-vector observations)")
    print("=" * 60)

    def build_mlp():
        env_tmp = make_env_state()
        n_obs = env_tmp.observation_space.shape[0]
        n_act = env_tmp.action_space.n
        env_tmp.close()
        p = MLPDQN(n_obs, n_act).to(device)
        t = MLPDQN(n_obs, n_act).to(device)
        t.load_state_dict(p.state_dict())
        return p, t

    def mlp_transform(env, obs, reset=False):
        return torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)

    mlp_dur, mlp_rew = train_dqn(
        label="MLP DQN",
        make_env_fn=make_env_state,
        build_net_fn=build_mlp,
        state_transform_fn=mlp_transform,
        num_episodes=num_episodes,
        checkpoint_name="mlp_dqn.pt",
    )
    if mlp_dur:
        plot_single(mlp_dur, mlp_rew, "MLP DQN – Lunar Lander", "mlp_dqn.png")
        all_results.append(("MLP DQN", mlp_dur, mlp_rew))

    # ── 2. CNN DQN (pixel observations) ─────────────────────────────
    print("\n" + "=" * 60)
    print("Training CNN DQN (pixel observations)")
    print("=" * 60)

    frame_stack = FrameStack(N_FRAMES)

    def build_cnn():
        env_tmp = make_env_pixel()
        n_act = env_tmp.action_space.n
        env_tmp.close()
        p = CNNDQN(N_FRAMES, n_act).to(device)
        t = CNNDQN(N_FRAMES, n_act).to(device)
        t.load_state_dict(p.state_dict())
        return p, t

    def cnn_transform(env, obs, reset=False):
        frame = env.render()  # get pixel frame from rgb_array render mode
        if reset:
            stacked = frame_stack.reset(frame)
        else:
            stacked = frame_stack.step(frame)
        return torch.tensor(stacked, dtype=torch.float32, device=device).unsqueeze(0)

    cnn_dur, cnn_rew = train_dqn(
        label="CNN DQN",
        make_env_fn=make_env_pixel,
        build_net_fn=build_cnn,
        state_transform_fn=cnn_transform,
        num_episodes=num_episodes,
        checkpoint_name="cnn_dqn.pt",
    )
    if cnn_dur:
        plot_single(cnn_dur, cnn_rew, "CNN DQN – Lunar Lander", "cnn_dqn.png")
        all_results.append(("CNN DQN", cnn_dur, cnn_rew))

    # ── 3. Comparison ───────────────────────────────────────────────
    if all_results:
        plot_comparison(all_results)

    print("\nDone! All results saved to", RESULTS_DIR)
