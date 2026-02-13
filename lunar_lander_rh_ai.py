"""
Part 2: DQN for Lunar Lander (LunarLander-v3)

Trains an MLP DQN on the Lunar Lander environment using the 8-dim state vector.

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


# ── Hyperparameters ──────────────────────────────────────────────────

BATCH_SIZE = 128
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.01
EPS_DECAY = 2500
TAU = 0.005
LR = 3e-4


# ── Random Baseline ─────────────────────────────────────────────────

def evaluate_random_baseline(make_env_fn, num_episodes=100):
    """Evaluate random policy as baseline."""
    env = make_env_fn()
    ep_durations = []
    ep_rewards = []
    
    print(f"  Evaluating random baseline over {num_episodes} episodes...")
    for ep in tqdm(range(num_episodes), desc="Random Baseline"):
        obs, info = env.reset()
        ep_reward = 0.0
        
        for t in count():
            action = env.action_space.sample()  # random action
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            
            if terminated or truncated:
                ep_durations.append(t + 1)
                ep_rewards.append(ep_reward)
                break
    
    env.close()
    return ep_durations, ep_rewards


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

    # Skip training if checkpoint already exists, but load metrics for chart regeneration
    ckpt_path = os.path.join(RESULTS_DIR, checkpoint_name)
    if os.path.exists(ckpt_path):
        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
        policy_net.load_state_dict(checkpoint["policy_net"])
        target_net.load_state_dict(checkpoint["target_net"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        steps_done = checkpoint["steps_done"]
        ep_durations = checkpoint.get("episode_durations", [])
        ep_rewards = checkpoint.get("episode_rewards", [])
        print(f"  Loaded checkpoint from {ckpt_path} (steps_done={steps_done})")
        print(f"  Skipping training — checkpoint already exists. Delete it to retrain.")
        print(f"  Regenerating charts from saved metrics...")
        env.close()
        return ep_durations, ep_rewards

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

    # Save checkpoint with training metrics
    ckpt_path = os.path.join(RESULTS_DIR, checkpoint_name)
    torch.save({
        "policy_net": policy_net.state_dict(),
        "target_net": target_net.state_dict(),
        "optimizer": optimizer.state_dict(),
        "steps_done": steps_done,
        "episode_durations": ep_durations,
        "episode_rewards": ep_rewards,
    }, ckpt_path)
    print(f"  Checkpoint saved to {ckpt_path}")

    return ep_durations, ep_rewards


# ── Plotting ─────────────────────────────────────────────────────────

def plot_single(durations, rewards, title, filename, baseline_avg_dur=None, baseline_avg_rew=None):
    """Save a 2-panel (duration + reward) chart for one agent."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(title, fontsize=14, fontweight='bold', y=0.98)

    dur_t = torch.tensor(durations, dtype=torch.float)
    ax1.set_title('Episode Duration Over Training', fontsize=11, pad=10)
    ax1.set_xlabel('Episode', fontsize=10)
    ax1.set_ylabel('Duration (steps)', fontsize=10)
    ax1.plot(dur_t.numpy(), color='green', alpha=0.6, label='raw')
    if len(dur_t) >= 20:
        m = dur_t.unfold(0, 20, 1).mean(1).view(-1)
        m = torch.cat((torch.zeros(19), m))
        ax1.plot(m.numpy(), color='red', label='20-ep avg')
    if baseline_avg_dur is not None:
        ax1.axhline(y=baseline_avg_dur, color='blue', linestyle='--', linewidth=1.5, alpha=0.7,
                    label=f'Random baseline ({baseline_avg_dur:.0f})')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Subcaption for duration chart
    duration_caption = ('Number of timesteps the agent survived per episode.\n'
                       'Successful landing terminates early with high reward,\n'
                       'while crashes or time limits end with lower rewards.\n'
                       'Trend shows learning efficiency and policy stability.')
    ax1.text(0.5, -0.25, duration_caption, transform=ax1.transAxes, 
             fontsize=9, ha='center', va='top', style='italic',
             bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.3))

    rew_t = torch.tensor(rewards, dtype=torch.float)
    ax2.set_title('Total Reward Per Episode', fontsize=11, pad=10)
    ax2.set_xlabel('Episode', fontsize=10)
    ax2.set_ylabel('Total Reward', fontsize=10)
    ax2.plot(rew_t.numpy(), color='green', alpha=0.6, label='raw')
    if len(rew_t) >= 20:
        m = rew_t.unfold(0, 20, 1).mean(1).view(-1)
        m = torch.cat((torch.zeros(19), m))
        ax2.plot(m.numpy(), color='red', label='20-ep avg')
    if baseline_avg_rew is not None:
        ax2.axhline(y=baseline_avg_rew, color='blue', linestyle='--', linewidth=1.5, alpha=0.7,
                    label=f'Random baseline ({baseline_avg_rew:.0f})')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Subcaption for reward chart
    reward_caption = ('Cumulative reward obtained during episode.\n'
                     'Rewards: +100 for landing, -100 for crashing,\n'
                     'negative for fuel usage, penalties for tilting.\n'
                     'Environment solved when avg ≥ 200 over 100 episodes.')
    ax2.text(0.5, -0.25, reward_caption, transform=ax2.transAxes, 
             fontsize=9, ha='center', va='top', style='italic',
             bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.3))

    plt.tight_layout(rect=[0, 0.08, 1, 0.96])
    path = os.path.join(RESULTS_DIR, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  Chart saved to {path}")
    plt.close(fig)


def plot_comparison(all_results):
    """Save a combined reward comparison chart for all agents."""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_title("Lunar Lander: MLP DQN vs Random Baseline – Learning Performance", 
                 fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel("Episode", fontsize=11)
    ax.set_ylabel("Total Reward (20-episode moving average)", fontsize=11)

    for label, _, rewards in all_results:
        rew_t = torch.tensor(rewards, dtype=torch.float)
        if len(rew_t) >= 20:
            m = rew_t.unfold(0, 20, 1).mean(1).view(-1)
            ax.plot(m.numpy(), label=label, linewidth=2)
        else:
            ax.plot(rew_t.numpy(), label=label, linewidth=2)
    
    ax.axhline(y=200, color='gray', linestyle='--', linewidth=1, alpha=0.7, 
               label='Solved threshold (200)')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Add caption box
    caption = ('MLP DQN learns from 8-dim state vector\n'
               'Random baseline takes uniformly random actions\n'
               'Environment considered solved at avg reward ≥ 200')
    ax.text(0.02, 0.02, caption, transform=ax.transAxes, fontsize=9,
            verticalalignment='bottom',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.4))

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
    parser.add_argument("--visualize", type=int, nargs='?', const=5, default=None,
                        metavar="N",
                        help="Watch the trained agent play N episodes (default: 5)")
    args = parser.parse_args()

    if args.episodes:
        num_episodes = args.episodes
    elif torch.cuda.is_available():
        num_episodes = 600
    else:
        num_episodes = 50

    all_results = []  # list of (label, durations, rewards)

    # ── 0. Random Baseline ──────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Evaluating Random Baseline")
    print("=" * 60)
    
    baseline_dur, baseline_rew = evaluate_random_baseline(
        make_env_fn=make_env_state,
        num_episodes=100
    )
    all_results.append(("Random Baseline", baseline_dur, baseline_rew))

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
        baseline_avg_dur = np.mean(baseline_dur) if baseline_dur else None
        baseline_avg_rew = np.mean(baseline_rew) if baseline_rew else None
        plot_single(mlp_dur, mlp_rew, "MLP DQN – Lunar Lander", "mlp_dqn.png",
                    baseline_avg_dur=baseline_avg_dur, baseline_avg_rew=baseline_avg_rew)
        all_results.append(("MLP DQN", mlp_dur, mlp_rew))

    # ── 2. Comparison ───────────────────────────────────────────────
    if all_results:
        plot_comparison(all_results)

    print("\nDone! All results saved to", RESULTS_DIR)

    # ── Visualize trained agent ─────────────────────────────────────
    if args.visualize:
        ckpt_path = os.path.join(RESULTS_DIR, "mlp_dqn.pt")
        if not os.path.exists(ckpt_path):
            print("No checkpoint found — train first before visualizing.")
        else:
            print(f"\nVisualizing trained agent for {args.visualize} episodes...")
            vis_env = gym.make("LunarLander-v3", continuous=False, gravity=-10.0,
                              enable_wind=False, wind_power=15.0, turbulence_power=1.5,
                              render_mode="human")
            vis_net = MLPDQN(vis_env.observation_space.shape[0], vis_env.action_space.n).to(device)
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            vis_net.load_state_dict(ckpt["policy_net"])
            vis_net.eval()

            for ep in range(args.visualize):
                obs, _ = vis_env.reset()
                ep_reward = 0.0
                done = False
                while not done:
                    state = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                    with torch.no_grad():
                        action = vis_net(state).argmax(dim=1).item()
                    obs, reward, terminated, truncated, _ = vis_env.step(action)
                    ep_reward += reward
                    done = terminated or truncated
                print(f"  Episode {ep+1}: reward = {ep_reward:.1f}")
            vis_env.close()
