"""
Part 2: DQN for Lunar Lander (LunarLander-v3)

Implements DQN with:
  - Experience replay buffer
  - Soft target network updates
  - Huber loss (SmoothL1Loss)

Based on the PyTorch DQN tutorial structure.
"""

import matplotlib
matplotlib.use("Agg")
import gymnasium as gym
import math
import random
import numpy as np
import matplotlib.pyplot as plt
from collections import deque, namedtuple
from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# ── Device selection ─────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ── Hyperparameters ──────────────────────────────────────────────────
BATCH_SIZE = 128
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 1000        # higher = slower decay
TAU = 0.005              # soft update coefficient
LR = 1e-4
MEMORY_SIZE = 10_000
NUM_EPISODES = 600
EVAL_EVERY = 50          # evaluate greedy policy every N episodes
EVAL_EPISODES = 20       # episodes per evaluation


# ── Replay memory ────────────────────────────────────────────────────
Transition = namedtuple("Transition",
                        ("state", "action", "next_state", "reward"))


class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


# ── DQN network ──────────────────────────────────────────────────────
class DQN(nn.Module):
    def __init__(self, n_observations, n_actions):
        super().__init__()
        self.fc1 = nn.Linear(n_observations, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, n_actions)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


# ── Action selection ─────────────────────────────────────────────────
steps_done = 0


def select_action(state, policy_net, env):
    global steps_done
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1.0 * steps_done / EPS_DECAY)
    steps_done += 1
    if random.random() > eps_threshold:
        with torch.no_grad():
            return policy_net(state).max(1).indices.view(1, 1)
    else:
        return torch.tensor([[env.action_space.sample()]],
                            device=device, dtype=torch.long)


# ── Optimization step ───────────────────────────────────────────────
def optimize_model(memory, policy_net, target_net, optimizer):
    if len(memory) < BATCH_SIZE:
        return

    transitions = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))

    # Mask for non-final next states
    non_final_mask = torch.tensor(
        tuple(s is not None for s in batch.next_state),
        device=device, dtype=torch.bool
    )
    non_final_next_states = torch.cat(
        [s for s in batch.next_state if s is not None]
    )

    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Q(s, a) from policy net
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # V(s') from target net
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    with torch.no_grad():
        next_state_values[non_final_mask] = \
            target_net(non_final_next_states).max(1).values

    # Expected Q values
    expected_state_action_values = reward_batch + GAMMA * next_state_values

    # Huber loss
    loss = F.smooth_l1_loss(state_action_values,
                            expected_state_action_values.unsqueeze(1))

    optimizer.zero_grad()
    loss.backward()
    # Gradient clipping
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()


# ── Soft target update ──────────────────────────────────────────────
def soft_update(policy_net, target_net, tau=TAU):
    for target_param, policy_param in zip(target_net.parameters(),
                                          policy_net.parameters()):
        target_param.data.copy_(
            tau * policy_param.data + (1.0 - tau) * target_param.data
        )


# ── Evaluation ──────────────────────────────────────────────────────
def evaluate_policy(policy_net, num_episodes=EVAL_EPISODES, render=False):
    render_mode = "human" if render else None
    env = gym.make("LunarLander-v3", continuous=False, render_mode=render_mode)
    total_rewards = []
    for _ in range(num_episodes):
        state, _ = env.reset()
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        ep_reward = 0
        for _ in count():
            with torch.no_grad():
                action = policy_net(state).max(1).indices.view(1, 1)
            next_state, reward, terminated, truncated, _ = env.step(action.item())
            ep_reward += reward
            if terminated or truncated:
                break
            state = torch.tensor(next_state, dtype=torch.float32,
                                 device=device).unsqueeze(0)
        total_rewards.append(ep_reward)
    env.close()
    return np.mean(total_rewards)


def evaluate_random(num_episodes=EVAL_EPISODES):
    env = gym.make("LunarLander-v3", continuous=False)
    total_rewards = []
    for _ in range(num_episodes):
        state, _ = env.reset()
        ep_reward = 0
        done = False
        while not done:
            action = env.action_space.sample()
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            ep_reward += reward
        total_rewards.append(ep_reward)
    env.close()
    return np.mean(total_rewards)


# ── Training ────────────────────────────────────────────────────────
def train():
    global steps_done
    steps_done = 0

    env = gym.make("LunarLander-v3", continuous=False)

    n_actions = env.action_space.n
    state, _ = env.reset()
    n_observations = len(state)

    policy_net = DQN(n_observations, n_actions).to(device)
    target_net = DQN(n_observations, n_actions).to(device)
    target_net.load_state_dict(policy_net.state_dict())

    optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
    memory = ReplayMemory(MEMORY_SIZE)

    episode_rewards = []
    eval_x, eval_y = [], []

    for ep in range(1, NUM_EPISODES + 1):
        state, _ = env.reset()
        state = torch.tensor(state, dtype=torch.float32,
                             device=device).unsqueeze(0)
        ep_reward = 0

        for t in count():
            action = select_action(state, policy_net, env)
            observation, reward, terminated, truncated, _ = env.step(action.item())
            ep_reward += reward
            reward_t = torch.tensor([reward], device=device)
            done = terminated or truncated

            if done:
                next_state = None
            else:
                next_state = torch.tensor(observation, dtype=torch.float32,
                                          device=device).unsqueeze(0)

            memory.push(state, action, next_state, reward_t)
            state = next_state

            optimize_model(memory, policy_net, target_net, optimizer)
            soft_update(policy_net, target_net)

            if done:
                break

        episode_rewards.append(ep_reward)

        if ep % 10 == 0:
            avg_last = np.mean(episode_rewards[-10:])
            print(f"Episode {ep:>4d} | last-10 avg: {avg_last:+8.2f} "
                  f"| ep reward: {ep_reward:+8.2f}")

        if ep % EVAL_EVERY == 0:
            avg = evaluate_policy(policy_net)
            eval_x.append(ep)
            eval_y.append(avg)
            print(f"  >> Eval ({EVAL_EPISODES} eps) avg reward: {avg:+.2f}")

    env.close()
    return policy_net, episode_rewards, (eval_x, eval_y)


# ── Plotting ────────────────────────────────────────────────────────
def plot_results(episode_rewards, eval_data):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # (a) Training reward per episode (smoothed)
    rewards = np.array(episode_rewards)
    window = 20
    smoothed = np.convolve(rewards, np.ones(window) / window, mode="valid")
    axes[0].plot(smoothed, alpha=0.9, label=f"DQN (smoothed, w={window})")
    axes[0].set_xlabel("Episode")
    axes[0].set_ylabel("Episode Reward")
    axes[0].set_title("Training Reward (Lunar Lander)")
    axes[0].legend()

    # (b) Periodic greedy evaluation
    eval_x, eval_y = eval_data
    random_avg = evaluate_random(num_episodes=100)
    axes[1].plot(eval_x, eval_y, marker="o", linewidth=2, label="DQN greedy")
    axes[1].axhline(random_avg, color="r", linestyle="--",
                    label=f"Random ({random_avg:+.1f})")
    axes[1].set_xlabel("Training Episode")
    axes[1].set_ylabel("Average Reward")
    axes[1].set_title("Greedy Policy Evaluation")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig("lunar_lander_results.png", dpi=150)
    print("Figure saved to lunar_lander_results.png")


# ── Visualize learned policy ────────────────────────────────────────
def visualize(policy_net, num_episodes=5):
    """Render the learned policy with human-visible graphics."""
    print(f"\nVisualizing learned policy for {num_episodes} episodes ...")
    for i in range(num_episodes):
        reward = evaluate_policy(policy_net, num_episodes=1, render=True)
        print(f"  Episode {i+1} reward: {reward:+.2f}")


# ── Main ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    policy_net, episode_rewards, eval_data = train()

    # Final comparison
    dqn_avg = evaluate_policy(policy_net, num_episodes=100)
    rand_avg = evaluate_random(num_episodes=100)
    print(f"\nFinal evaluation (100 episodes):")
    print(f"  DQN greedy policy avg reward: {dqn_avg:+.2f}")
    print(f"  Random policy avg reward:     {rand_avg:+.2f}")

    plot_results(episode_rewards, eval_data)

    # Uncomment the following line to watch your agent land:
    # visualize(policy_net, num_episodes=5)
