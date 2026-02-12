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

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


def make_env():
    """Create an env and reset it."""
    env = gym.make("LunarLander-v3", continuous=False, gravity=-10.0,
                   enable_wind=False, wind_power=15.0, turbulence_power=1.5)
    obs, info = env.reset()
    return obs, info, env


# A simple replay memory to store transitions
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
    

class DQN(nn.Module):
    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        # Just a simple MLP for now.
        self.fc1 = nn.Linear(n_observations, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, n_actions)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)
    

BATCH_SIZE = 128
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.01
EPS_DECAY = 2500
TAU = 0.005
LR = 3e-4


SAVE_DIR = os.path.join(os.path.dirname(__file__), "checkpoints")
CHECKPOINT_PATH = os.path.join(SAVE_DIR, "lunar_lander_dqn.pt")

parser = argparse.ArgumentParser()
parser.add_argument("--load", type=str, default=None, metavar="PATH",
                    help="Path to a checkpoint to resume from (default: None)")
args = parser.parse_args()

obs, info, env = make_env()

n_actions = env.action_space.n  # type: ignore
n_observations = len(obs)

policy_net = DQN(n_observations, n_actions).to(device)
target_net = DQN(n_observations, n_actions).to(device)

optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
memory = ReplayMemory(100000)
steps_done = 0

if args.load:
    checkpoint = torch.load(args.load, map_location=device, weights_only=True)
    policy_net.load_state_dict(checkpoint["policy_net"])
    target_net.load_state_dict(checkpoint["target_net"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    steps_done = checkpoint["steps_done"]
    print(f"Loaded checkpoint from {args.load} (steps_done={steps_done})")
else:
    target_net.load_state_dict(policy_net.state_dict())


def select_action(state):
    global steps_done
    sample = random.random()    # sample in [0, 1) for epsilon-greedy action selection
    # eps_threshold will use an exponential decay schedule, to lower
    # the exploration rate as the number of steps increases.
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            return policy_net(state).argmax(dim=1).view(1, 1)
    else:
        # sample a random action among the action space
        return torch.tensor([[env.action_space.sample()]], device=device, dtype=torch.long)
    

episode_durations = []
episode_rewards = []


def plot_durations(show_result=False):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), num=1)
    if not show_result:
        ax1.cla()
        ax2.cla()

    # Left: Duration
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    ax1.set_title('Result - Duration' if show_result else 'Training - Duration')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Duration')
    ax1.plot(durations_t.numpy(), color='green', alpha=0.6)
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        ax1.plot(means.numpy(), color='red')

    # Right: Reward
    rewards_t = torch.tensor(episode_rewards, dtype=torch.float)
    ax2.set_title('Result - Reward' if show_result else 'Training - Reward')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Total Reward')
    ax2.plot(rewards_t.numpy(), color='green', alpha=0.6)
    if len(rewards_t) >= 100:
        means = rewards_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        ax2.plot(means.numpy(), color='red')

    plt.tight_layout()
    plt.pause(0.001)


def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    # sample a batch of transitions from the replay memory
    transitions = memory.sample(BATCH_SIZE)
    # convert batch-array of Transitions to Transition of batch-arrays
    batch = Transition(*zip(*transitions))
    # compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                            batch.next_state)), device=device, dtype=torch.bool)
    
    non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    state_action_values = policy_net(state_batch).gather(1, action_batch)

    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    with torch.no_grad():
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1).values
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    # In-place gradient clipping
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()


if torch.cuda.is_available():
    print("Using GPU for training.")
    num_episodes = 600
else:    
    print("Using CPU for training.")
    num_episodes = 50

for i_episode in tqdm(range(num_episodes), desc="Training"):
    state, info = env.reset()
    state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    ep_reward = 0.0
    for t in count():
        action = select_action(state)
        next_state, reward, terminated, truncated, info = env.step(action.item())
        ep_reward += reward
        reward = torch.tensor([reward], device=device)

        done = terminated or truncated
        if terminated:
            next_state = None
        else:
            next_state = torch.tensor(next_state, dtype=torch.float32, device=device).unsqueeze(0)

        memory.push(state, action, next_state, reward)
        state = next_state
        optimize_model()
        target_net_state_dict = target_net.state_dict()
        policy_net_state_dict = policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
        target_net.load_state_dict(target_net_state_dict)
        if done:
            episode_durations.append(t + 1)
            episode_rewards.append(ep_reward)
            plot_durations()
            break


print('Complete')

# Save checkpoint
os.makedirs(SAVE_DIR, exist_ok=True)
torch.save({
    "policy_net": policy_net.state_dict(),
    "target_net": target_net.state_dict(),
    "optimizer": optimizer.state_dict(),
    "steps_done": steps_done,
}, CHECKPOINT_PATH)
print(f"Model saved to {CHECKPOINT_PATH}")

# Save final chart
plot_durations(show_result=True)
chart_path = os.path.join(os.path.dirname(__file__), "lunar_lander_results.png")
plt.savefig(chart_path, dpi=150, bbox_inches="tight")
print(f"Chart saved to {chart_path}")
plt.ioff()
plt.show()