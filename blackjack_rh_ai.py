"""
A module that implemented tabular one-step Q-learning for Blackjack.
"""

from collections import defaultdict
import random
from typing import Any, List, Tuple
import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import tqdm



EPISODES = 100000  # each episode is a round in the game. Set to a big number to let it learn a lot.
EVAL_INTERVAL = 100  # evaluate every N training episodes
EVAL_EPISODES = 100   # number of games per evaluation


def make_env() -> Tuple[Any, Any, gym.Env]:
    """Create an env and reset it."""
    env = gym.make('Blackjack-v1', natural=False, sab=False)
    obs, info = env.reset()
    return obs, info, env
    

class QLearningAgent:
    """The Q-learning Agent.
    In essence it should only keep the tabular Q values:
    a mapping from <state, action> pair to an estimated Q value.

    Supports two exploration strategies:
      - "epsilon_greedy": standard ε-greedy (default)
      - "boltzmann": Boltzmann (softmax) exploration with temperature τ
    """
    def __init__(self, val: float=0.0, epsilon: float=1.0,
                 alpha: float=0.01,
                 gamma: float=0.95,
                 epsilon_decay: float=0.0,
                 final_epsilon: float=0.1,
                 exploration: str="epsilon_greedy",
                 tau: float=1.0,
                 tau_decay: float=0.0,
                 final_tau: float=0.1):
        # Q is a table of mapping ((player_sum, dealer_showing, usable_ace), action) --> a value
        self._Q = defaultdict(lambda: val)
        self._epsilon = epsilon
        self._alpha = alpha
        self._gamma = gamma
        self._epsilon_decay = epsilon_decay
        self._final_epsilon = final_epsilon
        self._exploration = exploration
        self._tau = tau
        self._tau_decay = tau_decay
        self._final_tau = final_tau
        self.training_error = []

    def choose_action(self, obs: Any):
        if self._exploration == "boltzmann":
            q_values = np.array([self._Q[(obs, 0)], self._Q[(obs, 1)]])
            # numerically stable softmax
            scaled = q_values / self._tau
            scaled -= scaled.max()
            exp_vals = np.exp(scaled)
            probs = exp_vals / exp_vals.sum()
            return np.random.choice([0, 1], p=probs)
        else:
            # epsilon-greedy
            if random.random() < self._epsilon:
                return random.choice([0, 1])
            else:
                return 0 if self._Q[(obs, 0)] >= self._Q[(obs, 1)] else 1

    def update_q(self, obs, action, next_obs, reward, done) -> float:
        # This is important! since Blackjack only receives reward when done is True,
        # we cannot ignore the final step in each round.
        if done:
            target = reward
        else:
            target = reward + self._gamma*max(self._Q[(next_obs, 0)], self._Q[(next_obs, 1)])
        td_error = target - self._Q[(obs, action)]
        self._Q[(obs, action)] += self._alpha * td_error
        self.training_error.append(td_error)
        return td_error

    def decay_epsilon(self):
        self._epsilon = max(self._final_epsilon, self._epsilon - self._epsilon_decay)
        self._tau = max(self._final_tau, self._tau - self._tau_decay)


class RandomAgent:
    def choose_action(self, obs: Any):
        return random.choice([0, 1])
    

def evaluate(agent, episodes: int = EVAL_EPISODES) -> float:
    wins = 0
    for _ in range(episodes):
        obs, _, env = make_env()
        done = False
        while not done:
            if isinstance(agent, QLearningAgent):
                # greedy: always pick best action
                action = 0 if agent._Q[(obs, 0)] >= agent._Q[(obs, 1)] else 1
            else:
                action = agent.choose_action(obs)
            obs, reward, done, _, _ = env.step(action)
        if reward == 1.0:
            wins += 1
    return wins / episodes


def train(alpha: float, gamma: float, epsilon: float, final_epsilon: float, label: str,
          exploration: str = "epsilon_greedy", tau: float = 1.0, final_tau: float = 0.1):
    epsilon_decay = epsilon / (EPISODES / 2)
    tau_decay = (tau - final_tau) / (EPISODES / 2) if exploration == "boltzmann" else 0.0
    agent = QLearningAgent(
        alpha=alpha, gamma=gamma, epsilon=epsilon,
        epsilon_decay=epsilon_decay, final_epsilon=final_epsilon,
        exploration=exploration, tau=tau, tau_decay=tau_decay, final_tau=final_tau,
    )
    episode_rewards = []
    episode_lengths = []
    print(f"\nTraining: {label}")
    for ep in tqdm.tqdm(range(EPISODES)):
        obs, info, env = make_env()
        done = False
        ep_reward = 0.0
        ep_length = 0
        while not done:
            action = agent.choose_action(obs)
            next_obs, reward, done, _, info = env.step(action)
            agent.update_q(obs, action, next_obs, reward, done)
            ep_reward += reward
            ep_length += 1
            obs = next_obs
        episode_rewards.append(ep_reward)
        episode_lengths.append(ep_length)
        agent.decay_epsilon()
        # if (ep + 1) % EVAL_INTERVAL == 0:
        #     win_rate = evaluate(agent)
        #     eval_x.append(ep + 1)
        #     eval_y.append(win_rate)
    return episode_rewards, episode_lengths, agent


CONFIGS = [
    # Epsilon-greedy with fixed (constant) epsilons
    {"alpha": 0.01, "gamma": 0.95, "epsilon": 0.01, "final_epsilon": 0.01, "label": "ε-greedy ε=0.01"},
    {"alpha": 0.01, "gamma": 0.95, "epsilon": 0.1, "final_epsilon": 0.1, "label": "ε-greedy ε=0.1"},
    {"alpha": 0.01, "gamma": 0.95, "epsilon": 0.5, "final_epsilon": 0.5, "label": "ε-greedy ε=0.5"},
    {"alpha": 0.01, "gamma": 0.95, "epsilon": 0.9, "final_epsilon": 0.9, "label": "ε-greedy ε=0.9"},
    {"alpha": 0.01, "gamma": 0.95, "epsilon": 1.0, "final_epsilon": 1.0, "label": "ε-greedy ε=1.0"},
    # Boltzmann (Softmax) with different initial temperatures
    {"alpha": 0.01, "gamma": 0.95, "epsilon": 0.0, "final_epsilon": 0.0, "exploration": "boltzmann", "tau": 10.0, "final_tau": 0.1, "label": "Boltzmann τ=10.0→0.1"},
    {"alpha": 0.01, "gamma": 0.95, "epsilon": 0.0, "final_epsilon": 0.0, "exploration": "boltzmann", "tau": 1.0, "final_tau": 0.01, "label": "Boltzmann τ=1.0→0.01"},
    {"alpha": 0.01, "gamma": 0.95, "epsilon": 0.0, "final_epsilon": 0.0, "exploration": "boltzmann", "tau": 0.5, "final_tau": 0.01, "label": "Boltzmann τ=0.5→0.01"},
]


def run_random_baseline() -> List[float]:
    random_agent = RandomAgent()
    episode_rewards = []
    print("\nRunning: Random Policy")
    for _ in tqdm.tqdm(range(EPISODES)):
        obs, _, env = make_env()
        done = False
        ep_reward = 0.0
        while not done:
            action = random_agent.choose_action(obs)
            obs, reward, done, _, _ = env.step(action)
            ep_reward += reward
        episode_rewards.append(ep_reward)
    return episode_rewards


def plot_results(reward_curves, all_train_stats):
    rolling_length = 500

    # Plot 1: Cumulative reward over episodes (Q-learning vs Random)
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    for label, ep_rewards in reward_curves:
        cumulative = np.cumsum(ep_rewards)
        ax1.plot(range(len(cumulative)), cumulative, label=label)
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Cumulative Reward")
    ax1.set_title("Q-Learning Blackjack: Cumulative Reward over Training")
    ax1.legend()
    ax1.grid(True)
    fig1.savefig("blackjack_results.png", dpi=150)

    # Plot 2: Episode rewards, lengths, training error (per config)
    for label, ep_rewards, ep_lengths, agent in all_train_stats:
        fig, axs = plt.subplots(ncols=3, figsize=(12, 5))
        fig.suptitle(label)

        axs[0].set_title("Episode rewards")
        reward_moving_average = (
            np.convolve(
                np.array(ep_rewards).flatten(), np.ones(rolling_length), mode="valid"
            ) / rolling_length
        )
        axs[0].plot(range(len(reward_moving_average)), reward_moving_average)

        axs[1].set_title("Episode lengths")
        length_moving_average = (
            np.convolve(
                np.array(ep_lengths).flatten(), np.ones(rolling_length), mode="valid"
            ) / rolling_length
        )
        axs[1].plot(range(len(length_moving_average)), length_moving_average)

        axs[2].set_title("Training Error")
        training_error_moving_average = (
            np.convolve(
                np.array(agent.training_error), np.ones(rolling_length), mode="valid"
            ) / rolling_length
        )
        axs[2].plot(range(len(training_error_moving_average)), training_error_moving_average)

        plt.tight_layout()
        fig.savefig(f"blackjack_detail_{label}.png", dpi=150)

    plt.show()


if __name__ == "__main__":
    reward_curves = []
    all_train_stats = []

    random_rewards = run_random_baseline()
    reward_curves.append(("Random Policy", random_rewards))

    for cfg in CONFIGS:
        ep_rewards, ep_lengths, agent = train(**cfg)
        reward_curves.append((cfg["label"], ep_rewards))
        all_train_stats.append((cfg["label"], ep_rewards, ep_lengths, agent))

    plot_results(reward_curves, all_train_stats)

