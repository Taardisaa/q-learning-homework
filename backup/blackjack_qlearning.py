"""
Part 1: Tabular Q-Learning for Blackjack (Blackjack-v1)

Implements vanilla Q-Learning with an epsilon-greedy policy.
Plots a learning curve and compares against a random baseline.
"""

import matplotlib
matplotlib.use("Agg")
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict


# ── Hyperparameters ──────────────────────────────────────────────────
NUM_EPISODES = 500_000        # total training episodes
ALPHA = 0.01                  # learning rate
GAMMA = 1.0                   # discount factor (episodic, no discounting)
EPSILON_START = 1.0           # initial exploration rate
EPSILON_MIN = 0.05            # minimum exploration rate
EPSILON_DECAY = 0.999995      # multiplicative decay per step
EVAL_EVERY = 10_000           # evaluate learned policy every N episodes
EVAL_EPISODES = 1_000         # episodes per evaluation


def make_env():
    return gym.make("Blackjack-v1", natural=False, sab=False)


# ── Q-Learning agent ────────────────────────────────────────────────
class QLearningAgent:
    def __init__(self, action_space, alpha=ALPHA, gamma=GAMMA,
                 epsilon=EPSILON_START, epsilon_min=EPSILON_MIN,
                 epsilon_decay=EPSILON_DECAY):
        # Q-table: maps (state, action) -> value
        self.q_table = defaultdict(float)
        self.action_space = action_space
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

    def get_action(self, state):
        """Epsilon-greedy action selection."""
        if np.random.random() < self.epsilon:
            return self.action_space.sample()
        return self._greedy_action(state)

    def get_greedy_action(self, state):
        """Pure greedy action (for evaluation)."""
        return self._greedy_action(state)

    def _greedy_action(self, state):
        q_values = [self.q_table[(state, a)] for a in range(self.action_space.n)]
        return int(np.argmax(q_values))

    def update(self, state, action, reward, next_state, done):
        """Standard Q-Learning (off-policy TD(0)) update."""
        best_next = max(
            self.q_table[(next_state, a)] for a in range(self.action_space.n)
        )
        td_target = reward + (1 - done) * self.gamma * best_next
        td_error = td_target - self.q_table[(state, action)]
        self.q_table[(state, action)] += self.alpha * td_error

        # Decay epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)


# ── Training loop ───────────────────────────────────────────────────
def train(agent, num_episodes=NUM_EPISODES):
    env = make_env()
    cumulative_rewards = []
    cumulative = 0

    for ep in range(1, num_episodes + 1):
        state, _ = env.reset()
        done = False

        while not done:
            action = agent.get_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            agent.update(state, action, reward, next_state, done)
            state = next_state

        cumulative += reward
        cumulative_rewards.append(cumulative)

        if ep % 50_000 == 0:
            print(f"Episode {ep:>7d} | cumulative reward: {cumulative:+.0f} "
                  f"| epsilon: {agent.epsilon:.4f}")

    env.close()
    return cumulative_rewards


# ── Evaluation ──────────────────────────────────────────────────────
def evaluate(policy_fn, num_episodes=EVAL_EPISODES):
    """Run a policy for num_episodes and return mean reward."""
    env = make_env()
    total_reward = 0
    for _ in range(num_episodes):
        state, _ = env.reset()
        done = False
        while not done:
            action = policy_fn(state)
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
        total_reward += reward
    env.close()
    return total_reward / num_episodes


def evaluate_over_training(agent, num_episodes=NUM_EPISODES):
    """Periodically evaluate the learned policy during training and collect
    evaluation curve data. Returns (training_cumulative, eval_points)."""
    env = make_env()
    cumulative_rewards = []
    cumulative = 0
    eval_x, eval_y = [], []

    for ep in range(1, num_episodes + 1):
        state, _ = env.reset()
        done = False

        while not done:
            action = agent.get_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            agent.update(state, action, reward, next_state, done)
            state = next_state

        cumulative += reward
        cumulative_rewards.append(cumulative)

        if ep % EVAL_EVERY == 0:
            avg = evaluate(agent.get_greedy_action)
            eval_x.append(ep)
            eval_y.append(avg)
            print(f"Episode {ep:>7d} | eval avg reward: {avg:+.4f} "
                  f"| epsilon: {agent.epsilon:.4f}")

    env.close()
    return cumulative_rewards, (eval_x, eval_y)


# ── Random baseline ─────────────────────────────────────────────────
def random_cumulative_rewards(num_episodes=NUM_EPISODES):
    env = make_env()
    cumulative_rewards = []
    cumulative = 0
    for _ in range(num_episodes):
        state, _ = env.reset()
        done = False
        while not done:
            action = env.action_space.sample()
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
        cumulative += reward
        cumulative_rewards.append(cumulative)
    env.close()
    return cumulative_rewards


# ── Plotting ────────────────────────────────────────────────────────
def plot_results(q_cumulative, rand_cumulative, eval_data):
    episodes = np.arange(1, len(q_cumulative) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # (a) Cumulative reward
    axes[0].plot(episodes, q_cumulative, label="Q-Learning", alpha=0.8)
    axes[0].plot(episodes, rand_cumulative, label="Random", alpha=0.8)
    axes[0].set_xlabel("Episode")
    axes[0].set_ylabel("Cumulative Reward")
    axes[0].set_title("Cumulative Reward over Episodes")
    axes[0].legend()

    # (b) Periodic evaluation of greedy policy
    eval_x, eval_y = eval_data
    axes[1].plot(eval_x, eval_y, marker="o", linewidth=2)
    random_avg = evaluate(lambda s: make_env().action_space.sample())
    axes[1].axhline(random_avg, color="r", linestyle="--", label=f"Random ({random_avg:+.3f})")
    axes[1].set_xlabel("Training Episode")
    axes[1].set_ylabel("Average Reward (greedy policy)")
    axes[1].set_title("Learned Policy Evaluation")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig("blackjack_results.png", dpi=150)
    print("Figure saved to blackjack_results.png")


# ── Main ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    env_tmp = make_env()
    agent = QLearningAgent(env_tmp.action_space)
    env_tmp.close()

    print("=" * 60)
    print("Training Q-Learning agent on Blackjack-v1 ...")
    print("=" * 60)
    q_cumulative, eval_data = evaluate_over_training(agent)

    print("\nRunning random baseline ...")
    rand_cumulative = random_cumulative_rewards()

    # Final comparison
    q_avg = evaluate(agent.get_greedy_action, num_episodes=10_000)
    rand_avg = evaluate(lambda s: make_env().action_space.sample(), num_episodes=10_000)
    print(f"\nFinal evaluation (10 000 episodes):")
    print(f"  Q-Learning greedy policy avg reward: {q_avg:+.4f}")
    print(f"  Random policy avg reward:            {rand_avg:+.4f}")

    plot_results(q_cumulative, rand_cumulative, eval_data)
