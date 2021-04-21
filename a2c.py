import torch
import gym
import numpy as np
import matplotlib.pyplot as plt

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Device - {}".format(device))

n_episodes = 1000
n_steps = 10000

env = gym.make("CartPole-v1")
env._max_episode_steps = 10000
state = env.reset()
n_actions = env.action_space.n
n_states = env.observation_space.shape[0]


class Actor(torch.nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(in_dim, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, out_dim),
            torch.nn.Softmax()
        )

    def forward(self, x):
        x.to(device)
        return self.layers(x)

    def get_action(self, x):
        x.to(device)
        probs = self.forward(x)
        dist = torch.distributions.Categorical(probs)
        sample = dist.sample()
        return sample.item(), torch.log(probs[sample])


class Critic(torch.nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(in_dim, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 1),
        )

    def forward(self, x):
        x.to(device)
        return self.layers(x)


def update(rewards, log_probs, values, actor_optimizer, critic_optimizer):
    actor_optimizer.zero_grad()
    critic_optimizer.zero_grad()
    q_values = torch.zeros_like(log_probs)
    q_val = 0
    for i in reversed(range(len(rewards))):
        q_val = 0.9 * q_val + rewards[i]
        q_values[i] = q_val
    advantage = q_values - values
    actor_loss = (-log_probs * advantage.detach()).mean()
    critic_loss = 0.5 * advantage.pow(2).mean()
    actor_loss.backward()
    critic_loss.backward()
    actor_optimizer.step()
    critic_optimizer.step()

actor = Actor(n_states, n_actions)
critic = Critic(n_states)
actor_optimizer = torch.optim.Adam(actor.parameters(), lr=3e-4)
critic_optimizer = torch.optim.Adam(critic.parameters(), lr=3e-4)

count_steps = []

for episode in range(n_episodes):
    rewards = []
    prev_states = []
    states = []
    actions = []
    log_probs = []
    values = []
    state = env.reset()
    count = 0
    for step in range(n_steps):
        count += 1
        state = torch.FloatTensor(state)
        prev_states.append(state)
        a, log_prob = actor.get_action(state)
        value = critic.forward(state)
        values.append(value[0])
        state, reward, done, _ = env.step(a)
        env.render()
        rewards.append(reward)
        states.append(state)
        actions.append(a)
        log_probs.append(log_prob)
        if done:
            break
    count_steps.append(count)
    print('episode {}, count {}'.format(episode, count))
    update(rewards, torch.stack(log_probs), torch.stack(values), actor_optimizer, critic_optimizer)

exp_count_steps = [count_steps[0]]
for i in range(1, len(count_steps)):
    exp_count_steps.append(0.4 * exp_count_steps[i - 1] + 0.6 * count_steps[i])
plt.plot(count_steps)
plt.plot(exp_count_steps)
plt.show()
