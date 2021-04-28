import gym
import torch
import numpy as np
from torch.autograd import Variable
import matplotlib.pyplot as plt

if torch.cuda.is_available():
  dev = "cuda:0"
else:
  dev = "cpu"

save_model = True

env = gym.make('CartPole-v1')
env._max_episode_steps = 200
state = env.reset()
n_states = env.observation_space.shape[0]
n_actions = env.action_space.n
n_episodes = 200
max_steps = 200
GAMMA = 0.9


class Reinforce(torch.nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.num_actions = out_dim
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
        return self.layers(x)

    def get_action(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0)
        probs = self.forward(Variable(state))
        highest_prob_action = np.random.choice(self.num_actions, p=np.squeeze(probs.detach().numpy()))
        log_prob = torch.log(probs.squeeze(0)[highest_prob_action])
        return highest_prob_action, log_prob


def update_policy(optimizer, rewards, log_probs):
    discounted_rewards = []

    for t in range(len(rewards)):
        Gt = 0
        pw = 0
        for r in rewards[t:]:
            Gt = Gt + GAMMA ** pw * r
            pw = pw + 1
        discounted_rewards.append(Gt)

    discounted_rewards = torch.tensor(discounted_rewards)
    discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (
                discounted_rewards.std() + 1e-9)  # normalize discounted rewards

    policy_gradient = []
    for log_prob, Gt in zip(log_probs, discounted_rewards):
        policy_gradient.append(-log_prob * Gt)

    optimizer.zero_grad()
    policy_gradient = torch.stack(policy_gradient).sum()
    policy_gradient.backward()
    optimizer.step()


model = Reinforce(n_states, n_actions)
# model.load_state_dict(torch.load("reinforce"))
model.to(dev)

optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
rewards_size = []
avg_rewards_size = []

for episode in range(n_episodes):
    state = env.reset()
    print(episode)
    rewards = []
    actions = []
    states = []
    log_probs = []
    for step in range(max_steps):
        highest_prob_action, log_prob = model.get_action(state)
        state, reward, done, _ = env.step(highest_prob_action)
        env.render()

        states.append(state)
        actions.append(highest_prob_action)
        rewards.append(reward)
        log_probs.append(log_prob)
        if done:
            print('steps - %f', step)
            break

    update_policy(optimizer, rewards, log_probs)
    rewards_size.append(len(rewards))
    torch.save(model.state_dict(), 'reinforce')

plt.plot(rewards_size)
plt.show()
# alpha = 1
# while alpha != 0:
#     smoothed = []
#     smoothed.append(rewards_size[0])
#     for i in range(1, len(rewards_size)):
#         smoothed.append(alpha * rewards_size[i] + (1 - alpha) * smoothed[i - 1])
#     plt.plot(smoothed)
#     plt.legend(alpha)
#     plt.show()
#     alpha -= 0.05
