import argparse
import numpy as np
import os
import torch
import torch as T
import torch.nn as nn
import torch.optim as optim
import gym
import matplotlib.pyplot as plt


class PPOMemory:
    def __init__(self, batch_size):
        self.states = []
        self.probs = []
        self.vals = []
        self.actions = []
        self.rewards = []
        self.dones = []

        self.batch_size = batch_size

    def generate_batches(self):
        n_states = len(self.states)
        batch_start = np.arange(0, n_states, self.batch_size)
        indices = np.arange(n_states, dtype=np.int64)
        np.random.shuffle(indices)
        batches = [indices[i: i + self.batch_size] for i in batch_start]

        return np.array(self.states), \
               np.array(self.actions), \
               np.array(self.probs), \
               np.array(self.vals), \
               np.array(self.rewards), \
               np.array(self.dones), \
               batches

    def store_memory(self, state, action, probs, vals, reward, done):
        self.states.append(state)
        self.actions.append(action)
        self.probs.append(probs)
        self.vals.append(vals)
        self.rewards.append(reward)
        self.dones.append(done)

    def clear_memory(self):
        self.states = []
        self.probs = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.vals = []


class CriticNetwork(nn.Module):
    def __init__(self, input_dims, alpha, fc1_dims=512, fc2_dims=256, fc3_dims=128,
                 chkpt_dir=''):
        super(CriticNetwork, self).__init__()

        self.checkpoint_file = os.path.join(chkpt_dir, 'critic_torch_ppo')
        self.critic = nn.Sequential(
            nn.Linear(*input_dims, fc1_dims),
            nn.ReLU(),
            nn.Linear(fc1_dims, fc2_dims),
            nn.ReLU(),
            nn.Linear(fc2_dims, fc3_dims),
            nn.ReLU(),
            nn.Linear(fc3_dims, 1),
        )

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        value = self.critic(state)

        return value

    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))


mu_history_1 = []
mu_history_2 = []


class ActorNetworkAction1(nn.Module):
    def __init__(self, n_actions, input_dims, alpha,
                 fc1_dims=256, fc2_dims=256, fc3_dims=128, chkpt_dir=''):
        super(ActorNetworkAction1, self).__init__()

        self.checkpoint_file = os.path.join(chkpt_dir, 'actor_torch_ppo_1')

        self.base = nn.Sequential(
            nn.Linear(*input_dims, fc1_dims),
            nn.ReLU(),
            nn.Linear(fc1_dims, fc2_dims),
            nn.ReLU(),
            nn.Linear(fc2_dims, fc3_dims),
            nn.ReLU(),
        )
        self.mu = nn.Sequential(
            nn.Linear(fc3_dims, 1),
            nn.Tanh()
        )
        self.var = nn.Sequential(
            nn.Linear(fc3_dims, 1),
            nn.Softplus()
        )
        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        state = self.base(state)
        return torch.squeeze(self.mu(state)), torch.squeeze(self.var(state))


    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))


class ActorNetworkAction2(nn.Module):
    def __init__(self, n_actions, input_dims, alpha,
                 fc1_dims=256, fc2_dims=256, fc3_dims=128, chkpt_dir=''):
        super(ActorNetworkAction2, self).__init__()

        self.checkpoint_file = os.path.join(chkpt_dir, 'actor_torch_ppo_2')

        self.base = nn.Sequential(
            nn.Linear(*input_dims, fc1_dims),
            nn.ReLU(),
            nn.Linear(fc1_dims, fc2_dims),
            nn.ReLU(),
            nn.Linear(fc2_dims, fc3_dims),
            nn.ReLU(),
        )
        self.mu = nn.Sequential(
            nn.Linear(fc3_dims, 1),
            nn.Tanh()
        )
        self.var = nn.Sequential(
            nn.Linear(fc3_dims, 1),
            nn.Softplus()
        )
        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        state = self.base(state)
        return torch.squeeze(self.mu(state)), torch.squeeze(self.var(state))


    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))



class Agent:
    def __init__(self, n_actions, input_dims, gamma=0.99, alpha=0.0003, gae_lambda=0.95,
                 policy_clip=0.5, batch_size=64, n_epochs=10):
        self.gamma = gamma
        self.policy_clip = policy_clip
        self.n_epochs = n_epochs
        self.gae_lambda = gae_lambda

        self.actorAction1 = ActorNetworkAction1(n_actions, input_dims, alpha)
        self.actorAction2 = ActorNetworkAction2(n_actions, input_dims, alpha)
        self.critic = CriticNetwork(input_dims, alpha)
        self.memory = PPOMemory(batch_size)

    def remember(self, state, action, probs, vals, reward, done):
        self.memory.store_memory(state, action, probs, vals, reward, done)

    def save_models(self):
        print('... saving models ...')
        self.actorAction1.save_checkpoint()
        self.actorAction2.save_checkpoint()
        self.critic.save_checkpoint()

    def load_models(self):
        print('... loading models ...')
        self.actorAction1.load_checkpoint()
        self.actorAction2.load_checkpoint()
        self.critic.load_checkpoint()

    def choose_action(self, observation):
        state = T.tensor([observation], dtype=T.float).to(self.actorAction1.device)
        action_mu_1, action_sigma_1 = self.actorAction1.forward(state)
        action_mu_2, action_sigma_2 = self.actorAction2.forward(state)
        action_mu = torch.stack([action_mu_1, action_mu_2])
        action_sigma = torch.stack([action_sigma_1, action_sigma_2])
        mu_history_1.append(action_mu_1)
        mu_history_2.append(action_mu_2)

        action_dist = torch.distributions.multivariate_normal.MultivariateNormal(torch.squeeze(action_mu.cpu()), torch.diag(torch.squeeze(action_sigma.cpu())))
        # action_dist = torch.distributions.normal.Normal(torch.squeeze(action_mu.cpu()), torch.squeeze(action_sigma.cpu()))
        # action_dist = [torch.distributions.normal.Normal(mu, var) for mu, var in zip(torch.squeeze(action_mu.cpu()), torch.squeeze(action_sigma.cpu()))]
        act = action_dist.sample()
        act = torch.clamp(act, float(env.action_space.low[0]), float(env.action_space.high[0]))
        value = self.critic(state)
        value = T.squeeze(value).item()
        return act.detach().numpy(), T.squeeze(action_dist.log_prob(act)).item(), value

    def learn(self):
        for _ in range(self.n_epochs):
            state_arr, action_arr, old_prob_arr, vals_arr, \
            reward_arr, dones_arr, batches = \
                self.memory.generate_batches()

            values = vals_arr
            advantage = np.zeros(len(reward_arr), dtype=np.float32)

            for t in range(len(reward_arr) - 1):
                discount = 1
                a_t = 0
                for k in range(t, len(reward_arr) - 1):
                    a_t += discount * (reward_arr[k] + self.gamma * values[k + 1] *
                                       (1 - int(dones_arr[k])) - values[k])
                    discount *= self.gamma * self.gae_lambda
                advantage[t] = a_t
            advantage = T.tensor(advantage).to(self.actorAction1.device)

            values = T.tensor(values).to(self.actorAction1.device)
            for batch in batches:
                states = T.tensor(state_arr[batch], dtype=T.float).to(self.actorAction1.device)
                old_probs = T.tensor(old_prob_arr[batch]).to(self.actorAction1.device)
                actions = T.tensor(action_arr[batch]).to(self.actorAction1.device)

                action_mu_1, action_sigma_1 = self.actorAction1.forward(states)
                action_mu_2, action_sigma_2 = self.actorAction2.forward(states)
                mu = torch.stack([action_mu_1, action_mu_2], axis=1)
                var = torch.stack([action_sigma_1, action_sigma_2], axis=1)
                dist = torch.distributions.multivariate_normal.MultivariateNormal(torch.squeeze(mu.cpu()), torch.torch.diag_embed(var.cpu()))
                # dist = torch.distributions.normal.Normal(torch.squeeze(mu.cpu()),
                #                                                 torch.squeeze(var.cpu()))
                critic_value = self.critic(states)

                critic_value = T.squeeze(critic_value)

                new_probs = dist.log_prob(actions)
                # prob_ratio = new_probs.exp() / old_probs.exp()
                # ratios.append(prob_ratio[0].item())
                prob_ratio = (new_probs - old_probs).exp()
                weighted_probs = advantage[batch] * prob_ratio
                weighted_clipped_probs = T.clamp(prob_ratio, 1 - self.policy_clip,
                                                 1 + self.policy_clip) * advantage[batch]
                actor_loss = -T.min(weighted_probs, weighted_clipped_probs).mean()

                returns = advantage[batch] + values[batch]
                critic_loss = (returns - critic_value) ** 2
                critic_loss = critic_loss.mean()

                total_loss = actor_loss + 0.5 * critic_loss
                self.actorAction1.optimizer.zero_grad()
                self.actorAction2.optimizer.zero_grad()
                self.critic.optimizer.zero_grad()
                total_loss.backward()
                self.actorAction1.optimizer.step()
                self.actorAction2.optimizer.step()
                self.critic.optimizer.step()

        self.memory.clear_memory()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PPO continuous control')
    parser.add_argument(
        '--render',
        type=int,
        default=1,
        help='Render env'
    )
    arguments = parser.parse_args()
    print('Using rendering - {}'.format(arguments.render))

    env = gym.make('LunarLanderContinuous-v2')
    N = 30
    batch_size = 10
    n_epochs = 5
    policy_clip = 0.2
    alpha = 3e-4
    agent = Agent(n_actions=env.action_space.shape[0], batch_size=batch_size, policy_clip=policy_clip,
                  alpha=alpha, n_epochs=n_epochs,
                  input_dims=env.observation_space.shape)
    # agent.load_models()
    n_games = 500
    figure_file = 'plots/cartpole.png'
    best_score = env.reward_range[0]
    score_history = []
    avg_score_history = []
    learn_iters = 0
    avg_score = 0
    n_steps = 0

    for i in range(n_games):
        observation = env.reset()
        done = False
        score = 0
        while not done:
            action, prob, val = agent.choose_action(observation)
            observation_, reward, done, info = env.step(action)
            # if arguments.render == 1:
            #     env.render()
            n_steps += 1
            score += reward
            agent.remember(observation, action, prob, val, reward, done)
            if n_steps % N == 0:
                agent.learn()
                learn_iters += 1
            observation = observation_
        score_history.append(score)
        avg_score = np.mean(score_history[-70:])
        avg_score_history.append(avg_score)

        if avg_score > best_score:
            best_score = avg_score
            agent.save_models()

        print('episode', i, 'score %.1f' % score, 'avg score %.1f' % avg_score,
              'time_steps', n_steps, 'learning_steps', learn_iters)
    x = [i + 1 for i in range(len(score_history))]
    # plot_learning_curve(x, score_history, figure_file)
    plt.plot(score_history)
    plt.plot(avg_score_history)
    mu_history_1_ = [mu_history_1[i].item() for i in range(len(mu_history_1))]
    mu_history_2_ = [mu_history_2[i].item() for i in range(len(mu_history_2))]
    fig, axs = plt.subplots(4)
    axs[0].plot(mu_history_2_)
    axs[1].plot(mu_history_1_)
    axs[2].plot(mu_history_1_)
    axs[2].plot(mu_history_2_)
    axs[3].plot(score_history)
    axs[3].plot(avg_score_history)
    fig.tight_layout()


    plt.title('N - {}, batch_size - {}, n_epochs - {}, policy_clip - {}'.format(N, batch_size, n_epochs, policy_clip))
    plt.show()
