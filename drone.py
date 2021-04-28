import time
import numpy as np
from gym_pybullet_drones.utils.utils import sync
from env.LiftoffAviary import LiftoffAviary
from env.LiftoffAviary import ActionType
from drone_agent import Agent
import matplotlib.pyplot as plt

GUI = True  # Use GUI

n_episodes = 500  # Count of games to play
n_steps = 1000  # Count of steps in a single game
N = 1000  # Frequency of policy update and size of memory

if __name__ == "__main__":

    env = LiftoffAviary(gui=GUI,
                        record=False,
                        act=ActionType.RPM)
    agent = Agent(n_actions=4, input_dims=env.observation_space.shape[0])
    agent.load_models()

    current_step = 0
    learn_iters = 0
    avg_score = 0
    best_score = env.reward_range[0]

    score_history = []
    avg_score_history = []

    for episode in range(n_episodes):
        obs = env.reset()
        start = time.time()
        score = 0
        for i in range(n_steps):
            actions, prob, val = agent.choose_action(obs)
            obs, reward, done, info = env.step(actions)
            current_step += 1
            score += reward
            agent.remember(obs, actions, prob, val, reward, done)
            if i % env.SIM_FREQ == 0 and GUI == True:
                env.render()
            if current_step % N:
                agent.learn()
                learn_iters += 1
            sync(i, start, env.TIMESTEP)
            if done:
                obs = env.reset()
                break

        score_history.append(score)
        avg_score = np.mean(score_history[-70:])
        avg_score_history.append(avg_score)
        if avg_score > best_score:
            best_score = avg_score
            agent.save_models()
        print('episode', episode, 'score %.1f' % score, 'avg score %.1f' % avg_score,
              'time_steps', current_step, 'learning_steps', learn_iters)
    env.close()
    fig, axs = plt.subplots(2)
    axs[0].plot(score_history)
    axs[0].plot(avg_score_history)
    plt.show()
