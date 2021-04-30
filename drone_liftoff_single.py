import time
import numpy as np
from gym_pybullet_drones.utils.utils import sync
from env.LiftoffAviary import LiftoffAviary
from env.LiftoffAviary import ActionType
from drone_liftoff_single_agent import LiftoffSingleAgent
from utils.live_plotter import live_plotter
import matplotlib.pyplot as plt
import os.path
import csv
from time import gmtime, strftime

GUI = False  # Use GUI

n_episodes = 10000  # Count of games to play
n_steps = 1000  # Count of steps in a single game
N = 1000  # Frequency of policy update and size of memory
batch_size = 100
n_epochs = 4
alpha = 7e-4


def write_stat(episode, step, reward, writer, env):
    writer.writerow([episode, step, reward, env.pos[0, 0], env.pos[0, 1], env.pos[0, 2],
                     env.vel[0, 0], env.vel[0, 1], env.vel[0, 2],
                     env.rpy[0, 0] * env.RAD2DEG, env.rpy[0, 1] * env.RAD2DEG, env.rpy[0, 2] * env.RAD2DEG,
                     env.ang_v[0, 0], env.ang_v[0, 1], env.ang_v[0, 2]])


if __name__ == "__main__":

    goal = 0.7

    env = LiftoffAviary(goal, gui=GUI,
                        record=False,
                        act=ActionType.ONE_D_RPM)
    agent = LiftoffSingleAgent(n_actions=4, batch_size=batch_size,
                  alpha=alpha, n_epochs=n_epochs,
                  input_dims=env.observation_space.shape[0])
    # agent.load_models()

    current_step = 0
    learn_iters = 0
    avg_score = 0
    best_score = env.reward_range[0]

    score_history = []
    avg_score_history = []

    stat_filename = 'stat_liftoff/stat_' + strftime("%Y-%m-%d_%H:%M:%S", gmtime()) + '.csv'
    if not os.path.isfile(stat_filename):
        file = open(stat_filename, 'w')
        writer = csv.writer(file)
        writer.writerow(['episode', 'step', 'reward', 'X', 'Y', 'Z',
                         'vel_x', 'vel_y', 'vel_z',
                         'roll', 'pitch', 'yaw',
                         'ang_vel_x', 'ang_vel_y', 'ang_vel_z'])

    file = open(stat_filename, 'a')
    writer = csv.writer(file)

    # plotting
    if GUI:
        size = 1000
        x_vec = np.linspace(0, size, size + 1)[0:-1]
        y_vec = np.zeros(len(x_vec))
        line1 = []

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
            write_stat(episode, current_step, reward, writer, env)
            if i % env.SIM_FREQ == 0 and GUI:
                env.render()
            if current_step % N == 0:
                agent.learn()
                learn_iters += 1
            sync(i, start, env.TIMESTEP)
            if done:
                break
            if GUI:
                # y_vec[-1] = env._getDroneStateVector(0)[2]
                y_vec[-1] = reward
                line1 = live_plotter(x_vec, y_vec, line1)
                y_vec = np.append(y_vec[1:], 0.0)

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
