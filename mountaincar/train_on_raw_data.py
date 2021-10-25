from __future__ import print_function

import os
import sys
import gym
import random
import datetime
import numpy as np
import multiprocessing
sys.path.insert(0, ".")

import neat
from neat import VectorizedSpikingPFeedForwardNetwork

random.seed(int(sys.argv[1]))
np.random.seed(int(sys.argv[1]))

min_ = np.array([-1.2, -0.07])
max_ = np.array([0.6, 0.07])
episodes = 20
scale = 20


def eval_genome(genome, config):
    net = VectorizedSpikingPFeedForwardNetwork.create(genome, config)
    env = gym.make("MountainCar-v0")
    rewards = []

    for ep in range(episodes):
        done = False
        env.seed(ep)
        obs = env.reset()
        rewards.append(0)

        while not done:
            yp = net.activate((obs - min_)/(max_ - min_) * scale)
            obs, rew, done, _ = env.step(np.argmax(yp))
            rewards[-1] += rew

    return np.mean(rewards)


def run(config_file):
    # Load configuration.
    config = neat.Config(neat.SNPSGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)

    # Create the population, which is the top-level object for a NEAT run.
    p = neat.Population(config)

    # Add a stdout reporter to show progress in the terminal.
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)

    # Run for up to 300 generations.
    pe = neat.ParallelEvaluator(multiprocessing.cpu_count(), eval_genome)
    winner = p.run(pe.evaluate, 300)
    return winner, config


if __name__ == '__main__':
    # Determine path to configuration file. This path manipulation is
    # here so that the script will run successfully regardless of the
    # current working directory.
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config-spiking-p')
    winner, cfg = run(config_path)

    dirname = os.path.join("logs", datetime.datetime.now().strftime("%d-%b-%y_%H-%M-%S"))
    os.makedirs(dirname, exist_ok=True)

    with open(os.path.join(dirname, "winner.txt"), "w") as f:
        f.write(str(winner))
