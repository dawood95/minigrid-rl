import gym
import argparse
import numpy as np

import envs

parser = argparse.ArgumentParser()
parser.add_argument('-e', '--env-name', default="MiniGrid-SimpleEnv-9x9-v0")
args = parser.parse_args()

def main():

    # Initialize environment
    env = gym.make(args.env_name)
    env.reset()

    while True:
        renderer = env.render('human')
        obs, reward, done, info = env.step(env.action_space.sample())
        if done or renderer.window == None:
            break

if __name__ == "__main__":
    main()
