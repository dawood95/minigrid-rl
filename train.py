import gym
import argparse
import numpy as np

import envs
from algos.a2c import A2C

parser = argparse.ArgumentParser()
parser.add_argument('-e', '--env-name', default="MiniGrid-SimpleEnv-9x9-v0")
parser.add_argument('--algo', default="a2c", choices=["a2c", "a2c-ppo", "a2r"])
parser.add_argument('--epoch', default=100)
parser.add_argument('--cuda', action="store_true", default=False)
args = parser.parse_args()

PROJECT_NAME = "unsupervised-advantage"

def main():
    # Initialize environment
    env = gym.make(args.env_name)

    # Initialize algorithm
    if args.algo == "a2c":
        trainer = A2C(env, cuda=args.cuda)
    else:
        assert False, "Not implemented yet"

    for epoch in range(args.epoch):
        print("Epoch #%d"%(epoch))
        trainer.train(100)
        trainer.val(40)
        trainer.checkpoint(epoch)

if __name__ == "__main__": main()
