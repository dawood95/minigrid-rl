import gym
import argparse
import numpy as np
import torch

import envs
from algos.a2c import A2C

parser = argparse.ArgumentParser()
parser.add_argument('--model', required=True)
parser.add_argument('-e', '--env-name', default="MiniGrid-SimpleEnv-9x9-v0")
parser.add_argument('--algo', default="a2c", choices=["a2c", "a2c-ppo", "ua2c"])
parser.add_argument('--cuda', action="store_true", default=False)
args = parser.parse_args()

def main():
    env = gym.make(args.env_name)
    state_dict = torch.load(args.model, map_location=lambda storage, loc: storage)

    # Initialize algorithm
    if args.algo == "a2c":
        trainer = A2C(env, cuda=args.cuda, state_dict=state_dict)
    else:
        assert False, "Not implemented yet"

    trainer.visualize()

if __name__ == "__main__": main()
