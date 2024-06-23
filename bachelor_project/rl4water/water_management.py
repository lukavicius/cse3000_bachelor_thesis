import time
from pprint import pprint

import h5py
import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import nn

from core.learners.metrics import non_dominated
from core.learners.mones import MONES
from datetime import datetime
import uuid
from main import create_nile_river_env
import os, argparse


class Actor(nn.Module):
    def __init__(self, nS, nA, hidden=50):
        super(Actor, self).__init__()

        self.nA = nA
        self.fc1 = nn.Linear(nS, hidden)
        self.fc2 = nn.Linear(hidden, nA)

        nn.init.xavier_uniform_(self.fc1.weight, gain=1)
        nn.init.xavier_uniform_(self.fc2.weight, gain=1)

    def forward(self, state):
        # actor
        a = self.fc1(state.T)
        a = torch.tanh(a)
        a = self.fc2(a)
        return 100 * a


def train_agent(logdir, iterations=5, n_population=10, n_runs=1, parallel=False, directoryno=2, growth_coef=0.02,
                irr_lambda_std=0.01, evap_increase_coef=0.004, evap_lambda_mean=0.5,
                evap_lambda_std=0.5):
    epsilon = 1e-8

    ref_point_20_years = [0, -0.5, -2, 0]
    number_of_observations = 5
    number_of_actions = 4
    agent = MONES(
        create_nile_river_env,
        Actor(number_of_observations, number_of_actions, hidden=50),
        n_population=n_population,
        n_runs=n_runs,
        logdir=logdir,
        indicator="hypervolume",
        # TODO: Change depending on the time horizon
        ref_point=np.array(ref_point_20_years) + epsilon,
        parallel=parallel,
        directory=directoryno,
        growth_coef=growth_coef,
        irr_lambda_std=irr_lambda_std,
        evap_increase_coef=evap_increase_coef,
        evap_lambda_mean=evap_lambda_mean,
        evap_lambda_std=evap_lambda_std
    )
    timer = time.time()
    agent.train(iterations)
    agent.logger.put("train/time", time.time() - timer, 0, "scalar")
    print(f"Training took: {time.time() - timer} seconds")

    print("Logdir:", logdir)
    torch.save({"dist": agent.dist, "policy": agent.policy}, logdir + "checkpoint.pt")


def run_agent(logdir):
    # Load agent
    load_dir = os.path.join(logdir, "non_dominated_models")

    for model_file in range(1):
        checkpoint = torch.load(load_dir + f"/model_{model_file}.pt")
        print(checkpoint)
        agent = checkpoint

        timesteps = 240
        env = create_nile_river_env()
        obs, _ = env.reset()
        # print(obs)
        rewards = [0, 0, 0, 0]
        for _ in range(timesteps):
            action = agent.forward(torch.from_numpy(obs).float())
            action = action.detach().numpy().flatten()
            # action = [5, 5, 5, 5]
            # print("Action:")
            # pprint(action)
            (
                final_observation,
                final_reward,
                final_terminated,
                final_truncated,
                final_info,
            ) = env.step(action)
            rewards += final_reward
        print("Reward:")

        rewards[0] = yearly_avg_power_in_twh(rewards[0])
        rewards[1:3] = deficit_in_bcm_per_year(rewards[1:3])
        rewards[3] = frequency_of_month_below_min_level(rewards[3])
        pprint(rewards)
            # print("Observation:")
            # pprint(final_observation)

def yearly_avg_power_in_twh(power_in_mwh):
    return power_in_mwh / (20 * 1e6)

def deficit_in_bcm_per_year(deficit_in_m):
    return deficit_in_m * 3600 * 24 * 1e-9 / 20

def frequency_of_month_below_min_level(num_of_months):
    return num_of_months / 240


def show_logs(logdir):
    with h5py.File(logdir, "r") as f:
        # Print all root level object names (aka keys)
        # these can be group or dataset names
        print("Keys: %s" % f.keys())
        # get first object name/key; may or may NOT be a group
        a_group_key = list(f.keys())[0]

        # get the object type for a_group_key: usually group or dataset
        print(type(f[a_group_key]))

        # If a_group_key is a group name,
        # this gets the object names in the group and returns as a list
        data = list(f[a_group_key])
        print(data)
        print(list(f[list(f.keys())[1]]))

        params = f["params"]

        print("Iterations: \t", params["iterations"][0][1])
        print("N_populations: \t", params["n_population"][0][1])
        print("Parallel: \t", params["parallel"][0][1])

        group = f["train"]

        # print("Hypervolume:", group['hypervolume'][()])
        # print("Indicator metric:", group['metric'][()])
        print("ND returns:", non_dominated(group["returns"]["ndarray"][-1]))
        # print(group['returns']['step'][()])
        print("Training took", group["time"][0][1], "seconds")
        print(group["hypervolume"][0])
        plt.plot(group["hypervolume"][()][:, 0], group["hypervolume"][()][:, 1], marker=".")
        plt.show()


if __name__ == "__main__":
    # parser = argparse.ArgumentParser(description="Reservoir Simulation Script")
    # parser.add_argument("iterations", type=str, help="Description for param1")
    # parser.add_argument("n_population", type=str, help="Description for param2")
    # parser.add_argument("n_runs", type=str, help="Description for param1")
    # parser.add_argument("directory", type=str, help="Description for param1")
    # parser.add_argument("growth", type=str, help="Description for param2")
    # parser.add_argument("irr_lambda_std", type=str, help="Description for param1")
    # parser.add_argument("evap_increase", type=str, help="Description for param2")
    # parser.add_argument("evap_lambda_mean", type=str, help="Description for param1")
    # parser.add_argument("evap_lambda_std", type=str, help="Description for param2")
    #
    # args = parser.parse_args()
    #
    # logdir = "runs/"
    # logdir += datetime.now().strftime("%Y-%m-%d_%H-%M-%S_") + str(uuid.uuid4())[:4] + "/"
    #
    # train_agent(logdir, iterations=int(args.iterations), n_population=int(args.n_population), n_runs=int(args.n_runs),
    #             parallel=True, directoryno=int(args.directory), growth_coef=float(args.growth),
    #             irr_lambda_std=float(args.irr_lambda_std), evap_increase_coef=float(args.evap_increase),
    #             evap_lambda_mean=float(args.evap_lambda_mean), evap_lambda_std=float(args.evap_lambda_std))

    # Trained agent path
    temp = time.time()
    logdir = "runs/2024-06-14_07-49-45_c519/"
    #run_agent(logdir)
    print(time.time() - temp)
    # Read log file
    logdir = "runs/2024-06-14_07-49-45_c519/log.h5"
    show_logs(logdir)