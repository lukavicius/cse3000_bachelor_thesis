import time
import torch
import torch.nn as nn
import numpy as np
import copy
from core.learners.metrics import non_dominated, non_dominated_rank, crowding_distance, compute_hypervolume
from core.log.logger import Logger
import concurrent.futures
import os


def n_parameters(model):
    return np.sum([torch.prod(torch.tensor(p.shape)) for p in model.parameters()])


def set_parameters(model, z):
    assert len(z) == n_parameters(model), "not providing correct amount of parameters"
    s = 0
    for p in model.parameters():
        n_p = torch.prod(torch.tensor(p.shape))
        p.data = z[s: s + n_p].view_as(p)
        s += n_p


def run_episode(env, model):
    e_r = 0
    done = False
    o, _ = env.reset()

    while not done:
        with torch.no_grad():
            action = model(torch.from_numpy(o).float()[:, None])
            action = action.detach().numpy().flatten()
        n_o, r, terminated, truncated, _ = env.step(action)
        e_r += r
        o = n_o
        if terminated or truncated:
            done = True
    return torch.from_numpy(e_r).float()


def indicator_hypervolume(points, ref, nd_penalty=0.0):
    # compute hypervolume of dataset
    nd_i = non_dominated(points, return_indexes=True)[1]
    nd = points[nd_i]
    hv = compute_hypervolume(nd, ref)
    # hypervolume without a point from dataset
    hv_p = np.zeros(len(points))
    # penalization if point is dominated
    is_nd = np.zeros(len(points), dtype=bool)
    for i in range(len(points)):
        is_nd[i] = np.any(np.all(nd == points[i], axis=1))
        if is_nd[i]:
            # if there is only one non-dominated point, and this point is non-dominated,
            # then it amounts for the full hypervolume
            if len(nd) == 1:
                hv_p[i] = 0.0
            else:
                # remove point from nondominated points, compute hv
                rows = np.all(nd == points[i], axis=1)
                hv_p[i] = compute_hypervolume(nd[np.logical_not(rows)], ref)
        # if point is dominated, no impact on hypervolume
        else:
            hv_p[i] = hv

    indicator = hv - hv_p - nd_penalty * np.logical_not(is_nd)
    return indicator


def indicator_non_dominated(points):
    ranks = non_dominated_rank(points)
    indicator = -ranks + crowding_distance(points, ranks=ranks)
    return indicator


def evaluate_individual(make_env, individual, n_runs, n_objectives, directoryno=2, growth_coef=0.02,
                irr_lambda_std=0.01, evap_increase_coef=0.004, evap_lambda_mean=0.5,
                evap_lambda_std=0.5):
    env = make_env(directoryno, growth_coef,
                irr_lambda_std, evap_increase_coef, evap_lambda_mean, evap_lambda_std)

    p_return = torch.zeros(n_runs, n_objectives)
    for r in range(n_runs):
        p_return[r] = run_episode(env, individual)
    return torch.mean(p_return, dim=0)


def evaluate_population_parallel(make_env, population, n_runs, n_objectives, directoryno=2, growth_coef=0.02,
                irr_lambda_std=0.01, evap_increase_coef=0.004, evap_lambda_mean=0.5,
                evap_lambda_std=0.5):
    returns = torch.zeros(len(population), n_objectives)

    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = []
        for i, individual in enumerate(population):
            futures.append(executor.submit(evaluate_individual, make_env, individual, n_runs, n_objectives, directoryno, growth_coef,
                irr_lambda_std, evap_increase_coef, evap_lambda_mean, evap_lambda_std))

        for i, future in enumerate(concurrent.futures.as_completed(futures)):
            returns[i] = future.result()

    return returns


class MONES(object):

    def __init__(
            self,
            make_env,
            policy,
            n_population=1,
            n_runs=1,
            indicator="non_dominated",
            ref_point=None,
            logdir="runs",
            parallel=False,
            directory=2,
            growth_coef=0.02,
            irr_lambda_std=0.01, evap_increase_coef=0.004,
            evap_lambda_mean=0.5, evap_lambda_std=0.5
    ):
        self.make_env = make_env
        self.policy = policy

        self.directory = directory
        self.growth_coef = growth_coef
        self.irr_lambda_std = irr_lambda_std
        self.evap_increase_coef = evap_increase_coef
        self.evap_lambda_mean = evap_lambda_mean
        self.evap_lambda_std = evap_lambda_std

        self.n_population = n_population
        self.n_runs = n_runs
        self.ref_point = ref_point
        env = make_env(self.directory, self.growth_coef, self.irr_lambda_std, self.evap_increase_coef,
                       self.evap_lambda_mean, self.evap_lambda_std)
        self.n_objectives = 1 if not hasattr(env, "reward_space") else len(env.reward_space.low)

        self.logdir = logdir
        self.logger = Logger(self.logdir)

        self.parallel = parallel

        self.nondominated_models = []

        if indicator == "hypervolume":
            assert ref_point is not None, "reference point is needed for hypervolume indicator"
            self.indicator = lambda points, ref=ref_point: indicator_hypervolume(points, ref)
        elif indicator == "non_dominated":
            self.indicator = indicator_non_dominated
        elif indicator == "single_objective":
            self.indicator = lambda x: x.flatten()
            self.n_objectives = 1
        else:
            raise ValueError("unknown indicator, choose between hypervolume and non_dominated")

        self.logger.put("params/n_population", n_population, 0, "scalar")
        self.logger.put("params/n_runs", n_runs, 0, "scalar")
        self.logger.put("params/parallel", parallel, 0, "scalar")

    def start(self):
        # make distribution
        n_params = n_parameters(self.policy)
        mu, sigma = torch.rand(n_params, requires_grad=True), torch.rand(n_params, requires_grad=True)
        mu, sigma = nn.Parameter(mu), nn.Parameter(sigma)
        self.dist = torch.distributions.Normal(mu, sigma)

        # optimizer to change distribution parameters
        self.opt = torch.optim.Adam([{"params": mu}, {"params": sigma}], lr=1e-1)

    def step(self):
        # using current theta, sample policies from Normal(theta)
        population, z = self.sample_population()
        # run episode for these policies
        eval_time = time.time()
        # run population on multiple cores
        if self.parallel:
            returns = evaluate_population_parallel(self.make_env, population, self.n_runs, self.n_objectives, self.directory, self.growth_coef, self.irr_lambda_std, self.evap_increase_coef,
                       self.evap_lambda_mean, self.evap_lambda_std)
        else:
            returns = self.evaluate_population(self.make_env(self.directory, self.growth_coef, self.irr_lambda_std, self.evap_increase_coef,
                       self.evap_lambda_mean, self.evap_lambda_std), population)
        returns = returns.detach().numpy()

        returns[:, 0] = self.yearly_avg_power_in_twh(returns[:, 0])
        returns[:, 1:3] = self.deficit_in_bcm_per_year(returns[:, 1:3])
        returns[:, 3] = self.frequency_of_month_below_min_level(returns[:, 3])

        print(f"Population evaluation took: {time.time() - eval_time} seconds")

        indicator_metric = self.indicator(returns)
        metric = torch.tensor(indicator_metric, dtype=torch.float64)[:, None]

        epsilon = 1e-8  # Small value for numerical stability
        # standardize the rewards to have a gaussian distribution
        metric = (metric - torch.mean(metric, dim=0)) / (torch.std(metric, dim=0) + epsilon)

        # compute loss
        log_prob = self.dist.log_prob(z).sum(1, keepdim=True)
        mu, sigma = self.dist.mean, self.dist.scale

        # directly compute inverse Fisher Information Matrix (FIM)
        # only works because we use gaussian (and no correlation between variables)
        fim_mu_inv = torch.diag(sigma.detach() ** 2 + epsilon)
        fim_sigma_inv = torch.diag(sigma.detach() ** 4 * 2 + epsilon)

        loss = -log_prob * metric

        # update distribution parameters
        self.opt.zero_grad()
        loss.mean().backward()

        # now that we have grads, multiply them with FIM_INV
        nat_grad_mu = fim_mu_inv @ mu.grad
        nat_grad_sigma = fim_sigma_inv @ sigma.grad
        mu.grad = nat_grad_mu
        sigma.grad = nat_grad_sigma

        self.opt.step()
        # using Adam results in negative sigma values, should not be necessary using SGD
        sigma.data = torch.abs(sigma.data)

        self.update_non_dominated_models(population, returns)

        return {"returns": returns, "metric": np.mean(indicator_metric), "population" : population}

    @staticmethod
    def yearly_avg_power_in_twh(power_in_mwh):
        return power_in_mwh / (20 * 1e6)

    @staticmethod
    def deficit_in_bcm_per_year(deficit_in_m):
        return deficit_in_m * 3600 * 24 * 1e-9 / 20

    @staticmethod
    def frequency_of_month_below_min_level(num_of_months):
        return (num_of_months) / 240
    def train(self, iterations):
        self.logger.put("params/iterations", iterations, 0, "scalar")

        self.start()

        for i in range(iterations):
            print(f"Started epoch number: {i}")
            info = self.step()
            returns = info["returns"]
            # logging
            self.logger.put("train/metric", info["metric"], i, "scalar")
            self.logger.put("train/returns", returns, i, f"{returns.shape[-1]}d")
            if self.ref_point is not None:
                print(f"Returns {returns} and ref point {self.ref_point}")
                hv = compute_hypervolume(returns, self.ref_point)
                self.logger.put("train/hypervolume", hv, i, "scalar")

            print(f'Iteration {i} \t Metric {info["metric"]} \t')

        print("=" * 20)
        print("DONE TRAINING, LAST POPULATION ND RETURNS")
        [res, indices] = non_dominated(returns, return_indexes=True)
        print(res)
        self.save_non_dominated_models()

    def sample_population(self):
        population = []
        z = []
        for _ in range(self.n_population):
            z_i = self.dist.sample()
            m_i = copy.deepcopy(self.policy)
            set_parameters(m_i, z_i)
            population.append(m_i)
            z.append(z_i)
        return population, torch.stack(z)

    def evaluate_population(self, env, population):
        returns = torch.zeros(len(population), self.n_objectives)
        for i in range(len(population)):
            print(f"Population \t {i + 1}/{len(population)}")
            p_return = torch.zeros(self.n_runs, self.n_objectives)
            for r in range(self.n_runs):
                p_return[r] = run_episode(env, population[i])
            returns[i] = torch.mean(p_return, dim=0)
        return returns

    def update_non_dominated_models(self, population, results):
        # Combine current population with their results
        to_add = list(zip(population, results))
        # Extend the list with existing non-dominated models
        to_add.extend(self.nondominated_models)

        # Extract results for non-dominated check
        results_to_check = np.array([result for _, result in to_add])

        # Call the non_dominated function (assuming it returns indices of non-dominated items)
        res, indices = non_dominated(results_to_check, return_indexes=True)

        # Filter non-dominated models using the indices
        self.nondominated_models = [to_add[i] for i in range(len(population)) if indices[i]]

    def save_non_dominated_models(self):
        save_dir = os.path.join(self.logdir, "non_dominated_models")
        os.makedirs(save_dir, exist_ok=True)

        for i, (model, result) in enumerate(self.nondominated_models):
            torch.save(model, os.path.join(save_dir, f"model_{i}.pt"))

