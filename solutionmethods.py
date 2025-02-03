
from typing import NamedTuple, Callable, Dict, List, Tuple
import functools
import jax
import jax.numpy as jnp
import jax.scipy as jsp
import haiku as hk
import numpy as np
import optax
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time
from itertools import product
import pickle
from flax import struct
from mynetworks import *
from myjaxutil import *
from markoveconomy import *
from markoveconomy import _eqm_network, _br_network

def compute_exploitability(economy: MarkovEconomy, eqm_params: hk.Params, br_network_lr: float, num_epochs: int, num_samples: int, num_episodes:int, print_iter = 1):
    """ Given a eqm network, compute the exploitability of the given eqm policy. """
    cumul_regrets = []
    prng = hk.PRNGSequence(SEED)

    # Initialize eqm network
    eqm_network = hk.without_apply_rng(hk.transform(_eqm_network))

    # Initialize best-response network
    br_network = hk.without_apply_rng(hk.transform(_br_network))
    br_params = br_network.init(next(prng), economy.get_initial_state(), eqm_network.apply(eqm_params, economy.get_initial_state(), economy), economy)

    # Initialize Optimizer for best-response network
    br_optimizer, br_opt_state = init_optimiser(br_network_lr, br_params)

    @jax.jit
    def cumulative_neural_loss(eqm_params:  hk.Params, br_params:  hk.Params):
        eqm_generator = Policy(eqm_network, eqm_params)
        br_discriminator = Policy(br_network, br_params)

        eqm_policy = lambda s: eqm_generator(s, economy) # type: ignore
        br_policy = lambda s: br_discriminator(s, eqm_generator(s, economy), economy) #type: ignore

        # consumers_br_seller_policy_profile = lambda s: policy_profile(s, economy)[:2] + br_policy_profile(s, policy_profile(s, economy), economy)[2:]
        # seller_br_consumers_policy_profile = lambda s: br_policy_profile(s, policy_profile(s, economy), economy)[:2] +  policy_profile(s, economy)[2:]

        cumul_regret = EconomySimulator.estimate_cumulative_regret(economy, eqm_policy, br_policy, num_samples, num_episodes) # type: ignore

        return cumul_regret


    @jax.jit
    def get_grads(eqm_params:  hk.Params, br_params:  hk.Params):
        cumul_regret, br_network_grad = jax.value_and_grad(cumulative_neural_loss, argnums=1)(eqm_params, br_params)
        return cumul_regret, br_network_grad

    @jax.jit
    def update_params(eqm_params, br_params, br_opt_state):
        cumul_regret, br_network_grad = get_grads(eqm_params, br_params)

        # Br Network Update
        br_network_grad = jax.tree_util.tree_map(lambda x: -x, br_network_grad)
        br_updates, br_opt_state = br_optimizer(br_network_grad, br_opt_state, br_params)
        br_params = optax.apply_updates(br_params, br_updates)

        return cumul_regret, br_params, br_opt_state

    for epoch in range(num_epochs):

        if epoch % print_iter == 0:
            print(f"==================== Epoch {epoch} ====================")

            cumul_regret = cumulative_neural_loss(eqm_params, br_params)
            print(f"Cumulative Regret: {cumul_regret}")
            cumul_regrets.append(cumul_regret)

        cumul_regret, br_params, br_opt_state = update_params(eqm_params, br_params, br_opt_state)

    br_policy = Policy(br_network, br_params)

    return cumul_regrets, br_policy



def exploitability_minimization(economy: MarkovEconomy, eqm_network_lr: float, br_network_lr: float, exp_lr: float, num_epochs: int, num_epochs_exp: int, num_samples: int, num_episodes:int, print_iter = 1, compute_exploit = False):
    """ Training Procedure. """
    cumul_regrets = []
    avg_excess_demands = []
    consumer_first_order_violations = []
    total_first_order_violations = []
    avg_bellman_errors = []
    exploitabilities = []
    prng = hk.PRNGSequence(SEED)

    # Initialize eqm network
    eqm_network = hk.without_apply_rng(hk.transform(_eqm_network))
    eqm_params = eqm_network.init(next(prng), economy.get_initial_state(), economy)


    # Initialize Optimizer for eqm network
    fast_eqm_network_lr = eqm_network_lr * 10
    fast_eqm_optimizer, fast_eqm_opt_state = init_optimiser(fast_eqm_network_lr, eqm_params)
    eqm_optimizer, eqm_opt_state = init_optimiser(eqm_network_lr, eqm_params)


    # Initialize best-response network
    br_network = hk.without_apply_rng(hk.transform(_br_network))
    br_params = br_network.init(next(prng), economy.get_initial_state(), eqm_network.apply(
                eqm_params, economy.get_initial_state(), economy), economy)


    # Initialize Optimizer for best-response network
    fast_br_network_lr = br_network_lr * 10
    fast_br_optimizer, fast_br_opt_state = init_optimiser(fast_br_network_lr, br_params)
    br_optimizer, br_opt_state = init_optimiser(br_network_lr, br_params)


    @jax.jit
    def cumulative_neural_loss(eqm_params:  hk.Params, br_params:  hk.Params):
        eqm_generator = Policy(eqm_network, eqm_params)
        br_discriminator = Policy(br_network, br_params)

        eqm_policy = lambda s: eqm_generator(s, economy) # type: ignore
        br_policy = lambda s: br_discriminator(s, eqm_generator(s, economy), economy) #type: ignore

        # consumers_br_seller_policy_profile = lambda s: policy_profile(s, economy)[:2] + br_policy_profile(s, policy_profile(s, economy), economy)[2:]
        # seller_br_consumers_policy_profile = lambda s: br_policy_profile(s, policy_profile(s, economy), economy)[:2] +  policy_profile(s, economy)[2:]

        cumul_regret = EconomySimulator.estimate_cumulative_regret(economy, eqm_policy, br_policy, num_samples, num_episodes) # type: ignore

        return cumul_regret


    @jax.jit
    def get_grads(eqm_params:  hk.Params, br_params:  hk.Params):
        cumul_regret, (eqm_network_grad, br_network_grad) = jax.value_and_grad(cumulative_neural_loss, argnums=[0, 1])(eqm_params, br_params)
        return cumul_regret, eqm_network_grad, br_network_grad

    @jax.jit
    def update_params(eqm_params, br_params, eqm_opt_state, br_opt_state):
        cumul_regret, eqm_network_grad, br_network_grad = get_grads(eqm_params, br_params)

        # Eqm Network Update
        eqm_updates, eqm_opt_state = eqm_optimizer(eqm_network_grad, eqm_opt_state, eqm_params)
        eqm_params = optax.apply_updates(eqm_params, eqm_updates)

        # Br Network Update
        br_network_grad = jax.tree_util.tree_map(lambda x: -x, br_network_grad)
        br_updates, br_opt_state = br_optimizer(br_network_grad, br_opt_state, br_params)
        br_params = optax.apply_updates(br_params, br_updates)

        return cumul_regret, (eqm_params, br_params), (eqm_opt_state, br_opt_state)


    @jax.jit
    def fast_update_params(eqm_params, br_params, eqm_opt_state, br_opt_state):
        cumul_regret, eqm_network_grad, br_network_grad = get_grads(eqm_params, br_params)

        # Eqm Network Update
        eqm_updates, eqm_opt_state = fast_eqm_optimizer(eqm_network_grad, eqm_opt_state, eqm_params)
        eqm_params = optax.apply_updates(eqm_params, eqm_updates)

        # Br Network Update
        br_network_grad = jax.tree_util.tree_map(lambda x: -x, br_network_grad)
        br_updates, br_opt_state = fast_br_optimizer(br_network_grad, br_opt_state, br_params)
        br_params = optax.apply_updates(br_params, br_updates)

        return cumul_regret, (eqm_params, br_params), (eqm_opt_state, br_opt_state)



    state = economy.get_initial_state()
    for epoch in range(num_epochs):

        if epoch <  1000:
            cumul_regret, (eqm_params, br_params), (eqm_opt_state, br_opt_state) = fast_update_params(eqm_params, br_params, eqm_opt_state, br_opt_state)
        else:
            cumul_regret, (eqm_params, br_params), (eqm_opt_state, br_opt_state) = update_params(eqm_params, br_params, eqm_opt_state, br_opt_state)

        if epoch % print_iter == 0:
            print(f"==================== Epoch {epoch} ====================")

            cumul_regret = cumulative_neural_loss(eqm_params, br_params)
            eqm_generator = Policy(eqm_network, eqm_params)
            eqm_policy = lambda s: eqm_generator(s, economy)
            excess_demand_violation = EconomySimulator.estimate_first_order_seller_violation(economy, eqm_policy, num_samples, num_episodes) # type: ignore
            consumer_first_order_violation = EconomySimulator.estimate_first_order_consumer_violation(economy, eqm_policy, num_samples, num_episodes) # type: ignore
            total_first_order_loss = excess_demand_violation + consumer_first_order_violation #type: ignore
            avg_bellman_error = EconomySimulator.estimate_avg_bellman_error(economy, eqm_policy, num_samples, num_episodes) # type: ignore

            commod_price, asset_price, consumption, portfolio = eqm_policy(state)
            print(f"commod_price: {commod_price}, asset_price: {asset_price}, consumption: {consumption}, portfolio: {portfolio}")
            print(f"Cumulative Regret: {cumul_regret}")
            print(f"Total First Order Violation: {total_first_order_loss}")
            print(f"Average excess demand: {excess_demand_violation}")
            print(f"Consumer first order violation: {consumer_first_order_violation}")
            print(f"Average Bellman Error: {avg_bellman_error}")


            cumul_regrets.append(cumul_regret)
            total_first_order_violations.append(total_first_order_loss)
            avg_excess_demands.append(excess_demand_violation)
            consumer_first_order_violations.append(consumer_first_order_violation)
            avg_bellman_errors.append(avg_bellman_error)

            if compute_exploit == True:
              exploits, _ = compute_exploitability(economy, eqm_params, exp_lr, num_epochs_exp, num_samples, num_episodes, print_iter=print_iter)
              print(f"Exploitabilities: {exploits[-1]}")
              exploitabilities.append(exploits[-1])


    eqm_policy = Policy(eqm_network, eqm_params)
    br_policy = Policy(br_network, br_params)

    return (cumul_regrets, total_first_order_violations, avg_excess_demands, consumer_first_order_violations, avg_bellman_errors), eqm_policy, br_policy, exploitabilities




def neural_projection_method(economy: MarkovEconomy, eqm_network_lr: float, exp_lr: float, balance_constant: float, num_epochs: int, num_epochs_exp: int, num_samples: int, num_episodes:int, print_iter = 1, compute_exploit = False):
    """ Training Procedure. """

    total_first_order_violations = []
    avg_excess_demands = []
    consumer_first_order_violations = []
    avg_bellman_errors = []
    exploitabilities = []

    prng = hk.PRNGSequence(SEED)

    # Initialize eqm network
    eqm_network = hk.without_apply_rng(hk.transform(_eqm_network))
    eqm_params = eqm_network.init(next(prng), economy.get_initial_state(), economy)


    # Initialize Optimizer for eqm network
    eqm_optimizer, eqm_opt_state = init_optimiser(eqm_network_lr, eqm_params)


    @jax.jit
    def neural_loss(eqm_params:  hk.Params):
        eqm_generator = Policy(eqm_network, eqm_params)
        eqm_policy = lambda s: eqm_generator(s, economy) # type: ignore
        first_order_violation = EconomySimulator.estimate_first_order_violation(economy, eqm_policy, num_samples, num_episodes) # type: ignore
        avg_bellman_error = EconomySimulator.estimate_avg_bellman_error(economy, eqm_policy, num_samples, num_episodes) # type: ignore

        return balance_constant * first_order_violation + (1 - balance_constant) * avg_bellman_error



    @jax.jit
    def get_grads(eqm_params:  hk.Params):
        loss, eqm_network_grad = jax.value_and_grad(neural_loss)(eqm_params)
        return loss, eqm_network_grad

    @jax.jit
    def update_params(eqm_params, eqm_opt_state):
        loss, eqm_network_grad = get_grads(eqm_params)

        # Eqm Network Update
        eqm_updates, eqm_opt_state = eqm_optimizer(eqm_network_grad, eqm_opt_state, eqm_params)
        eqm_params = optax.apply_updates(eqm_params, eqm_updates)

        return loss, eqm_params, eqm_opt_state

    for epoch in range(num_epochs):

        if epoch % print_iter == 0:
            print(f"==================== Epoch {epoch} ====================")

            eqm_generator = Policy(eqm_network, eqm_params)
            eqm_policy = lambda s: eqm_generator(s, economy)
            excess_demand_violation = EconomySimulator.estimate_first_order_seller_violation(economy, eqm_policy, num_samples, num_episodes) # type: ignore
            consumer_first_order_violation = EconomySimulator.estimate_first_order_consumer_violation(economy, eqm_policy, num_samples, num_episodes) # type: ignore
            total_first_order_loss = excess_demand_violation + consumer_first_order_violation #type: ignore
            avg_bellman_error = EconomySimulator.estimate_avg_bellman_error(economy, eqm_policy, num_samples, num_episodes) # type: ignore

            print(f"Loss: {total_first_order_loss+avg_bellman_error}")
            print(f"Average excess demand: {excess_demand_violation}")
            print(f"Consumer first order violation: {consumer_first_order_violation}")
            print(f"avg bellman error: {avg_bellman_error}")

            total_first_order_violations.append(total_first_order_loss)
            avg_excess_demands.append(excess_demand_violation)
            consumer_first_order_violations.append(consumer_first_order_violation)
            avg_bellman_errors.append(avg_bellman_error)

            if compute_exploit == True:
              exploits, _ = compute_exploitability(economy, eqm_params, exp_lr, num_epochs_exp, num_samples, num_episodes, print_iter=print_iter)
              print(f"Exploitabilities: {exploits[-1]}")
              exploitabilities.append(exploits[-1])


        _, eqm_params, eqm_opt_state = update_params(eqm_params, eqm_opt_state)



    eqm_policy = Policy(eqm_network, eqm_params)

    return (total_first_order_violations, avg_excess_demands, consumer_first_order_violations, avg_bellman_errors), eqm_policy, exploitabilities