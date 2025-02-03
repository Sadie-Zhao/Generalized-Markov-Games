# Set Pylance to ignore jnp.ndarray errors
# pyright: reportPrivateImportUsage = none

from typing import NamedTuple, Callable, Dict, List, Tuple
import functools
import jax
import jax.numpy as jnp
import haiku as hk
from flax import struct
from mynetworks import *
from myjaxutil import *


#Set seed
SEED = 42
ZERO_CLIP_MIN = 1e-4

@struct.dataclass
class UtilityFunction:
    """A class that stores a utility function for one or multiple consumers with the same utility function consumer_type.
    Attributes:
        utility_type: A string that specifies the type of utility function.
    """

    utility_type: str = struct.field(pytree_node=False)

    def __call__(self, consumption: jnp.ndarray, consumer_type: Dict) -> jnp.ndarray:
        """Applies the utility function to an consumption and a valuation.
        Args:
            consumption: A jnp.ndarray of shape (num_agents, num_commods) representing the consumption.
            consumer_type: A dictionary of parameters for the utility function.
        """

        def get_ces_utility(consumption, valuation, rho):
            # Returns utility value for CES agent
            rho = rho.reshape(-1, 1)
            return jnp.sum(jnp.power(jnp.power(consumption, rho) * valuation, (1/rho)), axis = 1).clip(ZERO_CLIP_MIN)

        # Linear utility function: Perfect Substitutes
        def get_linear_utility(consumption, valuation):
            return jnp.sum(consumption*valuation, axis = 1).clip(ZERO_CLIP_MIN)

        # Cobb-Douglas utility function
        def get_CD_utility(consumption, valuation):
            # For ease of calculation, normalize valautions
            # This does not change the preference relation that the utility function represents
            normalized_vals = valuation / jnp.sum(valuation, keepdims=True)
            util = jnp.prod(jnp.power(consumption, normalized_vals), axis = 1).clip(ZERO_CLIP_MIN)
            # print(util)
            return util

        # Leontief utility function: Perfect Complements
        def get_leontief_utility(consumption, valuation):
            return jnp.min(consumption / valuation, axis = 1).clip(ZERO_CLIP_MIN)

        get_utils = {"CES": get_ces_utility, "linear": get_linear_utility, "Cobb-Douglas": get_CD_utility, "Leontief": get_leontief_utility}


        return get_utils[self.utility_type](consumption, **consumer_type)

@struct.dataclass
class SpotMarket:
    """A class that stores the state of the spot market."""
    consumer_endow: jnp.ndarray
    consumer_type: Dict


@struct.dataclass
class State:
    world_state: jnp.ndarray
    spot_market: SpotMarket

    def replace_state(self, new_world_state, **new_spot_market_params):

        new_spot_market = self.spot_market.replace(**new_spot_market_params) # type: ignore
        new_state = self.replace(world_state = new_world_state, spot_market = new_spot_market) # type: ignore

        return new_state


    def replace_world_state(self, new_world_state):

        new_state = self.replace(world_state = new_world_state) # type: ignore

        return new_state

    def replace_spot_market(self, **new_spot_market_params):

        new_spot_market = self.spot_market.replace(new_spot_market_params) # type: ignore
        new_state = self.replace(spot_market = new_spot_market) # type: ignore

        return new_state


@struct.dataclass
class MarkovEconomy:
    """A class that stores the parameters of a Markov Economy.
    Attributes:
        num_consumers: An integer representing the number of consumers in the economy.
        num_commods: An integer representing the number of goods in the economy.
        num_assets: An integer representing the number of assets in the economy.
        num_world_states: An integer representing the number of world states in the economy.
        util: A utility function for the consumers in the economy.
        asset_return_list: A jnp.ndarray representing the returns of the assets in each world state.
        transition_function: A function that takes the current state, consumption and portfolio and returns the next state.
        discount: A float representing the discount factor.
        init_state_function: A function that returns the initial state of the economy.
    """
    num_consumers: int = struct.field(pytree_node=False)
    num_commods: int = struct.field(pytree_node=False)
    num_assets: int = struct.field(pytree_node=False)
    num_world_states: int = struct.field(pytree_node=False)
    util: UtilityFunction = struct.field(pytree_node=False)
    asset_return: jnp.ndarray = struct.field(pytree_node=False)
    transition_function: Callable[["MarkovEconomy", State, jnp.ndarray, jnp.ndarray], State] = struct.field(pytree_node=False)
    discount: float = struct.field(pytree_node=False)
    init_state_function: Callable[["MarkovEconomy"], State] = struct.field(pytree_node=False)
    consumer_endow_range: Tuple[float, float] = struct.field(pytree_node=False)
    consumer_type_range: Tuple[float, float] = struct.field(pytree_node=False)
    asset_return_range: Tuple[float, float] = struct.field(pytree_node=False)
    commod_price_space_sum: float = struct.field(pytree_node=False)
    asset_price_range: Tuple[float, float] = struct.field(pytree_node=False)
    portfolio_range: Tuple[float, float] = struct.field(pytree_node=False)


    def get_next_state(self, state: State, consumption: jnp.ndarray, portfolio: jnp.ndarray) -> State:

        new_state = self.transition_function(self, state, consumption, portfolio)

        # Observe new world state and spot market
        new_world_state = new_state.world_state
        new_spot_market = new_state.spot_market

        # Calculate asset payouts
        asset_payout = portfolio @ self.asset_return[new_world_state]

        # Update Spot Market State
        new_consendow = new_spot_market.consumer_endow + asset_payout
        new_types = new_spot_market.consumer_type


        next_state = state.replace_state(new_world_state, consumer_endow = new_consendow, consumer_type = new_types)

        return next_state

    def get_initial_state(self) -> State:
        return self.init_state_function(self)

@struct.dataclass
class Policy:
    """A class that stores network and parameters for a policy, and applies the policy to a state.
    Attributes:
        network: A Haiku network.
        params: The parameters of the network.
    """

    network: hk.Transformed = struct.field(pytree_node=False)
    params: hk.Params

    def __call__(self, state: State, *args) -> jnp.ndarray:
        """Applies the policy to a state.
        Args:
            state: A state of the economy.
        Returns:
            The output of the network for the given state.
        """
        return self.network.apply(self.params, state, *args)

@struct.dataclass
class Episode:
    """A trajectory storage.
    Attributes:
        states: States of the trajectory
        actions: Actions of the trajectory
        rewards: Rewards of the trajectory
    """

    state: State
    commod_price: jnp.ndarray
    asset_price: jnp.ndarray
    consumption: jnp.ndarray
    portfolio: jnp.ndarray
    rewards: jnp.ndarray = struct.field(default_factory=jnp.ndarray)

    def get_transition(self, t):
        """Gets the SA(B)R tuple at time t.
        Args:
            t: The time step.
        Returns:
            The transition at time t.
        """
        return jax.tree_util.tree_map(lambda x: x[t], self)
    




class EconomySimulator:

    @staticmethod
    @jax.jit
    def get_consumer_rewards(economy: MarkovEconomy, state: State, consumption: jnp.ndarray) -> jnp.ndarray:
        """A function that takes the consumption and returns the rewards of the consumers.
        Args:
            economy: A MarkovEconomy object.
            state: A state of the economy.
            consumption: A jnp.ndarray of shape (num_consumers, num_commods) representing the consumption of each consumer.

        Returns:
            The rewards of the consumers.
        """
        spot_market = state.spot_market
        consumer_type = spot_market.consumer_type

        return economy.util(consumption, consumer_type)

    @staticmethod
    @jax.jit
    def get_seller_reward(state: State, commod_price: jnp.ndarray, asset_price: jnp.ndarray, consumption: jnp.ndarray, portfolio: jnp.ndarray) -> jnp.ndarray:
        """ A function that takes the commodity and asset prices and returns the reward of the seller.
        Args:
            economy: A MarkovEconomy object.
            commod_price: A jnp.ndarray of shape (num_commods,) representing the price of the goods.
            asset_price: A jnp.ndarray of shape (num_assets,) representing the price of the assets.
            consumption: A jnp.ndarray of shape (num_consumers, num_commods) representing the consumption of each consumer.
            portfolio: A jnp.ndarray of shape (num_consumers, num_assets) representing the portfolio of each consumer.

        Returns:
            The reward of the seller.
        """

        spot_market = state.spot_market
        consumer_endow = spot_market.consumer_endow

        excess_commod_demand = jnp.sum(consumption, axis = 0) - jnp.sum(consumer_endow, axis = 0)
        excess_asset_demand = jnp.sum(portfolio, axis = 0)

        return commod_price.T @ excess_commod_demand  + asset_price.T @ excess_asset_demand



    @staticmethod
    @jax.jit
    def get_rewards(economy: MarkovEconomy, state: State, commod_price: jnp.ndarray, asset_price: jnp.ndarray, consumption: jnp.ndarray, portfolio: jnp.ndarray) -> jnp.ndarray:
        """A function that takes the current state, consumption and portfolio and returns the rewards.
        Args:
            economy: A MarkovEconomy object.
            state: A state of the economy.
            commod_price: A jnp.ndarray of shape (num_commods,) representing the price of the goods.

            consumption: A jnp.ndarray of shape (num_consumers, num_commods) representing the consumption of each consumer.
            portfolio: A jnp.ndarray of shape (num_consumers, num_assets) representing the portfolio of each consumer.

        Returns:
            The rewards of the of the consumers and seller.
        """

        consumer_rewards = EconomySimulator.get_consumer_rewards(economy, state, consumption)
        seller_reward = EconomySimulator.get_seller_reward(state, commod_price, asset_price, consumption, portfolio)
        rewards = jnp.concatenate([consumer_rewards, seller_reward[..., jnp.newaxis]], axis = 0)

        return rewards

    @staticmethod
    @jax.jit
    def step(economy: MarkovEconomy, state: State, consumption: jnp.ndarray, portfolio: jnp.ndarray) -> State:
        """A function that takes the current state, consumption and portfolio and returns the next state.
        Args:
            economy: A MarkovEconomy object.
            state: A state of the economy.
            consumption: A jnp.ndarray of shape (num_consumers, num_commods) representing the consumption of each consumer.
            portfolio: A jnp.ndarray of shape (num_consumers, num_assets) representing the portfolio of each consumer.

        Returns:
            The next state of the economy.
        """

        next_state = economy.get_next_state(state, consumption, portfolio)

        return next_state


    @staticmethod
    @jax.jit
    def get_rewards_and_step(economy: MarkovEconomy, state: State, commod_price: jnp.ndarray, asset_price: jnp.ndarray, consumption: jnp.ndarray, portfolio: jnp.ndarray) -> Tuple[jnp.ndarray, State]:
        """A function that takes the current state, consumption and portfolio and returns the rewards and next state.
        Args:
            economy: A MarkovEconomy object.
            state: A state of the economy.
            commod_price: A jnp.ndarray of shape (num_commods,) representing the price of each commodity.
            asset_price: A jnp.ndarray of shape (num_assets,) representing the price of each asset.
            consumption: A jnp.ndarray of shape (num_consumers, num_commods) representing the consumption of each consumer.
            portfolio: A jnp.ndarray of shape (num_consumers, num_assets) representing the portfolio of each consumer.

        Returns:
            The rewards of the of the consumers and seller and the next state of the economy.
        """
        rewards = EconomySimulator.get_rewards(economy, state, commod_price, asset_price, consumption, portfolio)
        next_state = EconomySimulator.step(economy, state, consumption, portfolio)

        return (rewards, next_state)


    @staticmethod
    @functools.partial(jax.jit, static_argnames=["policy_profile", "num_episodes"])
    def sample_state_value(economy: MarkovEconomy, state: State, policy_profile, num_episodes: int) -> jnp.ndarray:
        discount = economy.discount
        init_cumul_rewards = jnp.repeat(0.0, repeats = economy.num_consumers + 1)
        init_state = state
        init_val  = (init_cumul_rewards, init_state)

        @jax.jit
        def episode_step(episode_num, episode_state):
            cumul_rewards, state = episode_state
            commod_price, asset_price, consumption, portfolio = policy_profile(state)
            reward, next_state = EconomySimulator.get_rewards_and_step(economy, state, commod_price, asset_price, consumption, portfolio)
            cumul_rewards += (discount**episode_num)*reward

            return (cumul_rewards, next_state)

        cumul_rewards, state = jax.lax.fori_loop(0, num_episodes, episode_step, init_val)

        return cumul_rewards

    @staticmethod
    @functools.partial(jax.jit, static_argnames=["policy_profile", "num_episodes"])
    def sample_trajectory(economy: MarkovEconomy, policy_profile, num_episodes: int) -> jnp.ndarray:
        discount = economy.discount
        init_state = economy.get_initial_state()

        @jax.jit
        def episode_step(state, traj):
            commod_price, asset_price, consumption, portfolio = policy_profile(state)
            next_state = EconomySimulator.step(economy, state, consumption, portfolio)

            # Return next state and transition
            episode = Episode(state, commod_price, asset_price, consumption, portfolio)
            return next_state, episode

        # Get trajectory
        final_state, trajectory = jax.lax.scan(episode_step, init_state, None, length = num_episodes)

        return trajectory

    @staticmethod
    @functools.partial(jax.jit, static_argnames=[ "policy_profile",  "num_episodes"])
    def sample_trajectory_and_payoff(economy: MarkovEconomy, policy_profile, num_episodes: int) -> jnp.ndarray:
        discount = economy.discount
        init_state = economy.get_initial_state()

        @jax.jit
        def episode_step(episode_num, episode_state):
            cumul_rewards, trajectory, state = episode_state
            commod_price, asset_price, consumption, portfolio = policy_profile(state)
            reward, next_state = EconomySimulator.get_rewards_and_step(economy, state, commod_price, asset_price, consumption, portfolio)
            cumul_rewards += (discount**episode_num)*reward
            trajectory += Episode(state, commod_price, asset_price, consumption, portfolio, reward)

            return (cumul_rewards, trajectory, next_state)

        cumul_rewards, state = jax.lax.fori_loop(0, num_episodes, episode_step, (0.0, [], init_state))

        return cumul_rewards

    @staticmethod
    @functools.partial(jax.jit, static_argnames=[ "policy_profile", "num_samples", "num_episodes"])
    def estimate_state_value(economy: MarkovEconomy, state: State, policy_profile, num_samples: int, num_episodes: int) -> jnp.ndarray:
        """Returns an estimate of the state value function from a batch of state value samples at a given state.
        Args:
            economy: A MarkovEconomy object.
            state: A state of the economy.
            policy_profile: A function that takes a state and returns the policy.
            num_samples: An integer representing the number of samples of state value samples to draw.
            num_episodes: An integer representing the number of episodes to sample.
        Returns:
            An estimate of the state value function at the given state.
        """
        num_samples_list = jnp.arange(num_samples)
        enumerated_sample_state_value = lambda i: EconomySimulator.sample_state_value(economy, state, policy_profile, num_episodes)
        state_value_estimate = jnp.mean(jax.vmap(enumerated_sample_state_value)(num_samples_list), axis = 0)

        return state_value_estimate

    @staticmethod
    @functools.partial(jax.jit, static_argnames=["policy_profile", "num_episodes"])
    def sample_expected_future_value(economy: MarkovEconomy, state: State, commod_price: jnp.ndarray, asset_price: jnp.ndarray, consumption: jnp.ndarray, portfolio: jnp.ndarray, policy_profile, num_episodes: int) -> jnp.ndarray:
        next_state = EconomySimulator.step(economy, state, consumption, portfolio)
        state_value_sample = EconomySimulator.sample_state_value(economy, next_state, policy_profile, num_episodes)

        return state_value_sample

    @staticmethod
    @functools.partial(jax.jit, static_argnames=[ "policy_profile", "num_samples", "num_episodes"])
    def estimate_expected_future_value(economy: MarkovEconomy, state: State, commod_price: jnp.ndarray, asset_price: jnp.ndarray, consumption: jnp.ndarray, portfolio: jnp.ndarray, policy_profile, num_samples: int, num_episodes: int) -> jnp.ndarray:
        """Returns an estimate of the expected future value function from a batch of expected future value samples at a given state.
        Args:
            economy: A MarkovEconomy object.
            state: A state of the economy.
            commod_price: A jnp.ndarray of shape (num_commods,) representing the price of the goods.
            asset_price: A jnp.ndarray of shape (num_assets,) representing the price of the assets.
            consumption: A jnp.ndarray of shape (num_consumers, num_commods) representing the consumption of each consumer.
            portfolio: A jnp.ndarray of shape (num_consumers, num_assets) representing the portfolio of each consumer.
            policy_profile: A function that takes a state and returns the policy.
            num_samples: An integer representing the number of samples of expected future value samples to take.
            num_episodes: An integer representing the number of episodes to sample.
        Returns:
            An estimate of the expected future value function at the given state.
        """
        num_samples_list = jnp.arange(num_samples)
        enumerated_sample_state_value = lambda i: EconomySimulator.sample_expected_future_value(economy, state, commod_price, asset_price, consumption, portfolio, policy_profile, num_episodes)
        state_value_estimate = jnp.mean(jax.vmap(enumerated_sample_state_value)(num_samples_list), axis = 0)

        return state_value_estimate


    @staticmethod
    @functools.partial(jax.jit, static_argnames=[ "policy_profile", "num_episodes"])
    def sample_action_value(economy: MarkovEconomy, state: State, commod_price: jnp.ndarray, asset_price: jnp.ndarray, consumption: jnp.ndarray, portfolio: jnp.ndarray, policy_profile, num_episodes: int) -> jnp.ndarray:
        """Returns a sample of the action value function at a given state.
        Args:
            economy: A MarkovEconomy object.
            state: A state of the economy.
            commod_price: A jnp.ndarray of shape (num_commods,) representing the price of the goods.
            asset_price: A jnp.ndarray of shape (num_assets,) representing the price of the assets.
            consumption: A jnp.ndarray of shape (num_consumers, num_commods) representing the consumption of each consumer.
            portfolio: A jnp.ndarray of shape (num_consumers, num_assets) representing the portfolio of each consumer.
            policy_profile: A function that takes a state and returns the policy.
            num_episodes: An integer representing the number of episodes to sample.
        Returns:
            A sample of the action value function at the given state.
        """
        discount = economy.discount

        rewards, next_state = EconomySimulator.get_rewards_and_step(economy, state, commod_price, asset_price, consumption, portfolio)
        state_value_sample = EconomySimulator.sample_state_value(economy, next_state, policy_profile, num_episodes)
        action_value_sample = rewards + discount*state_value_sample

        return action_value_sample

    @staticmethod
    @functools.partial(jax.jit, static_argnames=[ "policy_profile", "num_samples", "num_episodes"])
    def estimate_action_value(economy: MarkovEconomy, state: State, commod_price: jnp.ndarray, asset_price: jnp.ndarray, consumption: jnp.ndarray, portfolio: jnp.ndarray, policy_profile, num_samples: int, num_episodes: int) -> jnp.ndarray:
        """Returns an estimate of the action value function from a batch of action value samples at a given state.
        Args:
            economy: A MarkovEconomy object.
            state: A state of the economy.
            commod_price: A jnp.ndarray of shape (num_commods,) representing the price of the goods.
            asset_price: A jnp.ndarray of shape (num_assets,) representing the price of the assets.
            consumption: A jnp.ndarray of shape (num_consumers, num_commods) representing the consumption of each consumer.
            portfolio: A jnp.ndarray of shape (num_consumers, num_assets) representing the portfolio of each consumer.
            policy_profile: A function that takes a state and returns the policy.
            num_samples: An integer representing the number of samples of action value samples to take.
            num_episodes: An integer representing the number of episodes to sample.
        Returns:
            An estimate of the action value function at the given state.
        """
        num_samples_list = jnp.arange(num_samples)
        enumerated_sample_action_value = lambda i: EconomySimulator.sample_action_value(economy, state, commod_price, asset_price, consumption, portfolio, policy_profile, num_episodes)
        action_value_estimate = jnp.mean(jax.vmap(enumerated_sample_action_value)(num_samples_list), axis = 0)

        return action_value_estimate


    @staticmethod
    @functools.partial(jax.jit, static_argnames=["policy_profile", "num_samples", "num_episodes"])
    def estimate_payoff(economy: MarkovEconomy, policy_profile, num_samples, num_episodes: int):
        init_state_samples = EconomySimulator.sample_state(economy, num_samples)

        enumerated_sample_state_value = lambda s: EconomySimulator.sample_state_value(economy, s, policy_profile, num_episodes)
        state_value_sample = jnp.mean(jax.vmap(enumerated_sample_state_value)(init_state_samples), axis = 0)

        return state_value_sample

    @staticmethod
    @functools.partial(jax.jit, static_argnames=["policy_profile", "br_policy_profile", "num_samples", "num_episodes"])
    def estimate_state_cumulative_regret(economy: MarkovEconomy, state: State, policy_profile: Callable[[State], Tuple], br_policy_profile: Callable[[State], Tuple], num_samples: int, num_episodes: int):
        num_consumers = economy.num_consumers

        consumers_br_seller_policy_profile = lambda s: policy_profile(s)[:2] + br_policy_profile(s)[2:]
        seller_br_consumers_policy_profile = lambda s: br_policy_profile(s)[:2] +  policy_profile(s)[2:]

        consumers_br_payoff = EconomySimulator.estimate_state_value(economy, state, consumers_br_seller_policy_profile, num_samples, num_episodes)[:num_consumers]
        sellers_br_payoff = EconomySimulator.estimate_state_value(economy, state, seller_br_consumers_policy_profile, num_samples, num_episodes)[num_consumers:]

        payoff = EconomySimulator.estimate_state_value(economy, state, policy_profile, num_samples, num_episodes)
        consumers_payoff = payoff[:num_consumers]
        sellers_payoff = payoff[num_consumers:]

        consumers_cumul_regret = jnp.sum(consumers_br_payoff - consumers_payoff).squeeze()
        seller_cumul_regret = (sellers_br_payoff - sellers_payoff).squeeze()

        return consumers_cumul_regret + seller_cumul_regret

    @staticmethod
    @functools.partial(jax.jit, static_argnames=["policy_profile", "br_policy_profile", "num_samples", "num_episodes"])
    def estimate_cumulative_regret(economy: MarkovEconomy, policy_profile: Callable[[State], Tuple], br_policy_profile: Callable[[State], Tuple], num_samples:int, num_episodes: int):
        num_consumers = economy.num_consumers

        consumers_br_seller_policy_profile = lambda s: policy_profile(s)[:2] + br_policy_profile(s)[2:]
        seller_br_consumers_policy_profile = lambda s: br_policy_profile(s)[:2] +  policy_profile(s)[2:]

        consumers_br_payoff = EconomySimulator.estimate_payoff(economy, consumers_br_seller_policy_profile, num_samples, num_episodes)[:num_consumers]
        sellers_br_payoff = EconomySimulator.estimate_payoff(economy, seller_br_consumers_policy_profile, num_samples, num_episodes)[num_consumers:]

        payoff = EconomySimulator.estimate_payoff(economy, policy_profile, num_samples, num_episodes)

        consumers_payoff = payoff[:num_consumers]
        sellers_payoff = payoff[num_consumers:]

        consumers_cumul_regret = jnp.sum(consumers_br_payoff - consumers_payoff).squeeze()
        seller_cumul_regret = (sellers_br_payoff - sellers_payoff).squeeze()

        return consumers_cumul_regret + seller_cumul_regret

    ############### Market Statistic Functions ##################

    @staticmethod
    @jax.jit
    def get_state_excess_demand(state: State, consumption: jnp.ndarray, portfolio: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """A function that takes the current state, consumption and portfolio and returns the excess demand of the goods and assets.
        Args:
            economy: A MarkovEconomy object.
            state: A state of the economy.
            consumption: A jnp.ndarray of shape (num_consumers, num_commods) representing the consumption of each consumer.
            portfolio: A jnp.ndarray of shape (num_consumers, num_assets) representing the portfolio of each consumer.
        Returns:
            The excess demand of the goods and assets.
        """
        spot_market = state.spot_market
        consumer_endow = spot_market.consumer_endow
        excess_commod_demand = jnp.sum(consumption, axis = 0) - jnp.sum(consumer_endow, axis = 0)
        excess_asset_demand = jnp.sum(portfolio, axis = 0)

        return (excess_commod_demand, excess_asset_demand)

    @staticmethod
    @jax.jit
    def get_excess_spendings(state: State, commod_price: jnp.ndarray, asset_price: jnp.ndarray, consumption: jnp.ndarray, portfolio: jnp.ndarray) -> jnp.ndarray:
        """A function that takes the current state, consumption and portfolio and returns the excess spending of the consumers.
        Args:
            state: A state of the economy.
            commod_price: A jnp.ndarray of shape (num_commods,) representing the price of the goods.
            asset_price: A jnp.ndarray of shape (num_assets,) representing the price of the assets.
            consumption: A jnp.ndarray of shape (num_consumers, num_commods) representing the consumption of each consumer.
            portfolio: A jnp.ndarray of shape (num_consumers, num_assets) representing the portfolio of each consumer.
        Returns:
            The excess spending of the consumers.
        """
        spot_market = state.spot_market
        consumer_endow = spot_market.consumer_endow

        excess_spendings =  consumer_endow @ commod_price +  portfolio @ asset_price

        return excess_spendings


    @staticmethod
    @functools.partial(jax.jit, static_argnames=[ "policy_profile", "num_samples", "num_episodes"])
    def estimate_state_best_response_and_value(economy: MarkovEconomy, state: State, policy_profile: Callable[[State], Tuple], num_samples: int, num_episodes: int, solver_params = {}):
        """ Estimate the best response commodity prices, asset prices, consumptions, and portfolio to a given policy at a given state.
        Args:
            economy: A MarkovEconomy object.
            state: A state of the economy.
            policy_profile: A function that takes a state and returns the policy.
            num_samples: An integer representing the number of samples of action value samples to take.
            num_episodes: An integer representing the number of episodes to sample.
        Returns:
            The best responses the given policy and associated best-response state-values at the given state.
        """
        num_commods = economy.num_commods
        num_assets = economy.num_assets
        consumer_endow_max = economy.consumer_endow_range[1]
        price_space_sum = economy.commod_price_space_sum

        commod_price, asset_price, consumption, portfolio = policy_profile(state)

        excess_commod_demand, excess_asset_demand = EconomySimulator.get_state_excess_demand(state, consumption, portfolio)

        br_commod_price = jax.nn.one_hot(jnp.argmax(excess_commod_demand), num_commods)*price_space_sum
        br_asset_price = jnp.clip(excess_asset_demand, a_min= 0.0, a_max = 1.0)*consumer_endow_max

        br_seller_state_value = EconomySimulator.estimate_action_value(economy, state, br_commod_price, br_asset_price, consumption, portfolio, policy_profile, num_samples, num_episodes)[-1]

        # Get best response consumption and portfolio
        br_action_value_function = lambda action: EconomySimulator.estimate_action_value(economy, state, commod_price, asset_price, action[0], action[1], policy_profile, num_samples, num_episodes)

        br_consumer_state_value, (br_consumption, br_portfolio) = maximize(br_action_value_function, (consumption, portfolio), proj=lambda x: x)

        br_state_value = jnp.concatenate([br_consumer_state_value, br_seller_state_value], axis = 0)

        return br_state_value, (br_commod_price, br_asset_price, br_consumption, br_portfolio)

    @staticmethod
    @functools.partial(jax.jit, static_argnames=["policy_profile", "num_samples", "num_episodes"])
    def estimate_state_bellman_error(economy: MarkovEconomy, state: State, policy_profile: Callable[[State], Tuple], num_samples: int, num_episodes: int, solver_params = {}):
        """ Compute the Bellman error of a policy at a given state.
        Args:
            economy: A MarkovEconomy object.
            state: A state of the economy.
            policy_profile: A function that takes a state and returns the policy.
            num_episodes: An integer representing the number of episodes to sample.
        Returns:
            The Bellman error of the policy at the given state.
        """

        num_world_states = economy.num_world_states
        num_commods = economy.num_commods
        num_assets = economy.num_assets
        num_consumers = economy.num_consumers
        consendow_range = economy.consumer_endow_range
        type_endow_range = economy.consumer_type_range


        policy_state_value = EconomySimulator.estimate_state_value(economy, state, policy_profile, num_samples, num_episodes)

        commod_price, asset_price, consumption, portfolio = policy_profile(state)
        policy_action_value = EconomySimulator.estimate_action_value(economy, state, commod_price, asset_price, consumption, portfolio, policy_profile, num_samples, num_episodes)

        bellman_error = jnp.mean(jnp.square(policy_state_value - policy_action_value))

        return  bellman_error

    @staticmethod
    @functools.partial(jax.jit, static_argnames=["policy_profile", "num_samples", "num_episodes"])
    def estimate_avg_bellman_error(economy: MarkovEconomy, policy_profile: Callable[[State], Tuple], num_samples: int, num_episodes: int, solver_params = {}):
        """ Compute the average Bellman error of a policy at any state.
        Args:
            economy: A MarkovEconomy object.
            policy_profile: A function that takes a state and returns the policy.
            num_episodes: An integer representing the number of episodes to sample.
        Returns:
            The avergae Bellman error of the policy at any state.
        """

        state_samples = EconomySimulator.sample_state(economy, num_samples)

        enumerated_sample_state_bellman_error = lambda s: EconomySimulator.estimate_state_bellman_error(economy, s, policy_profile, num_samples, num_episodes)
        mean_bellman_error = jnp.mean(jax.vmap(enumerated_sample_state_bellman_error)(state_samples), axis = 0)

        return mean_bellman_error


    @staticmethod
    @functools.partial(jax.jit, static_argnames=["num_samples"])
    def sample_state(economy: MarkovEconomy, num_samples):
        num_world_states = economy.num_world_states
        num_commods = economy.num_commods
        num_assets = economy.num_assets
        num_consumers = economy.num_consumers
        consendow_range = economy.consumer_endow_range
        type_endow_range = economy.consumer_type_range

        world_states_samples = jax.random.randint(jax.random.PRNGKey(0), shape = (num_samples,), minval = 0, maxval = num_world_states)
        consumer_endow_samples = jax.random.uniform(jax.random.PRNGKey(0), shape = (num_samples, num_consumers, num_commods), minval = consendow_range[0], maxval = consendow_range[1])
        consumer_type_samples = jax.random.uniform(jax.random.PRNGKey(0), shape = (num_samples, num_consumers, num_commods), minval = type_endow_range[0], maxval = type_endow_range[1])
        state_samples = jax.vmap(lambda o, e, t: State(o, SpotMarket(e, {"valuation": t})), in_axes = 0)(world_states_samples, consumer_endow_samples, consumer_type_samples)
        # state_samples = [State(o, SpotMarket(e, t)) for (o, e, t) in zip(world_states_samples, consumer_endow_samples, consumer_type_samples)]
        return state_samples


    @staticmethod
    @functools.partial(jax.jit, static_argnames=[ "policy_profile"])
    def seller_first_order_state_violation(economy: MarkovEconomy, state: State, policy_profile: Callable[[State], Tuple]):
        price_space_sum = economy.commod_price_space_sum
        commod_price, asset_price, consumption, portfolio = policy_profile(state)
        excess_commod_demand, excess_asset_demand = EconomySimulator.get_state_excess_demand(state, consumption, portfolio)
        asset_price_feasibility = jnp.sum(asset_price) - price_space_sum
        return jnp.mean(jnp.square(excess_commod_demand)) + jnp.mean(jnp.square(excess_asset_demand))

    @staticmethod
    @functools.partial(jax.jit, static_argnames=[ "policy_profile", "num_samples", "num_episodes"])
    def consumers_first_order_state_violation(economy: MarkovEconomy, state: State, policy_profile: Callable[[State], Tuple], num_samples: int, num_episodes: int, solver_params = {}):
        commod_price, asset_price, consumption, portfolio = policy_profile(state)
        consendow = state.spot_market.consumer_endow
        num_consumers = economy.num_consumers
        discount = economy.discount


        util = lambda x: economy.util(x, state.spot_market.consumer_type)
        util_value = util(consumption)
        grad_util_jac = jax.jacfwd(util)(consumption)
        grad_util = jnp.einsum("iij->ij", grad_util_jac)

        consumer_future_value = lambda x, y: EconomySimulator.estimate_expected_future_value(economy, state, commod_price, asset_price, x, y, policy_profile, num_samples, num_episodes)[:num_consumers]
        # jac_consumption_state_value = jax.jacfwd(consumer_future_value, argnums = 0)(consumption, portfolio)
        jac_portfolio_state_value = jax.jacfwd(consumer_future_value, argnums = 1)(consumption, portfolio)

        # grad_consumption_state_value = jnp.einsum("iij->ij", jac_consumption_state_value)
        grad_portfolio_state_value = jnp.einsum("iij->ij", jac_portfolio_state_value)

        # bang = util_value + discount*jnp.einsum("ij,ij->i", grad_consumption_state_value, consumption) + discount*jnp.einsum("ij,ij->i", grad_portfolio_state_value, portfolio)
        # bang = util_value + discount*jnp.einsum("ij,ij->i", grad_consumption_state_value, consumption)
        bang = util_value
        # bang_per_buck = bang/(consendow @ commod_price)
        bang_per_buck = bang/(consendow @ commod_price - portfolio @ asset_price)

        # consumption_first_order_violation = grad_util + discount*grad_consumption_state_value - jnp.einsum("i,j->ij", bang_per_buck, commod_price)
        consumption_first_order_violation = grad_util - jnp.einsum("i,j->ij", bang_per_buck, commod_price)
        portfolio_first_order_violation =  discount*grad_portfolio_state_value - jnp.einsum("i,j->ij", bang_per_buck, asset_price)


        consumption_norm = jnp.mean(jnp.square(consumption_first_order_violation))
        portfolio_norm = jnp.mean(jnp.square(portfolio_first_order_violation))

        return consumption_norm + portfolio_norm

    @staticmethod
    @functools.partial(jax.jit, static_argnames=[ "policy_profile", "num_samples", "num_episodes"])
    def first_order_state_violation(economy: MarkovEconomy, state: State, policy_profile: Callable[[State], Tuple], num_samples: int, num_episodes: int, solver_params = {}):
        seller_violation = EconomySimulator.seller_first_order_state_violation(economy, state, policy_profile)
        consumers_violation = EconomySimulator.consumers_first_order_state_violation(economy, state, policy_profile, num_samples, num_episodes, solver_params)
        return (seller_violation + consumers_violation)/2.0

    @staticmethod
    @functools.partial(jax.jit, static_argnames=["policy_profile", "num_samples", "num_episodes"])
    def estimate_first_order_seller_violation(economy: MarkovEconomy, policy_profile: Callable[[State], Tuple], num_samples: int, num_episodes: int, solver_params = {}):
        state_samples = EconomySimulator.sample_state(economy, num_samples)
        violation_samples = jax.vmap(lambda s: EconomySimulator.seller_first_order_state_violation(economy, s, policy_profile) , in_axes = 0)(state_samples)
        violation_estimate = jnp.mean(violation_samples)
        return violation_estimate

    @staticmethod
    @functools.partial(jax.jit, static_argnames=["policy_profile", "num_samples", "num_episodes"])
    def estimate_first_order_consumer_violation(economy: MarkovEconomy,policy_profile: Callable[[State], Tuple], num_samples: int, num_episodes: int, solver_params = {}):
        state_samples = EconomySimulator.sample_state(economy, num_samples)
        violation_samples = jax.vmap(lambda s: EconomySimulator.consumers_first_order_state_violation(economy, s, policy_profile, num_samples, num_episodes, solver_params), in_axes = 0)(state_samples)
        violation_estimate = jnp.mean(violation_samples)
        return violation_estimate


    @staticmethod
    @functools.partial(jax.jit, static_argnames=["policy_profile", "num_samples", "num_episodes"])
    def estimate_first_order_violation(economy: MarkovEconomy,policy_profile: Callable[[State], Tuple], num_samples: int, num_episodes: int, solver_params = {}):
        state_samples = EconomySimulator.sample_state(economy, num_samples)
        violation_samples = jax.vmap(lambda s: EconomySimulator.first_order_state_violation(economy, s, policy_profile, num_samples, num_episodes, solver_params), in_axes = 0)(state_samples)
        violation_estimate = jnp.mean(violation_samples)
        return violation_estimate





network_size = 20

def _eqm_network(state: State, economy: MarkovEconomy):
    num_consumers = economy.num_consumers
    num_commods = economy.num_commods
    num_assets = economy.num_assets

    world_state = state.world_state
    asset_return = economy.asset_return[world_state]

    consumer_endow = state.spot_market.consumer_endow
    consumer_type = state.spot_market.consumer_type

    consumer_endow_min = economy.consumer_endow_range[0]
    consumer_endow_max = economy.consumer_endow_range[1]

    discount = economy.discount

    commod_price_space_sum = economy.commod_price_space_sum
    asset_price_range = economy.asset_price_range
    portfolio_range = economy.portfolio_range

    aggregate_supply = jnp.sum(consumer_endow, axis = 0)

    consumer_type_latent = LinearLayer(network_size, jax.nn.relu)(jnp.concatenate(list(consumer_type.values()), axis = -1))

    consumer_endow_latent = LinearLayer(network_size, jax.nn.relu)(consumer_endow)

    assets_latent_flattened = jnp.concatenate([asset_return, world_state, discount], axis = None)
    consumer_assets_latent = LinearLayer(num_consumers*network_size, jax.nn.relu)(assets_latent_flattened).reshape(num_consumers, network_size)

    consumer_latent = jnp.concatenate([consumer_type_latent, consumer_endow_latent, consumer_assets_latent], axis = -1)
    consumer_latent = LinearLayer(network_size, jax.nn.relu)(consumer_latent)

    price_latent = jnp.sum(consumer_latent, axis = 0)
    price_latent = LinearLayer(network_size, jax.nn.relu)(price_latent)

    commod_price_latent = LinearLayer(network_size, jax.nn.relu)(price_latent)
    commod_price = LinearLayer(num_commods, jax.nn.softmax)(commod_price_latent)*commod_price_space_sum

    asset_price_latent = LinearLayer(network_size, jax.nn.relu)(price_latent)
    asset_price = LinearLayer(num_assets, jax.nn.sigmoid)(asset_price_latent)*asset_price_range[1]

    commod_price_tiled = jnp.tile(commod_price, [num_consumers, 1])
    asset_price_tiled = jnp.tile(asset_price, [num_consumers, 1])

    market_state = jnp.concatenate([consumer_latent, commod_price_tiled, asset_price_tiled], axis = -1)

    market_latent = LinearLayer(network_size, jax.nn.relu)(market_state)
    portfolio_sell = LinearLayer(num_assets, jax.nn.sigmoid)(market_latent)*portfolio_range[1]

    budgets = consumer_endow @ commod_price + portfolio_sell @ asset_price

    coefs = LinearLayer(num_commods+num_assets, jax.nn.softmax)(market_latent)

    consumption_coefs = coefs[:, :num_commods]
    portfolio_coefs = coefs[:, num_commods:]
    price_tiled = jnp.tile(jnp.concatenate([commod_price, asset_price]), [num_consumers, 1])

    # Project to budget simplex
    budget_matrix = jnp.tile(budgets, [num_commods+num_assets, 1]).T
    budget_simplices = budget_matrix / price_tiled.clip(min = ZERO_CLIP_MIN)

    budget_simplices_commods = budget_simplices[:, :num_commods]
    budget_simplices_assets = budget_simplices[:, num_commods:]

    consumption = consumption_coefs * budget_simplices_commods
    portfolio_buy = portfolio_coefs * budget_simplices_assets
    portfolio = portfolio_buy - portfolio_sell

    aggregate_demand = jnp.sum(consumption, axis = 0)
    # consumption = consumption*jnp.clip(aggregate_supply/(ZERO_CLIP_MIN+aggregate_demand), a_min = ZERO_CLIP_MIN, a_max = 1)
    consumption = jnp.clip(consumption, a_min = ZERO_CLIP_MIN)

    return  (commod_price, asset_price, consumption, portfolio)

###########################################################################################################################################################
def _br_network(state: State, action_profile, economy: MarkovEconomy):
    # pi: s=(b, p) -> a' where a' is the budget share matrix
    commod_price, asset_price, consumption, portfolio = action_profile


    num_consumers = economy.num_consumers
    num_commods = economy.num_commods
    num_assets = economy.num_assets

    world_state = state.world_state
    asset_return = economy.asset_return[world_state]

    consumer_endow = state.spot_market.consumer_endow
    consumer_type = state.spot_market.consumer_type

    consumer_endow_min = economy.consumer_endow_range[0]
    consumer_endow_max = economy.consumer_endow_range[1]

    discount = economy.discount

    commod_price_space_sum = economy.commod_price_space_sum
    asset_price_range = economy.asset_price_range
    portfolio_range = economy.portfolio_range

    consumer_type_latent = LinearLayer(network_size, jax.nn.relu)(jnp.concatenate(list(consumer_type.values()), axis = -1))

    consumer_endow_latent = LinearLayer(network_size, jax.nn.relu)(consumer_endow)

    assets_latent_flattened = jnp.concatenate([asset_return, world_state, discount], axis = None)
    consumer_assets_latent = LinearLayer(num_consumers*network_size, jax.nn.relu)(assets_latent_flattened).reshape(num_consumers, network_size)


    consumer_latent = jnp.concatenate([consumer_type_latent, consumer_endow_latent, consumer_assets_latent], axis = -1)
    consumer_latent = LinearLayer(network_size, jax.nn.relu)(consumer_latent)


    commod_price_tiled = jnp.tile(commod_price, [num_consumers, 1])
    asset_price_tiled = jnp.tile(asset_price, [num_consumers, 1])
    market_state = jnp.concatenate([consumer_latent, commod_price_tiled, asset_price_tiled], axis = -1)

    market_latent = LinearLayer(network_size, jax.nn.relu)(market_state)
    portfolio_sell = LinearLayer(num_assets, jax.nn.sigmoid)(market_latent)*portfolio_range[1]

    budgets = consumer_endow @ commod_price + portfolio_sell @ asset_price

    coefs = LinearLayer(num_commods+num_assets, jax.nn.softmax)(market_latent)

    consumption_coefs = coefs[:, :num_commods]
    portfolio_coefs = coefs[:, num_commods:]
    price_tiled = jnp.tile(jnp.concatenate([commod_price, asset_price]), [num_consumers, 1])

    # Project to budget simplex
    budget_matrix = jnp.tile(budgets, [num_commods+num_assets, 1]).T
    budget_simplices = budget_matrix / price_tiled.clip(min = ZERO_CLIP_MIN)

    budget_simplices_commods = budget_simplices[:, :num_commods]
    budget_simplices_assets = budget_simplices[:, num_commods:]

    br_consumption = consumption_coefs * budget_simplices_commods
    portfolio_buy = portfolio_coefs * budget_simplices_assets
    br_portfolio = portfolio_buy - portfolio_sell


    ######### Best-Response Prices #########
    excess_commod_demand = jnp.sum(consumption, axis = 0) - jnp.sum(consumer_endow, axis = 0)
    excess_asset_demand = jnp.sum(portfolio, axis = 0)

    br_commod_price = jax.nn.softmax(excess_commod_demand)*commod_price_space_sum
    br_asset_price = jax.nn.sigmoid(excess_asset_demand)*asset_price_range[1]

    br_consumption = jnp.clip(br_consumption, a_min = ZERO_CLIP_MIN)

    return  (br_commod_price, br_asset_price, br_consumption, br_portfolio)
