from multi_agent_env import MultiAgentEnv
import chex
import jax
import jax.numpy as jnp
from typing import Tuple, Dict
from gymnax.environments.spaces import Discrete
from functools import partial
from flax import struct
from jax import lax


def max_of_n_gaussians(n, mean, sigma):
    """
    Utility funciton which gives the best possible performance
    in NoisyRewardLever game.
    Logic from:
    https://math.stackexchange.com/questions/473229/expected-value-of-maximum-and-minimum-of-n-normal-random-variables/510580#510580
    """
    E = \
    {
         1: 1,
         2: 0.56418,
         3: 0.8462843,
         4: 1.0293,
         5: 1.16296,
         6: 1.2672063,
         7: 1.3521783,
         8: 1.4236003,
         9: 1.4850131,
         10: 1.5387527,
    }

    return mean + sigma*E[n]

@struct.dataclass
class SelfPlayNRLGState:
    payoffs: chex.Array
    terminal: bool
    cur_player_idx: chex.Array
    non_coordinating_reward: chex.Array
    agent_steps: int
    eta1: chex.Array
    eta2: chex.Array
    obs1: chex.Array
    obs2: chex.Array
    reward: int
    total_reward: chex.Array

class SelfPlayNRLG(MultiAgentEnv):
    def __init__(self, r_mean=[5,5,5], sigma=3, sigma1=0, sigma2=0, 
        num_agents=2, n_actions=3, agents=None, obs_size=None, action_spaces=None, observation_space=None,
        non_coordinating_reward=-33, num_agent_steps=1024,
        ):
        super().__init__(num_agents)

        self.num_agents = num_agents
        self.agent_range = jnp.arange(num_agents)
        self.n_actions = n_actions
        self.num_agent_steps = num_agent_steps
        if agents is None:
            self.agents = [f"agent_{i}" for i in range(num_agents)]
        else:
            assert len(
                agents) == num_agents, f"Number of agents {len(agents)} does not match number of agents {num_agents}"
            self.agents = agents
        if obs_size is None:
            obs_dim = self.n_actions
            self.obs_size = obs_dim
            self.action_set = jnp.arange(self.n_actions)
            if action_spaces is None:
                self.action_spaces = {i: Discrete(self.n_actions) for i in self.agents}
            if observation_space is None:
                self.observation_spaces = {i: Discrete(self.obs_size) for i in self.agents}

        self.non_coordinating_reward = non_coordinating_reward
        self.payoffs = jnp.zeros((1, self.n_actions))
        self.r_mean = jnp.array(r_mean)
        self.sigma = sigma
        self.sigma1 = sigma1
        self.sigma2 = sigma2
        best_reward_possible = max_of_n_gaussians(3,
                                              self.r_mean[0],
                                              self.sigma)*1
        self.best_reward_possible = best_reward_possible

    def reset(self, key: chex.PRNGKey) -> Tuple[Dict, SelfPlayNRLGState]:
        key, subkey1, subkey2, subkey3 = jax.random.split(key, num=4)
        terminal = False
        payoffs = jax.random.normal(subkey1, (1, self.n_actions)) * self.sigma + self.r_mean
        eta1 = jax.random.normal(subkey2, (1, self.n_actions)) * self.sigma1
        eta2 = jax.random.normal(subkey3, (1, self.n_actions)) * self.sigma2
        obs1 = payoffs + eta1
        obs2 = payoffs + eta2
        agent_steps = 0
        cur_player_idx = jnp.zeros(self.num_agents).at[0].set(1)
        non_coordinating_reward = jnp.zeros((1, 1)).at[0].set(self.non_coordinating_reward)

        reward = self.non_coordinating_reward
        total_reward = jax.numpy.zeros((1, 1))

        state = SelfPlayNRLGState(payoffs=payoffs, terminal=terminal, cur_player_idx=cur_player_idx,
            non_coordinating_reward=non_coordinating_reward, agent_steps=agent_steps, eta1=eta1,
            eta2=eta2, obs1=obs1, obs2=obs2, reward=reward, total_reward=total_reward)
        return self.get_obs(state), state

    def get_pos_moves(self, state: SelfPlayNRLGState) -> chex.Array:
        @partial(jax.vmap, in_axes=[0, None])
        def _legal_moves(aidx: int, state: SelfPlayNRLGState) -> chex.Array:
            moves = jnp.zeros(self.n_actions)
            return moves
        pos_moves = _legal_moves(self.agent_range, state)

        return {a: pos_moves[i] for i, a in enumerate(self.agents)}

    @partial(jax.jit, static_argnums=[0])
    def get_obs(self, state: SelfPlayNRLGState) -> Dict:
        """Get all the agents' observations(both agent a and b)"""
        @partial(jax.vmap, in_axes=[0, None])
        def _observation(agent_id: int, state: SelfPlayNRLGState) -> chex.Array:
            curr_obs = jax.lax.select(agent_id==0, state.obs1, state.obs2)
            return curr_obs
        obs = _observation(self.agent_range, state)
        return {a: obs[i] for i, a in enumerate(self.agents)}

    @partial(jax.jit, static_argnums=[0])
    def step_env(self, key: chex.PRNGKey, state: SelfPlayNRLGState, actions: Dict,
                 ) -> Tuple[chex.Array, SelfPlayNRLGState, Dict, Dict, Dict]:
        actions = jnp.array([actions[i] for i in self.agents])
        action1, action2 = actions[0], actions[1]
        actions1 = jnp.asarray(action1, dtype=jnp.int32)
        actions2 = jnp.asarray(action2, dtype=jnp.int32)
        coor = actions1 == actions2
        aidx = jnp.nonzero(state.cur_player_idx, size=1)[0][0]
        state, reward = self.step_agent(key, state, aidx, actions, coor)


        max_lever_w_noise1 = jnp.argmax(state.obs1[0,...])
        max_lever_w_noise2 = jnp.argmax(state.obs2[0,...])
        min_lever_w_noise1 = jnp.argmin(state.obs1[0,...])
        min_lever_w_noise2 = jnp.argmin(state.obs2[0,...])

        otherplayinfo = {
                    "agent_1_noisy_max": action1==max_lever_w_noise1,
                    "agent_1_noisy_mid": (action1!=max_lever_w_noise1) & (action1!=min_lever_w_noise1),
                    "agent_1_noisy_min": action1==min_lever_w_noise1,
                    "agent_2_noisy_max": action2==max_lever_w_noise2,
                    "agent_2_noisy_mid": (action2!=max_lever_w_noise2) & (action2!=min_lever_w_noise2),
                    "agent_2_noisy_min": action2==min_lever_w_noise2,
                }

        max_reward = jnp.max(state.payoffs[0, ...])
        max_lever = jnp.argmax(state.payoffs[0, ...])
        min_lever = jnp.argmin(state.payoffs[0, ...])

        done = self.terminal(state) 
        dones = {agent: done for agent in self.agents}
        dones["__all__"] = done

        rewards = {agent: reward for agent in self.agents}
        rewards["__all__"] = reward
        regret = (self.best_reward_possible - reward) / self.best_reward_possible
        info_agent_0 = {
            "reward": reward,
            "reward_delta": (max_reward - reward),
            "max_reward": max_reward,
            "max_r": jnp.max(state.payoffs),
            "true_action1": action1,
            "true_action2": action2,
            "agent_1_max": (action1 == max_lever),
            "agent_2_max": (action2 == max_lever),
            "agent_1_min": (action1 == min_lever),
            "agent_2_min": (action2 == min_lever),
            "agent_1_mid": (action1 != max_lever) & (action1 != min_lever),
            "agent_2_mid": (action2 != max_lever) & (action2 != min_lever),
            "both_max": (action1 == max_lever) & (action2 == max_lever),
            "both_mid": ~((action1 == max_lever) | (action1 == min_lever) | (action2 == max_lever) | (action2 == min_lever)),
            "both_min": (action1 == min_lever) & (action2 == min_lever),
            "non_coord": action1 != action2,
            "regret": regret,
            "total_reward": state.total_reward
        }

        info_agent_1 = {
            "reward_delta": (max_reward - reward),
            "true_action1": action1,
            "true_action2": action2,
            "agent_1_max": (action1 == max_lever),
            "agent_2_max": (action2 == max_lever),
            "agent_1_min": (action1 == min_lever),
            "agent_2_min": (action2 == min_lever),
            "agent_1_mid": (action1 != max_lever) & (action1 != min_lever),
            "agent_2_mid": (action2 != max_lever) & (action2 != min_lever),
            "both_max": (action1 == max_lever) & (action2 == max_lever),
            "both_mid": ~((action1 == max_lever) | (action1 == min_lever) | (action2 == max_lever) | (action2 == min_lever)),
            "both_min": (action1 == min_lever) & (action2 == min_lever),
            "non_coord": action1 != action2,
            "regret": regret
        }

        info = {
            "agent_0": info_agent_0,
            "agent_1": info_agent_1,
            "other_play_info": otherplayinfo
        }

        return (
            lax.stop_gradient(self.get_obs(state)),
            lax.stop_gradient(state),
            rewards,
            dones,
            info
        )

    def step_agent(self, key: chex.PRNGKey, state: SelfPlayNRLGState, aidx: int, actions: chex.Array, coor: bool
                ) -> Tuple[SelfPlayNRLGState, int]:
        cur_player_idx = jnp.zeros(self.num_agents).at[aidx].set(0)
        aidx = (aidx + 1) % self.num_agents
        cur_player_idx = jnp.zeros(self.num_agents).at[aidx].set(1)
        terminal = jax.lax.select(state.agent_steps + 2 ==self.num_agent_steps, True, False)
        actions1 = actions[0]
        reward = jax.lax.select(coor[0], state.payoffs[:, actions1], state.non_coordinating_reward)
        total_rew = state.total_reward + reward
        steps = state.agent_steps + 1
        key, subkey1, subkey2, subkey3 = jax.random.split(key, num=4)
        payoffs = jax.random.normal(subkey1, (1, self.n_actions)) * self.sigma + self.r_mean
        eta1 = jax.random.normal(subkey2, (1, self.n_actions)) * self.sigma1
        eta2 = jax.random.normal(subkey3, (1, self.n_actions)) * self.sigma2
        obs1 = payoffs + eta1
        obs2 = payoffs + eta2
        return state.replace(terminal=terminal, payoffs=state.payoffs, cur_player_idx=cur_player_idx, agent_steps=steps, eta1=eta1, eta2=eta2, obs1=obs1, obs2=obs2, total_reward=total_rew), reward

    def terminal(self, state: SelfPlayNRLGState) -> bool:
        return state.terminal

    @property
    def name(self) -> str:
        """Environment Name """
        return "Self Play Noisy Lever One Shot Game"

    @property
    def num_actions(self) -> int:
        """Number of actions possible in environment."""
        return self.n_actions

    def observation_space(self, agent: str):
        return self.observation_spaces[agent]

    def action_space(self, agent: str):
        return self.action_spaces[agent]

    def deepcopy(self, x):
        return jax.tree_util.tree_map(lambda y: y, x)
