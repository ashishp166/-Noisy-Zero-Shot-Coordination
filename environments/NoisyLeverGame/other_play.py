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
class OtherPlayNRLGState:
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
    perm1: chex.Array
    perm2: chex.Array
    perm_action1: int
    perm_action2: int
    sigma1: float
    sigma2: float
    noise_1: float
    noise_2: float

class OtherPlayNZSC(MultiAgentEnv):
    def __init__(self, r_mean=[5,5,5], sigma=3, sigma1=0, sigma2=0, 
        num_agents=2, n_actions=3, agents=None, obs_size=None, action_spaces=None, observation_space=None,
        non_coordinating_reward=-33, num_agent_steps=1024,
        include_prev_reward_in_obs=False, include_prev_acts_in_obs=True, include_r_mean_noise_sigma=True, 
        include_agent_noise_sigmas=False, disable_other_play=True, dont_resample_permutation=True, 
        dont_resample_obs_noise=False, use_reward_delta=False, override_obs_with_zeros=True, sigma_rand=False,
        sigma1_idx=0, sigma2_idx=0, sigma_values=[0, 2, 5], noise_level=0
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

        self.non_coordinating_reward = non_coordinating_reward
        self.payoffs = jnp.zeros((1, self.n_actions))
        self.r_mean = jnp.array(r_mean)
        self.sigma = sigma
        self.sigma1 = sigma1
        self.sigma2 = sigma2
        self.noise_level = noise_level
        best_reward_possible = max_of_n_gaussians(3,
                                              self.r_mean[0],
                                              self.sigma)*self.num_agent_steps
        self.best_reward_possible = best_reward_possible


        if obs_size is None:
            obs_dim = self.n_actions # index 0-2
            obs_dim += 1 # index 3: include_prev_reward_in_obs
            obs_dim += 1 # index 4: include_prev_acts_in_obs
            obs_dim += n_actions+1 # index 5-8: include_r_mean_noise_sigma
            obs_dim += 2 # index 9 - 10: include_agent_noise_sigmas
            self.obs_size = obs_dim
            self.action_set = jnp.arange(self.n_actions)
            if action_spaces is None:
                self.action_spaces = {i: Discrete(self.n_actions) for i in self.agents}
            if observation_space is None:
                self.observation_spaces = {i: Discrete(self.obs_size) for i in self.agents}
        self.include_prev_reward_in_obs = include_prev_reward_in_obs # Index 3
        self.include_prev_acts_in_obs = include_prev_acts_in_obs # Index 4: observation space includes prev action
        self.include_r_mean_noise_sigma = include_r_mean_noise_sigma # Index 5-8
        self.include_agent_noise_sigmas = include_agent_noise_sigmas # Index 9-10
        
        self.disable_other_play = disable_other_play
        self.perm1 = jnp.arange(self.n_actions)
        self.perm2 = jnp.arange(self.n_actions)
        self.resample_permutation = not dont_resample_permutation
        self.resample_obs_noise = not dont_resample_obs_noise
        self.use_reward_delta = use_reward_delta
        self.override_obs_with_zeros = override_obs_with_zeros
        self.sigma_values = jnp.array(sigma_values)
        self.sigma1_idx = sigma1_idx
        self.sigma2_idx = sigma2_idx
        self.sigma_rand = sigma_rand
    
    def _unpermute_actions(self, state, actions1, actions2):
        perm1 = state.perm1
        perm2 = state.perm2
        return jnp.take(perm1, actions1), jnp.take(perm2, actions2)


    def reset(self, key: chex.PRNGKey) -> Tuple[Dict, OtherPlayNRLGState]:
        obs1 = jnp.zeros((1, self.obs_size))
        obs2 = jnp.zeros((1, self.obs_size))
        key, subkey1, subkey2, subkey3, subkey4, subkey5, subkey6, subkey7 = jax.random.split(key, num=8)
        terminal = False
        payoffs = jax.random.normal(subkey1, (1, self.n_actions)) * self.sigma + self.r_mean
        sigma1_random_index = jax.lax.select(self.sigma_rand, jax.random.randint(subkey4, (), minval=0, maxval=len(self.sigma_values)), self.sigma1_idx)
        sigma2_random_index = jax.lax.select(self.sigma_rand, jax.random.randint(subkey5, (), minval=0, maxval=len(self.sigma_values)), self.sigma2_idx)
        sigma1 = self.sigma_values[sigma1_random_index]
        sigma2 = self.sigma_values[sigma2_random_index]
        eta1 = jax.random.normal(subkey2, (1, self.n_actions)) * sigma1
        eta2 = jax.random.normal(subkey3, (1, self.n_actions)) * sigma2
        obs1_val = payoffs + eta1
        obs2_val = payoffs + eta2

        obs1 = obs1.at[:, jnp.arange(0, 3)].set(obs1_val)
        obs2 = obs2.at[:, jnp.arange(0, 3)].set(obs2_val)


        key, agent1key, agent2key = jax.random.split(key, 3)
        lever_perm1 = jax.lax.select(self.disable_other_play, jnp.arange(3), jax.random.permutation(agent1key, 3))
        lever_perm2 = jax.lax.select(self.disable_other_play, jnp.arange(3), jax.random.permutation(agent2key, 3))

        perm_obs1 = jnp.take(obs1_val, lever_perm1, axis=-1)
        perm_obs2 = jnp.take(obs2_val, lever_perm2, axis=-1)
        
        obs1 = obs1.at[:, jnp.arange(0, 3)].set(perm_obs1)
        obs2 = obs2.at[:, jnp.arange(0, 3)].set(perm_obs2)

        agent_steps = 1
        cur_player_idx = jnp.zeros(self.num_agents).at[0].set(1)
        non_coordinating_reward = jnp.zeros((1, 1)).at[0].set(self.non_coordinating_reward)

        reward = self.non_coordinating_reward
        total_reward = jax.numpy.zeros((1, 1))

        reward = jax.lax.select(self.use_reward_delta, self.r_mean[0] - self.non_coordinating_reward, self.non_coordinating_reward)
        obs1 = jax.lax.select(self.include_prev_reward_in_obs, obs1.at[:, 3:4].set(reward), obs1)
        obs2 = jax.lax.select(self.include_prev_reward_in_obs, obs2.at[:, 3:4].set(reward), obs2)


        a_action_b_pov, b_action_a_pov = -1, -1
        a_action_b_pov = jnp.array(a_action_b_pov).reshape(1, 1) 
        b_action_a_pov = jnp.array(b_action_a_pov).reshape(1, 1)
        obs1 = jax.lax.select(self.include_prev_acts_in_obs, obs1.at[:, 4:5].set(a_action_b_pov), obs1)
        obs2 = jax.lax.select(self.include_prev_acts_in_obs, obs2.at[:, 4:5].set(b_action_a_pov), obs2)

        r_mean_arr = jnp.array(self.r_mean).reshape(1, -1)
        r_mean_and_noise_sigma = jnp.concatenate((r_mean_arr, jnp.array(self.sigma).reshape(1, 1)), axis=-1)
        obs1 = jax.lax.select(self.include_r_mean_noise_sigma, obs1.at[:, 5:9].set(r_mean_and_noise_sigma), obs1)
        obs2 = jax.lax.select(self.include_r_mean_noise_sigma, obs2.at[:, 5:9].set(r_mean_and_noise_sigma), obs2)

        agent_1_sigma = jnp.array(sigma1).reshape(1, 1) 
        agent_2_sigma = jnp.array(sigma2).reshape(1, 1) 
        noise_1 = jax.random.normal(subkey6, (1, 1)) * self.noise_level
        noise_2 = jax.random.normal(subkey7, (1, 1)) * self.noise_level
        obs1 = jax.lax.select(self.include_agent_noise_sigmas, obs1.at[:, 9:10].set(agent_1_sigma), obs1)
        obs1 = jax.lax.select(self.include_agent_noise_sigmas, obs1.at[:, 10:11].set(agent_2_sigma + noise_2), obs1)
        obs2 = jax.lax.select(self.include_agent_noise_sigmas, obs2.at[:, 9:10].set(agent_2_sigma), obs2)
        obs2 = jax.lax.select(self.include_agent_noise_sigmas, obs2.at[:, 10:11].set(agent_1_sigma + noise_1), obs2)

        perm_action1 = -1
        perm_action2 = -1

        state = OtherPlayNRLGState(perm_action1=perm_action1, perm_action2=perm_action2, payoffs=payoffs, terminal=terminal, cur_player_idx=cur_player_idx,
            non_coordinating_reward=non_coordinating_reward, agent_steps=agent_steps, eta1=eta1,
            eta2=eta2, obs1=obs1, obs2=obs2, perm1=lever_perm1, perm2=lever_perm2, reward=reward, total_reward=total_reward, sigma1=sigma1, sigma2=sigma2,
            noise_1=noise_1, noise_2=noise_2)
        return self.get_obs(state), state

    def get_pos_moves(self, state: OtherPlayNRLGState) -> chex.Array:
        @partial(jax.vmap, in_axes=[0, None])
        def _legal_moves(aidx: int, state: OtherPlayNRLGState) -> chex.Array:
            moves = jnp.ones(self.n_actions)
            return moves
        pos_moves = _legal_moves(self.agent_range, state)

        return {a: pos_moves[i] for i, a in enumerate(self.agents)}

    @partial(jax.jit, static_argnums=[0])
    def get_obs(self, state: OtherPlayNRLGState) -> Dict:
        """Get all the agents' observations(both agent a and b)"""
        @partial(jax.vmap, in_axes=[0, None])
        def _observation(agent_id: int, state: OtherPlayNRLGState) -> chex.Array:
            curr_obs = jax.lax.select(agent_id==0, state.obs1, state.obs2)
            return curr_obs
        obs = _observation(self.agent_range, state)
        return {a: obs[i] for i, a in enumerate(self.agents)}

    @partial(jax.jit, static_argnums=[0])
    def step_env(self, key: chex.PRNGKey, state: OtherPlayNRLGState, actions: Dict,
                 ) -> Tuple[chex.Array, OtherPlayNRLGState, Dict, Dict, Dict]:
        actions = jnp.array([actions[i] for i in self.agents])
        action1, action2 = actions[0], actions[1]
        actions1 = jnp.asarray(action1, dtype=jnp.int32)
        actions2 = jnp.asarray(action2, dtype=jnp.int32)
        coor = actions1 == actions2
        aidx = jnp.nonzero(state.cur_player_idx, size=1)[0][0]
        state, reward = self.step_agent(key, state, aidx, actions, coor)

        action1 = state.perm_action1
        action2 = state.perm_action2
        actions1 = jnp.asarray(action1, dtype=jnp.int32)
        actions2 = jnp.asarray(action2, dtype=jnp.int32)

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
        regret = (self.best_reward_possible - state.total_reward) / self.best_reward_possible
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
            "total_reward": state.total_reward,
            "best_reward_possible": self.best_reward_possible,
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

    def step_agent(self, key: chex.PRNGKey, state: OtherPlayNRLGState, aidx: int, actions: chex.Array, coor: bool
                ) -> Tuple[OtherPlayNRLGState, int]:
        cur_player_idx = jnp.zeros(self.num_agents).at[aidx].set(0)
        aidx = (aidx + 1) % self.num_agents
        cur_player_idx = jnp.zeros(self.num_agents).at[aidx].set(1)
        terminal = jax.lax.select(state.agent_steps >= self.num_agent_steps, True, False)

        actions1, actions2 = actions[0], actions[1]
        actions1, actions2 = jnp.take(state.perm1, actions1), jnp.take(state.perm2, actions2)
        
        a_action_b_pov, b_action_a_pov = jnp.where(state.perm2 == actions1, size=1)[0][0], jnp.where(state.perm1 == actions2, size=1)[0][0]

        reward = jax.lax.select(actions1[0]==actions2[0], state.payoffs[:, actions1], state.non_coordinating_reward)
        total_rew = state.total_reward + reward


        steps = state.agent_steps + 1
        key, subkey1, subkey2, subkey3 = jax.random.split(key, num=4)
        payoffs = jax.random.normal(subkey1, (1, self.n_actions)) * self.sigma + self.r_mean

        eta1 = jax.random.normal(subkey2, (1, self.n_actions)) * state.sigma1
        eta2 = jax.random.normal(subkey3, (1, self.n_actions)) * state.sigma2

        obs1 = jax.lax.select(self.override_obs_with_zeros, jnp.zeros((1, self.obs_size)), state.obs1)
        obs2 = jax.lax.select(self.override_obs_with_zeros, jnp.zeros((1, self.obs_size)), state.obs2)

        obs1_val = payoffs + eta1
        obs2_val = payoffs + eta2

        obs1 = jax.lax.select(self.resample_obs_noise, obs1.at[:, :3].set(obs1_val), obs1.at[:, :3].set(state.obs1[:, :3]))
        obs2 = jax.lax.select(self.resample_obs_noise, obs2.at[:, :3].set(obs2_val), obs2.at[:, :3].set(state.obs2[:, :3]))

                
        key, agent1key, agent2key = jax.random.split(key, 3)
        update_perm1 = jax.lax.select(self.disable_other_play, jnp.arange(3), jax.random.permutation(agent1key, 3))
        update_perm2 = jax.lax.select(self.disable_other_play, jnp.arange(3), jax.random.permutation(agent2key, 3))


        lever_perm1 = jax.lax.select(self.resample_permutation, update_perm1, state.perm1)
        lever_perm2 = jax.lax.select(self.resample_permutation, update_perm2, state.perm2)

        perm_obs1 = jnp.take(obs1_val, lever_perm1, axis=-1)
        perm_obs2 = jnp.take(obs2_val, lever_perm2, axis=-1)

        obs1 = jax.lax.select(self.resample_obs_noise, obs1.at[:, 0:3].set(perm_obs1), obs1)
        obs2 = jax.lax.select(self.resample_obs_noise, obs2.at[:, 0:3].set(perm_obs2), obs2)

        obs1 = jax.lax.select(self.include_prev_acts_in_obs, obs1.at[:, 4:5].set(b_action_a_pov), obs1)
        obs2 = jax.lax.select(self.include_prev_acts_in_obs, obs2.at[:, 4:5].set(a_action_b_pov), obs2)

        r_mean_arr = jnp.array(self.r_mean).reshape(1, -1)
        r_mean_and_noise_sigma = jnp.concatenate((r_mean_arr, jnp.array(self.sigma).reshape(1, 1)), axis=-1)
        obs1 = jax.lax.select(self.include_r_mean_noise_sigma, obs1.at[:, 5:9].set(r_mean_and_noise_sigma), obs1)
        obs2 = jax.lax.select(self.include_r_mean_noise_sigma, obs2.at[:, 5:9].set(r_mean_and_noise_sigma), obs2)

        agent_1_sigma = jnp.array(state.sigma1).reshape(1, 1) 
        agent_2_sigma = jnp.array(state.sigma2).reshape(1, 1) 
        obs1 = jax.lax.select(self.include_agent_noise_sigmas, obs1.at[:, 9:10].set(agent_1_sigma), obs1)
        obs1 = jax.lax.select(self.include_agent_noise_sigmas, obs1.at[:, 10:11].set(agent_2_sigma+state.noise_2), obs1)
        obs2 = jax.lax.select(self.include_agent_noise_sigmas, obs2.at[:, 9:10].set(agent_2_sigma), obs2)
        obs2 = jax.lax.select(self.include_agent_noise_sigmas, obs2.at[:, 10:11].set(agent_1_sigma+state.noise_1), obs2)

        return state.replace(perm_action1=actions1[0], perm_action2=actions2[0], terminal=terminal, payoffs=state.payoffs, cur_player_idx=cur_player_idx, agent_steps=steps, eta1=eta1, eta2=eta2, obs1=obs1, obs2=obs2, total_reward=total_rew, perm1=lever_perm1, perm2=lever_perm2), reward

    def terminal(self, state: OtherPlayNRLGState) -> bool:
        return state.terminal

    @property
    def name(self) -> str:
        """Environment Name """
        return "Other Play Noisy Lever Zero Shot Game"

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
