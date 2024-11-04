from multi_agent_env import MultiAgentEnv
import chex
import jax
import jax.numpy as jnp
from typing import Tuple, Dict
from gymnax.environments.spaces import Discrete
from functools import partial
from flax import struct
from jax import lax

ACTION_TO_VECTOR = jnp.array([(dx, dy) for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0), (0, 0)]])

@struct.dataclass
class GridState:
    """
    Dataclass to store the state of the GridEnv environment.
    
    Attributes:
        payoffs (chex.Array): Payoffs for the boxes.
        terminal_cond (bool): True if the episode is terminated.
        agent_b_pos_a_pov (chex.Array): Position(x, y) of agent B from the perspective of agent A.
        agent_a_pos_b_pov (chex.Array): Position(x, y) of agent A from the perspective of agent B.
        terminal_agent (chex.Array): Not used currently.
        full_obs1 (chex.Array): Full observation vector for agent 1(includes agent 1 partial observability grid).
        full_obs2 (chex.Array): Full observation vector for agent 2(includes agent 2 partial observability grid).
        obs_vec_1 (chex.Array): Observation vector for agent 1.
        obs_vec_2 (chex.Array): Observation vector for agent 2.
        agent_grid_1 (chex.Array): Grid view for agent 1.
        agent_grid_2 (chex.Array): Grid view for agent 2.
        agent_steps (int): Number of steps taken by the agents.
        cur_player_idx (chex.Array): Index of the current player.
        reward_map (chex.Array): Mapping of box positions(x, y) and their rewards.
        grid_1 (chex.Array): Global grid for agent 1 with their noisy reward
        grid_1 (chex.Array): Global grid for agent 2 with their noisy reward
        global_grid (chex.Array): Global grid for agent 2 with the global reward
        stationary_reward (chex.Array): Reward for not landing on a stionary reward.
        non_coordinating_reward (chex.Array): Reward for not coordinating with the other agent.
        agent_a_pos (chex.Array): Position(x, y) of agent A.
        agent_b_pos (chex.Array): Position(x, y) of agent B.
    """
    payoffs: chex.Array
    terminal_cond: bool
    agent_b_pos_a_pov: chex.Array
    agent_a_pos_b_pov: chex.Array
    terminal_agent: chex.Array
    full_obs1: chex.Array
    full_obs2: chex.Array
    obs_vec_1: chex.Array
    obs_vec_2: chex.Array
    agent_grid_1: chex.Array
    agent_grid_2: chex.Array
    agent_steps: int
    cur_player_idx: chex.Array
    reward_map: chex.Array
    grid_1: chex.Array
    grid_2: chex.Array
    stationary_reward: chex.Array
    non_coordinating_reward: chex.Array
    agent_a_pos: chex.Array
    agent_b_pos: chex.Array
    agent_1_reward_map: chex.Array
    agent_2_reward_map: chex.Array
    total_reward: float
    last_non_stationary_move: int
    agent_1_start_pos: chex.Array
    agent_2_start_pos: chex.Array
    agent_1_low_reward_seen: bool
    agent_1_mid_reward_seen: bool
    agent_1_high_reward_seen: bool
    agent_2_low_reward_seen: bool
    agent_2_mid_reward_seen: bool
    agent_2_high_reward_seen: bool
    agent_1_reward_box_seen: int
    agent_2_reward_box_seen: int
    agent_1_key: chex.Array
    agent_2_key: chex.Array
    both_agent_coord_before_stationary: int
    high_reward_idx: int
    mid_reward_idx: int
    low_reward_idx: int
    logging_reward: chex.Array
    agent_1_delay: int
    agent_2_delay: int

def max_of_n_gaussians(n, mean, sigma):
    """
    Utility funciton which gives the best possible performance
    in NoisyRewardLever game.
    Logic from:
    https://math.stackexchange.com/questions/473229/expected-value-of-maximum-and-minimum-of-n-normal-random-variables/510580#510580
    
    Returns:
        float: The best possible performance in the NoisyRewardLever game.
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


class GridEnvComplex(MultiAgentEnv):
    def __init__(self, num_agents=2, n_actions=5, agents=None, obs_size=None,
             non_coordinating_reward=1, stationary_reward=-1, num_agent_steps=1024,
            width=10, height=10, agent_1_start_pos=[0, 0], agent_2_start_pos=[0, 0], r_mean=[5, 5, 5],
            include_r_mean_noise_sigma=True, include_prev_acts_in_obs=True, include_noise_sigmas=True, include_other_agent_pos=True,
            action_spaces=None, observation_space=None, agent_view_size=3, sigma=2, sigma1=0, sigma2=0,
            agent_pos_other_play=False, include_agent_noise_sigmas=True, include_prev_reward_in_obs=False, override_obs_with_zeros=True,
            lever_other_play=False, lever_subregion_width=3, lever_subregion_height=3, include_reward_pos=False, include_agent_pos=False,
            max_rand_start_agent_x_pos=3, max_rand_start_agent_y_pos=3, include_time_step=False, agent_1_view_size=3, agent_2_view_size=3, 
            min_rand_start_agent_x_pos=0, min_rand_start_agent_y_pos=0, key_multiplier=1.5, time_horizon_arr=[8, 16, 24], rand_time_horizon=False,
            agent_1_rand_time=True):
            """
            Initialize the GridEnv environment with the specified parameters. (Refer to the class docstring)
            """
            super().__init__(num_agents)

            self.num_agents = num_agents
            self.agent_range = jnp.arange(num_agents)
            self.num_boxes = len(r_mean)
            self.r_mean = jnp.array(r_mean)
            self.n_actions = n_actions
            self.num_agent_steps = num_agent_steps
            self.time_horizon_arr=jnp.array(time_horizon_arr)
            self.max_time_horizon = jnp.max(self.time_horizon_arr)
            self.delay = self.time_horizon_arr / self.max_time_horizon
            self.rand_time_horizon = rand_time_horizon
            self.agent_1_rand_time = agent_1_rand_time

            assert(agent_view_size % 2 == 1)
            self.agent_view_size = agent_view_size
            self.lever_other_play = lever_other_play
            self.grid_half_size = (self.agent_view_size - 1) // 2
            self.key_multiplier = key_multiplier
            
            if agents is None:
                self.agents = [f"agent_{i}" for i in range(num_agents)]
            else:
                assert len(
                    agents) == num_agents, f"Number of agents {len(agents)} does not match number of agents {num_agents}"
                self.agents = agents

            if obs_size is None:
                obs_dim = self.num_boxes
                # if include_prev_acts_in_obs:
                obs_dim += 1
                # if include_r_mean_noise_sigma:
                obs_dim += self.r_mean.shape[0]+1 # 1 is for the sigma value
                # if include_noise_sigmas:
                obs_dim += 1
                # include_agent_pos:
                obs_dim += 2
                # if include_other_agent_pos:
                obs_dim += 2 # for x and y pos
                # if include_reward_pos:
                obs_dim += 3 * self.num_boxes
                obs_1_dim = obs_dim
                obs_2_dim = obs_dim
                obs_dim += self.agent_view_size*self.agent_view_size
                self.agent_1_view_size = agent_1_view_size
                self.agent_2_view_size = agent_2_view_size
                obs_1_dim += self.agent_1_view_size*self.agent_1_view_size
                obs_2_dim += self.agent_2_view_size*self.agent_2_view_size
                self.obs_size = obs_dim
                self.obs_1_size = obs_1_dim
                self.obs_2_size = obs_2_dim
                self.action_set = jnp.arange(self.n_actions)
                if action_spaces is None:
                    self.action_spaces = {i: Discrete(self.n_actions) for i in self.agents}
                if observation_space is None:
                    self.observation_spaces = {i: Discrete(self.obs_1_size if i == 'agent_0' else self.obs_2_size) for i in self.agents}
            self.width = width
            self.height = height

            self.agent_1_start_pos = agent_1_start_pos
            self.agent_2_start_pos = agent_2_start_pos
            self.non_coordinating_reward = non_coordinating_reward
            self.sigma = sigma
            self.sigma1 = sigma1
            self.sigma2 = sigma2
            self.stationary_reward = stationary_reward
            best_reward_possible = max_of_n_gaussians(3,
                                              self.r_mean[0],
                                              self.sigma)*(1)
            self.best_reward_possible = best_reward_possible
            self.payoffs = jnp.zeros((1, self.num_boxes))
            self.agent_pos_other_play = agent_pos_other_play
            self.lever_pos = jnp.array([[self.width - 1, self.width - 1, self.width - 1], [1, 2, 3]])
            self.include_r_mean_noise_sigma = include_r_mean_noise_sigma
            self.include_prev_acts_in_obs = include_prev_acts_in_obs
            self.include_other_agent_pos = include_other_agent_pos
            self.include_agent_noise_sigmas = include_agent_noise_sigmas
            self.include_prev_reward_in_obs = include_prev_reward_in_obs
            self.override_obs_with_zeros = override_obs_with_zeros
            self.lever_subregion_width = lever_subregion_width
            self.lever_subregion_height = lever_subregion_height
            self.include_reward_pos = include_reward_pos
            self.include_agent_pos = include_agent_pos
            self.max_rand_start_agent_x_pos = max_rand_start_agent_x_pos
            self.max_rand_start_agent_y_pos = max_rand_start_agent_y_pos
            self.min_rand_start_agent_x_pos = min_rand_start_agent_x_pos
            self.min_rand_start_agent_y_pos = min_rand_start_agent_y_pos
            self.include_time_step = include_time_step
    
    def _gen_grid(self, key: chex.PRNGKey, reward_1, reward_2, lever_x, lever_y):
        """
        On reset(), generate the grid with the boxes and agent positions.
        
        Args:
            key (chex.PRNGKey): Random number generator key.
            reward (chex.Array): Rewards for the boxes.
            lever_x (chex.Array): X-coordinates of the boxes.
            lever_y (chex.Array): Y-coordinates of the boxes.
            
        Returns:
            Tuple[chex.Array, Tuple[int, int]]: The generated grid and the initial agent position.
        """

        grid_1 = jnp.zeros((self.width, self.height))
        grid_2 = jnp.zeros((self.width, self.height))

        _key, leverpos_x_key, leverpos_y_key = jax.random.split(key, num=3)

        _key, leverpos_key = jax.random.split(key, num=2)

        num_indices = self.lever_subregion_width * self.lever_subregion_height
        indices = jnp.arange(num_indices)
        shuffled_indices = jax.random.permutation(leverpos_key, indices)[:3]

        shuffle_x_lever = shuffled_indices // self.lever_subregion_width
        shuffle_y_lever = shuffled_indices % self.lever_subregion_width

        x_lever = jax.lax.select(self.lever_other_play, self.width - 1 - shuffle_x_lever, lever_x[0])
        y_lever = jax.lax.select(self.lever_other_play, self.height - 1 - shuffle_y_lever, lever_y[0])

        for i in range(3):
            grid_1 = grid_1.at[x_lever[i], y_lever[i]].set(reward_1[0][i])
            grid_2 = grid_2.at[x_lever[i], y_lever[i]].set(reward_2[0][i])

        grid_1.at[0, self.height - 1].set(self.stationary_reward)
        grid_2.at[0, self.height - 1].set(self.stationary_reward)
        key, agent_1_pos_x, agent_1_pos_y, agent_2_pos_x, agent_2_pos_y = jax.random.split(_key, num=5)

        rand_1_x = jax.lax.select(self.agent_pos_other_play, (jax.random.randint(agent_1_pos_x, (1,), minval=self.min_rand_start_agent_x_pos, maxval=self.max_rand_start_agent_x_pos)), jnp.array([self.agent_1_start_pos[0]]))
        rand_1_y = jax.lax.select(self.agent_pos_other_play, (jax.random.randint(agent_1_pos_y, (1,), minval=self.min_rand_start_agent_y_pos, maxval=self.max_rand_start_agent_y_pos)), jnp.array([self.agent_1_start_pos[1]]))
        agent_1_pos = (rand_1_x, rand_1_y)

        rand_2_x = jax.lax.select(self.agent_pos_other_play, (jax.random.randint(agent_2_pos_x, (1,), minval=self.min_rand_start_agent_x_pos, maxval=self.max_rand_start_agent_x_pos)), jnp.array([self.agent_2_start_pos[0]]))
        rand_2_y = jax.lax.select(self.agent_pos_other_play, (jax.random.randint(agent_2_pos_y, (1,), minval=self.min_rand_start_agent_y_pos, maxval=self.max_rand_start_agent_y_pos)), jnp.array([self.agent_2_start_pos[1]]))
        agent_2_pos = (rand_2_x, rand_2_y)
        self.mission = "Coordinate with other agent to one of the boxes"
        return grid_1, agent_1_pos, grid_2, agent_2_pos, x_lever, y_lever
    
    def reset(self, key: chex.PRNGKey) -> Tuple[Dict, GridState]:
        """
        Reset the environment and return the initial observations and state.
        
        Args:
            key (chex.PRNGKey): Random number generator key.
            
        Returns:
            Tuple[Dict, GridState]: A dictionary of initial observations and the initial state.
        """
        terminal_agent = jnp.zeros(2, dtype=jnp.int32)
        cur_player_idx = jnp.zeros(self.num_agents).at[0].set(1)

        obs1 = jnp.zeros((1, self.obs_1_size))
        obs2 = jnp.zeros((1, self.obs_2_size))

        key, subkey1, subkey2, subkey3 = jax.random.split(key, num=4)
        terminal_cond = False
        payoffs = jax.random.normal(subkey1, (1, self.num_boxes)) * self.sigma + self.r_mean
        reward_indices = jnp.argsort(-payoffs)

        eta1 = jax.random.normal(subkey2, (1, self.num_boxes)) * self.sigma1
        eta2 = jax.random.normal(subkey3, (1, self.num_boxes)) * self.sigma2
    
        obs1_val = payoffs + eta1
        obs2_val = payoffs + eta2

        lever_pos_x = jnp.array(self.lever_pos[0]).reshape(1, self.num_boxes)
        lever_pos_y = jnp.array(self.lever_pos[1]).reshape(1, self.num_boxes)

        grid_1, agent_a_pos, grid_2, agent_b_pos, lever_pos_x, lever_pos_y  = self._gen_grid(key=key, reward_1=obs1_val, reward_2=obs2_val, lever_x=lever_pos_x, lever_y=lever_pos_y)

        reward_map = jnp.array([lever_pos_x, lever_pos_y, payoffs[0]])
        agent_1_reward_map = jnp.array([lever_pos_x, lever_pos_y, obs1_val[0]])
        agent_2_reward_map = jnp.array([lever_pos_x, lever_pos_y, obs2_val[0]])

        obs1 = jax.lax.select(self.include_reward_pos, obs1.at[:, self.obs_1_size - 9:self.obs_1_size].set(agent_1_reward_map.reshape(1, -1)), obs1)
        obs2 = jax.lax.select(self.include_reward_pos, obs2.at[:, self.obs_2_size - 9:self.obs_2_size].set(agent_2_reward_map.reshape(1, -1)), obs2)


        agent_b_pos = (agent_b_pos[0][0], agent_b_pos[1][0])
        agent_a_pos = (agent_a_pos[0][0], agent_a_pos[1][0])

        agent_steps = 0
        
        a_action_b_pov, b_action_a_pov = -1, -1
        a_action_b_pov = jnp.array(a_action_b_pov).reshape(1, 1) 
        b_action_a_pov = jnp.array(b_action_a_pov).reshape(1, 1)
        agent_steps_arr = jnp.array(agent_steps).reshape(1, 1)
        obs1 = jax.lax.select(self.include_time_step, obs1.at[:, 0:1].set(agent_steps_arr), obs1)
        obs2 = jax.lax.select(self.include_time_step, obs2.at[:, 0:1].set(agent_steps_arr), obs2)

        r_mean_arr = jnp.array(self.r_mean).reshape(1, -1)
        r_mean_and_noise_sigma = jnp.concatenate((r_mean_arr, jnp.array(0).reshape(1, 1)), axis=-1)
        obs1 = jax.lax.select(self.include_r_mean_noise_sigma, obs1.at[:, 1:5].set(r_mean_and_noise_sigma), obs1)
        obs2 = jax.lax.select(self.include_r_mean_noise_sigma, obs2.at[:, 1:5].set(r_mean_and_noise_sigma), obs2)

        agent_1_sigma = jnp.array(self.sigma1).reshape(1, 1) 
        agent_2_sigma = jnp.array(self.sigma2).reshape(1, 1) 
        obs1 = jax.lax.select(self.include_agent_noise_sigmas, obs1.at[:, 5:6].set(agent_1_sigma), obs1)
        obs2 = jax.lax.select(self.include_agent_noise_sigmas, obs2.at[:, 5:6].set(agent_2_sigma), obs2)

        agent_b_pos_a_pov = jnp.array(self._get_other_agent_pos(agent_a_pos, agent_b_pos)).reshape(1, -1)
        agent_a_pos_b_pov = jnp.array(self._get_other_agent_pos(agent_b_pos, agent_a_pos)).reshape(1, -1)
        obs1 = jax.lax.select(self.include_other_agent_pos, obs1.at[:, 6:8].set(agent_b_pos_a_pov), obs1)
        obs2 = jax.lax.select(self.include_other_agent_pos, obs2.at[:, 6:8].set(agent_a_pos_b_pov), obs2)

        obs_reward_val_1 = jnp.array(obs1_val[0]).reshape(1, -1)
        obs_reward_val_2 = jnp.array(obs2_val[0]).reshape(1, -1)
        obs1 = obs1.at[:, 8:11].set(obs_reward_val_1)
        obs2 = obs2.at[:, 8:11].set(obs_reward_val_2)

        obs1 = jax.lax.select(self.include_agent_pos, obs1.at[:, 11:13].set(jnp.array(agent_a_pos).reshape(1, -1)), obs1)
        obs2 = jax.lax.select(self.include_agent_pos, obs2.at[:, 11:13].set(jnp.array(agent_b_pos).reshape(1, -1)), obs2)

        obs_vec_1 = obs1
        obs_vec_2 = obs2

        agent_grid = self.cross_play_grid_view(agent_a_pos, agent_b_pos, grid_1, grid_2)
        agent_grid_1, agent_grid_2 = jnp.ravel(agent_grid['agent_0']).reshape(1, -1), jnp.ravel(agent_grid['agent_1']).reshape(1, -1)

        full_obs1 = obs_vec_1.at[:, 13:13 + self.agent_1_view_size*self.agent_1_view_size].set(agent_grid_1)
        full_obs2 = obs_vec_2.at[:, 13:13 + self.agent_2_view_size*self.agent_2_view_size].set(agent_grid_2)

        non_coordinating_reward = jnp.zeros((1, 1)).at[0].set(self.non_coordinating_reward)
        stationary_reward = jnp.zeros((1, 1)).at[0].set(self.stationary_reward)
        key, agent_1_num_steps_key, agent_2_num_steps_key = jax.random.split(key, num=3)
        agent_1_steps = jax.random.choice(agent_1_num_steps_key, self.time_horizon_arr) 
        agent_2_steps = jax.random.choice(agent_2_num_steps_key, self.time_horizon_arr) 

        agent_1_delay = jax.lax.select(self.rand_time_horizon, (self.max_time_horizon / agent_1_steps).astype(int), 1)
        agent_2_delay = jax.lax.select(self.rand_time_horizon, (self.max_time_horizon / agent_2_steps).astype(int), 1)

        agent_a_pos = jnp.array(agent_a_pos).reshape(1, -1)
        agent_b_pos = jnp.array(agent_b_pos).reshape(1, -1)
        total_reward = 0.0
        logging_reward = non_coordinating_reward
        last_non_stationary_move = 0
        agent_1_reward_box_seen = 0
        agent_2_reward_box_seen = 0
        agent_1_low_reward_seen = False
        agent_1_mid_reward_seen = False
        agent_1_high_reward_seen = False
        agent_2_low_reward_seen = False
        agent_2_mid_reward_seen = False
        agent_2_high_reward_seen = False

        high_reward_idx = reward_indices[0][0]
        mid_reward_idx = reward_indices[0][1]
        low_reward_idx = reward_indices[0][2]

        agent_1_high_reward_box_seen = (reward_map[0][high_reward_idx] == agent_a_pos[0][0]) & (reward_map[1][high_reward_idx] == agent_a_pos[0][1])
        agent_1_mid_reward_box_seen = (reward_map[0][mid_reward_idx] == agent_a_pos[0][0]) & (reward_map[1][mid_reward_idx] == agent_a_pos[0][1])
        agent_1_low_reward_box_seen = (reward_map[0][low_reward_idx] == agent_a_pos[0][0]) & (reward_map[1][low_reward_idx] == agent_a_pos[0][1])
        agent_2_high_reward_box_seen = (reward_map[0][high_reward_idx] == agent_b_pos[0][0]) & (reward_map[1][high_reward_idx] == agent_b_pos[0][1])
        agent_2_mid_reward_box_seen = (reward_map[0][mid_reward_idx] == agent_b_pos[0][0]) & (reward_map[1][mid_reward_idx] == agent_b_pos[0][1])
        agent_2_low_reward_box_seen = (reward_map[0][low_reward_idx] == agent_b_pos[0][0]) & (reward_map[1][low_reward_idx] == agent_b_pos[0][1])
        agent_1_low_reward_seen = agent_1_low_reward_box_seen
        agent_1_mid_reward_seen = agent_1_mid_reward_box_seen
        agent_1_high_reward_seen = agent_1_high_reward_box_seen
        agent_2_low_reward_seen = agent_2_low_reward_box_seen
        agent_2_mid_reward_seen = agent_2_mid_reward_box_seen
        agent_2_high_reward_seen = agent_2_high_reward_box_seen
        
        agent_1_reward_box_seen = jnp.int32(agent_1_low_reward_seen) + jnp.int32(agent_1_mid_reward_seen) + jnp.int32(agent_1_high_reward_seen)
        agent_2_reward_box_seen = jnp.int32(agent_2_low_reward_seen) + jnp.int32(agent_2_mid_reward_seen) + jnp.int32(agent_2_high_reward_seen)

        both_agent_coord_before_stationary = 0
        agent_pos_coord = ((agent_b_pos[0][0] == agent_a_pos[0][0]) & (agent_b_pos[0][1] == agent_a_pos[0][1]))
        both_agent_coord_before_stationary = jax.lax.select(agent_pos_coord, both_agent_coord_before_stationary + 1, both_agent_coord_before_stationary)
        state = GridState(payoffs=payoffs, terminal_cond=terminal_cond, agent_b_pos_a_pov=agent_b_pos_a_pov, agent_a_pos_b_pov=agent_a_pos_b_pov,
                        terminal_agent=terminal_agent, agent_steps=agent_steps, full_obs1=full_obs1, full_obs2=full_obs2, reward_map=reward_map,
                        obs_vec_1=obs_vec_1, obs_vec_2=obs_vec_2, agent_grid_1=agent_grid_1, agent_grid_2=agent_grid_2, cur_player_idx=cur_player_idx,
                        grid_1=grid_1, grid_2=grid_2, non_coordinating_reward=non_coordinating_reward, stationary_reward=stationary_reward, agent_a_pos=agent_a_pos,
                        agent_b_pos=agent_b_pos, agent_1_reward_map=agent_1_reward_map, agent_2_reward_map=agent_2_reward_map, total_reward=total_reward, last_non_stationary_move=last_non_stationary_move,
                        agent_1_start_pos=agent_a_pos, agent_2_start_pos=agent_b_pos, agent_1_reward_box_seen=agent_1_reward_box_seen, agent_2_reward_box_seen=agent_2_reward_box_seen, 
                        agent_1_low_reward_seen=agent_1_low_reward_seen, agent_1_mid_reward_seen=agent_1_mid_reward_seen, agent_1_high_reward_seen=agent_1_high_reward_seen,
                        agent_2_low_reward_seen=agent_2_low_reward_seen, agent_2_mid_reward_seen=agent_2_mid_reward_seen, agent_2_high_reward_seen=agent_2_high_reward_seen,
                        both_agent_coord_before_stationary=both_agent_coord_before_stationary, high_reward_idx=high_reward_idx, mid_reward_idx=mid_reward_idx, low_reward_idx=low_reward_idx,
                        logging_reward=logging_reward, agent_1_key=0.0, agent_2_key=0.0, agent_1_delay=agent_1_delay, agent_2_delay=agent_2_delay)
        return self.get_obs(state), state
    
    @partial(jax.jit, static_argnums=[0])
    def get_pos_moves(self, state: GridState) -> chex.Array:
        """
        Get the legal moves for each agent based on their current position.
        
        Args:
            state (GridState): The current state of the environment.
            
        Returns:
            chex.Array: A dictionary mapping agent names to boolean array for legal moves out of 5 actions.
        """
        @partial(jax.vmap, in_axes=[0, None])
        def _legal_moves(aidx: int, state: GridState) -> chex.Array:
            agent_x = jax.lax.select(aidx==0, state.agent_a_pos[0][0], state.agent_b_pos[0][0])
            agent_y = jax.lax.select(aidx==0, state.agent_a_pos[0][1], state.agent_b_pos[0][1])
            actions = jnp.arange(self.n_actions)
            def grid_moves(i):
                dx, dy = ACTION_TO_VECTOR[i]
                nx, ny = agent_x + dx, agent_y + dy
                nx_ge_0 = nx >= 0
                nx_lt_width = nx < self.width
                ny_ge_0 = ny >= 0
                ny_lt_height = ny < self.height

                in_bound = jnp.logical_and(jnp.logical_and(nx_ge_0, nx_lt_width),
                                        jnp.logical_and(ny_ge_0, ny_lt_height))
                return in_bound
            moves = jax.vmap(grid_moves)(actions)

            agent_move = jax.lax.select(
                aidx == 0,
                state.agent_steps % state.agent_1_delay == 0,
                state.agent_steps % state.agent_2_delay == 0
            )
            forced_moves = jnp.where(agent_move, moves, jnp.array([0, 0, 0, 0, 1]))


            return forced_moves
        pos_moves = _legal_moves(self.agent_range, state)
        return {a: pos_moves[i] for i, a in enumerate(self.agents)}

    def get_agent_grid_view(self, agent_a_pos, agent_b_pos, grid_1, grid_2) -> chex.Array:
        """
        Get the grid view for each agent based on their position and the current grid.
        If goal box, then provide the agent's noisy reward
        Otherwise, if a valid position in the grid give value of 0
        Else if wall or outside the grid, provide value of -1
        
        Args:
            agent_a_pos (Tuple[int, int]): Position of agent A.
            agent_b_pos (Tuple[int, int]): Position of agent B.
            grid (chex.Array): The current grid.
            
        Returns:
            chex.Array: A dictionary mapping agent names to their grid view.
        """
        @partial(jax.vmap, in_axes=[0, None, None])
        def _get_agent_grid(aidx: int, agent_a_pos, agent_b_pos) -> chex.Array:
            agent_x = jnp.array(jax.lax.select(aidx==0, agent_a_pos[0], agent_b_pos[0]))
            agent_y = jnp.array(jax.lax.select(aidx==0, agent_a_pos[1], agent_b_pos[1]))
            grid = jnp.array(jax.lax.select(aidx==0, grid_1, grid_2))
            agent_grid = jnp.zeros((self.agent_view_size, self.agent_view_size))
            for i in range(self.agent_view_size * self.agent_view_size):
                x = agent_x - self.grid_half_size + (i % self.agent_view_size)
                y = agent_y - self.grid_half_size + (i // self.agent_view_size)
                in_bounds = jnp.all((0 <= x) & (x < self.width) & (0 <= y) & (y < self.height))
                updated_grid_val = jax.lax.select(in_bounds, grid[x,y], jnp.array([-1.0])[0])
                agent_grid = agent_grid.at[i // self.agent_view_size, i % self.agent_view_size].set(updated_grid_val)
            return agent_grid
        agent_grid = _get_agent_grid(self.agent_range, agent_a_pos, agent_b_pos)
        return {a: agent_grid[i] for i, a in enumerate(self.agents)}

    def cross_play_grid_view(self, agent_a_pos, agent_b_pos, grid_1, grid_2) -> chex.Array:
        """
        Get the grid view for each agent based on their position and the current grid.
        If goal box, then provide the agent's noisy reward
        Otherwise, if a valid position in the grid give value of 0
        Else if wall or outside the grid, provide value of -1
        
        Args:
            agent_a_pos (Tuple[int, int]): Position of agent A.
            agent_b_pos (Tuple[int, int]): Position of agent B.
            grid (chex.Array): The current grid.
            
        Returns:
            chex.Array: A dictionary mapping agent names to their grid view.
        """
        def get_agent_grid(agent_pos, view_size, grid):
            half_size = (view_size - 1) // 2
            agent_grid = jnp.zeros((view_size, view_size))
            for i in range(view_size * view_size):
                x = agent_pos[0] - half_size + (i % view_size)
                y = agent_pos[1] - half_size + (i // view_size)
                in_bounds = (0 <= x) & (x < self.width) & (0 <= y) & (y < self.height)
                updated_grid_val = jax.lax.cond(in_bounds, lambda: grid[x, y], lambda: jnp.array([-1.0])[0])
                agent_grid = agent_grid.at[i // view_size, i % view_size].set(updated_grid_val)
            return agent_grid
        
        agent_a_grid = get_agent_grid(agent_a_pos, self.agent_1_view_size, grid_1)
        agent_b_grid = get_agent_grid(agent_b_pos, self.agent_2_view_size, grid_2)
        
        return {self.agents[0]: agent_a_grid, self.agents[1]: agent_b_grid}

    @partial(jax.jit, static_argnums=[0])
    def get_obs(self, state: GridState) -> Dict:
        """Get all the agents' observations(both agent a and b)"""
        def _observation(agent_id):
            if agent_id == 0:
                return state.full_obs1
            else:
                return state.full_obs2

        obs = {a: _observation(agent_id) for agent_id, a in enumerate(self.agents)}
        return obs

    @partial(jax.jit, static_argnums=[0])
    def step_env(self, key: chex.PRNGKey, state: GridState, actions: Dict,
                 ) -> Tuple[chex.Array, GridState, Dict, Dict, Dict]:
        actions = jnp.array([actions[i] for i in self.agents])

        action1, action2 = actions[0], actions[1]
        actions1 = jnp.asarray(action1, dtype=jnp.int32)
        actions2 = jnp.asarray(action2, dtype=jnp.int32)
        coor = actions1 == actions2

        aidx = jnp.nonzero(state.cur_player_idx, size=1)[0][0]
        state, reward = self.step_agent(key, state, aidx, actions, coor)

        max_reward = jnp.max(state.payoffs[0, ...])

        done = self.terminal(state) 
        dones = {agent: done for agent in self.agents}
        dones["__all__"] = done

        rewards = {agent: reward for agent in self.agents}
        rewards["__all__"] = reward
        regret = (self.best_reward_possible - state.logging_reward) / self.best_reward_possible

        agent_pos_coord = ((state.agent_b_pos[0][0] == state.agent_a_pos[0][0]) & (state.agent_b_pos[0][1] == state.agent_a_pos[0][1]))

        high_reward_box_coord = agent_pos_coord & (state.reward_map[0][state.high_reward_idx] == state.agent_a_pos[0][0]) & (state.reward_map[1][state.high_reward_idx] == state.agent_a_pos[0][1])
        mid_reward_box_coord = agent_pos_coord & (state.reward_map[0][state.mid_reward_idx] == state.agent_a_pos[0][0]) & (state.reward_map[1][state.mid_reward_idx] == state.agent_a_pos[0][1])
        low_reward_box_coord = agent_pos_coord & (state.reward_map[0][state.low_reward_idx] == state.agent_a_pos[0][0]) & (state.reward_map[1][state.low_reward_idx] == state.agent_a_pos[0][1])
        info_agent_0 = {
            "reward": state.logging_reward,
            "reward_delta": (max_reward - state.logging_reward),
            "max_reward": max_reward,
            "max_r": jnp.max(state.payoffs),
            "true_action1": action1,
            "true_action2": action2,
            "agent2_pos_x": state.agent_b_pos[0][0],
            "agent2_pos_y": state.agent_b_pos[0][1],
            "agent1_pos_x": state.agent_a_pos[0][0],
            "agent1_pos_y": state.agent_a_pos[0][1],
            "agent_pos_coord": agent_pos_coord,
            "non_coord": action1 != action2,
            "regret": regret,
            "high_reward_box_coord": high_reward_box_coord,
            "mid_reward_box_coord": mid_reward_box_coord,
            "low_reward_box_coord": low_reward_box_coord,
            "high_reward_box_pos": state.reward_map[0][state.high_reward_idx] + state.reward_map[1][state.high_reward_idx],
            "mid_reward_box_pos": state.reward_map[0][state.mid_reward_idx] + state.reward_map[1][state.mid_reward_idx],
            "low_reward_box_pos": state.reward_map[0][state.low_reward_idx] + state.reward_map[1][state.low_reward_idx],
            "high_reward": state.reward_map[2][state.high_reward_idx],
            "mid_reward": state.reward_map[2][state.mid_reward_idx],
            "low_reward": state.reward_map[2][state.low_reward_idx],
            "total_reward": state.total_reward + 16,
            "last_non_stationary_move": state.last_non_stationary_move,
            "agent_1_reward_box_stepped_on": state.agent_1_reward_box_seen,
            "agent_2_reward_box_stepped_on": state.agent_2_reward_box_seen,
            "both_agent_coord_before_stationary": state.both_agent_coord_before_stationary,
            "agent_1_key": state.agent_1_key,
            "agent_2_key": state.agent_2_key,
            "agent_1_stationary_reward": (state.agent_a_pos[0][0]==0)&(state.agent_a_pos[0][1]==self.height-1),
            "agent_2_stationary_reward": (state.agent_b_pos[0][0]==0)&(state.agent_b_pos[0][1]==self.height-1),
        }


        info = {
            "agent_0": info_agent_0,
        }

        return (
            lax.stop_gradient(self.get_obs(state)),
            lax.stop_gradient(state),
            rewards,
            dones,
            info
        )

    def _update_pos(self, state: GridState, actions: chex.Array) -> chex.Array:
        """Get updated players position based on their action"""
        @partial(jax.vmap, in_axes=[0, None])
        def _pos(agent_id: int, state: GridState) -> Tuple[int, int]:
            agent_x = jax.lax.select(agent_id==0, state.agent_a_pos[0][0], state.agent_b_pos[0][0])
            agent_y = jax.lax.select(agent_id==0, state.agent_a_pos[0][1], state.agent_b_pos[0][1])
            (dx, dy) = (ACTION_TO_VECTOR[actions[agent_id][0]][0], ACTION_TO_VECTOR[actions[agent_id][0]][1])
            return (agent_x + dx, agent_y + dy)
        pos = _pos(self.agent_range, state)
        return {a: pos[i] for i, a in enumerate(self.agents)}
    
    def _get_other_agent_pos(self, curr_agent_pos: Tuple[int, int], other_agent_pos: Tuple[int, int]) -> chex.Array:
        """Get updated players position if it is in the observable grid size"""
        in_view = (jnp.abs(curr_agent_pos[0] - other_agent_pos[0]) <= self.grid_half_size) & (jnp.abs(curr_agent_pos[1] - other_agent_pos[1]) <= self.grid_half_size)
        x = jax.lax.select(in_view, other_agent_pos[0], -1)
        y = jax.lax.select(in_view, other_agent_pos[1], -1)
        return (x, y)

    def step_agent(self, key: chex.PRNGKey, state: GridState, aidx: int, actions: chex.Array, coor: bool
                ) -> Tuple[GridState, int]:
        cur_player_idx = jnp.zeros(self.num_agents).at[aidx].set(0)
        aidx = (aidx + 1) % self.num_agents
        cur_player_idx = jnp.zeros(self.num_agents).at[aidx].set(1)
        actions1, actions2 = actions[0][0], actions[1][0]
        agent_a_pos = (state.agent_a_pos[0][0] + ACTION_TO_VECTOR[actions1][0], state.agent_a_pos[0][1] + ACTION_TO_VECTOR[actions1][1])
        agent_b_pos = (state.agent_b_pos[0][0] + ACTION_TO_VECTOR[actions2][0], state.agent_b_pos[0][1] + ACTION_TO_VECTOR[actions2][1])

        column_0 = jnp.take(state.reward_map, 0, axis=0)
        column_1 = jnp.take(state.reward_map, 1, axis=0)
        condition_a = (column_0 == agent_a_pos[0]) & (column_1 == agent_a_pos[1])
        condition_b = (column_0 == agent_b_pos[0]) & (column_1 == agent_b_pos[1])

        check_cond_a = jnp.any(condition_a)

        @jax.jit
        def find_index(cond):
            def true_case(cond):
                return jnp.argmax(cond) # only 1 will be true
    
            def false_case(cond):
                return -1
            
            return jax.lax.cond(jnp.any(cond), true_case, false_case, cond)
        idx_a = find_index(condition_a)
        idx_b = find_index(condition_b)

        agent_1_key = jax.lax.select(
            (agent_a_pos[0] == self.width - 1) & (agent_a_pos[1] == self.height - 1),
            jnp.float32(1.0),
            jnp.float32(0.0)
        )

        agent_2_key = jax.lax.select(
            (agent_b_pos[0] == self.width - 1) & (agent_b_pos[1] == self.height - 1),
            jnp.float32(1.0),
            jnp.float32(0.0)
        )
        terminal_cond = jax.lax.select(state.agent_steps + 1 >= self.num_agent_steps, True, False)
        agent_1_key = jax.numpy.where(state.agent_1_key != 0.0, state.agent_1_key, agent_1_key)

        agent_2_key = jax.numpy.where(state.agent_2_key != 0.0, state.agent_2_key, agent_2_key)
        agent_1_box = jnp.float32(jax.lax.select((agent_a_pos[0]==0)&(agent_a_pos[1]==self.height-1), state.stationary_reward, jax.lax.select(terminal_cond, 20*state.non_coordinating_reward, state.non_coordinating_reward)))
        agent_2_box = jnp.float32(jax.lax.select((agent_b_pos[0]==0)&(agent_b_pos[1]==self.height-1), state.stationary_reward, jax.lax.select(terminal_cond, 20*state.non_coordinating_reward, state.non_coordinating_reward)))

        both_agent_key_coord = (agent_1_key == 1.0) & (agent_2_key == 1.0)
        reward = jax.lax.select((idx_a!=-1) & (idx_a==idx_b), jax.lax.select(both_agent_key_coord, self.key_multiplier * jnp.zeros((1, 1)).at[0].set(state.reward_map[2, idx_a]),  jnp.zeros((1, 1)).at[0].set(state.reward_map[2, idx_a])), agent_1_box + agent_2_box)
        logging_reward = reward


        steps = state.agent_steps + 1
        last_non_stationary_move = jax.lax.select((actions1 == 4) & (actions2 == 4), state.last_non_stationary_move, steps)

        obs1 = jax.lax.select(self.override_obs_with_zeros, jnp.zeros((1, self.obs_1_size)), state.obs_vec_1)
        obs2 = jax.lax.select(self.override_obs_with_zeros, jnp.zeros((1, self.obs_2_size)), state.obs_vec_2)

        obs1 = jax.lax.select(self.include_reward_pos, obs1.at[:, self.obs_1_size - 9:self.obs_1_size].set(state.agent_1_reward_map.reshape(1, -1)), obs1)
        obs2 = jax.lax.select(self.include_reward_pos, obs2.at[:, self.obs_2_size - 9:self.obs_2_size].set(state.agent_2_reward_map.reshape(1, -1)), obs2)

        key, subkey1, subkey2, subkey3 = jax.random.split(key, num=4) 
        provide_action = (jnp.abs(agent_a_pos[0] - agent_b_pos[0]) <= self.grid_half_size) & (jnp.abs(agent_a_pos[1] - agent_b_pos[1]) <= self.grid_half_size)
        a_action_b_pov = jax.lax.select(provide_action, actions1, -1)
        b_action_a_pov = jax.lax.select(provide_action, actions2, -1)

        a_action_b_pov = jnp.array(a_action_b_pov).reshape(1, 1) 
        b_action_a_pov = jnp.array(b_action_a_pov).reshape(1, 1)
        agent_steps_arr = jnp.array(steps).reshape(1, 1)
        obs1 = jax.lax.select(self.include_time_step, obs1.at[:, 0:1].set(agent_steps_arr), obs1)
        obs2 = jax.lax.select(self.include_time_step, obs2.at[:, 0:1].set(agent_steps_arr), obs2)

        r_mean_arr = jnp.array(self.r_mean).reshape(1, -1)
        key_coord_obs = jax.numpy.where(both_agent_key_coord, 1.0, 0.0)
        r_mean_and_noise_sigma = jnp.concatenate((r_mean_arr, jnp.array(key_coord_obs).reshape(1, 1)), axis=-1)
        obs1 = jax.lax.select(self.include_r_mean_noise_sigma, obs1.at[:, 1:5].set(r_mean_and_noise_sigma), obs1)
        obs2 = jax.lax.select(self.include_r_mean_noise_sigma, obs2.at[:, 1:5].set(r_mean_and_noise_sigma), obs2)

        agent_1_sigma = jnp.array(self.sigma1).reshape(1, 1) 
        agent_2_sigma = jnp.array(self.sigma2).reshape(1, 1) 
        obs1 = jax.lax.select(self.include_agent_noise_sigmas, obs1.at[:, 5:6].set(agent_1_sigma), obs1)
        obs2 = jax.lax.select(self.include_agent_noise_sigmas, obs2.at[:, 5:6].set(agent_2_sigma), obs2)


        agent_b_pos_a_pov = jnp.array(self._get_other_agent_pos(agent_a_pos, agent_b_pos)).reshape(1, -1) 
        agent_a_pos_b_pov = jnp.array(self._get_other_agent_pos(agent_b_pos, agent_a_pos)).reshape(1, -1)

        obs1 = jax.lax.select(self.include_other_agent_pos, obs1.at[:, 6:8].set(agent_b_pos_a_pov), obs1)
        obs2 = jax.lax.select(self.include_other_agent_pos, obs2.at[:, 6:8].set(agent_a_pos_b_pov), obs2)

        obs1 = obs1.at[:, 8:11].set(state.full_obs1[:, 8:11])
        obs2 = obs2.at[:, 8:11].set(state.full_obs2[:, 8:11])

        obs1 = jax.lax.select(self.include_agent_pos, obs1.at[:, 11:13].set(jnp.array(agent_a_pos).reshape(1, -1)), obs1)
        obs2 = jax.lax.select(self.include_agent_pos, obs2.at[:, 11:13].set(jnp.array(agent_b_pos).reshape(1, -1)), obs2)

        obs_vec_1 = obs1
        obs_vec_2 = obs2
        agent_grid = self.cross_play_grid_view(agent_a_pos, agent_b_pos, state.grid_1, state.grid_2)
        agent_grid_1, agent_grid_2 = jnp.ravel(agent_grid['agent_0']).reshape(1, -1), jnp.ravel(agent_grid['agent_1']).reshape(1, -1)
        
        full_obs1 = obs_vec_1.at[:, 13:13 + self.agent_1_view_size*self.agent_1_view_size].set(agent_grid_1)
        full_obs2 = obs_vec_2.at[:, 13:13 + self.agent_2_view_size*self.agent_2_view_size].set(agent_grid_2)

        agent_a_pos = jnp.array(agent_a_pos).reshape(1, -1)
        agent_b_pos = jnp.array(agent_b_pos).reshape(1, -1)
        total_reward = state.total_reward + logging_reward[0][0]

        agent_pos_coord = ((agent_b_pos[0][0] == agent_a_pos[0][0]) & (agent_b_pos[0][1] == agent_a_pos[0][1]))
        both_agent_coord_before_stationary = jax.lax.select((actions1 != 4) & (actions2 != 4) & agent_pos_coord, state.both_agent_coord_before_stationary + 1, state.both_agent_coord_before_stationary)

        agent_1_high_reward_box_seen = (state.reward_map[0][state.high_reward_idx] == agent_a_pos[0][0]) & (state.reward_map[1][state.high_reward_idx] == agent_a_pos[0][1])
        agent_1_mid_reward_box_seen = (state.reward_map[0][state.mid_reward_idx] == agent_a_pos[0][0]) & (state.reward_map[1][state.mid_reward_idx] == agent_a_pos[0][1])
        agent_1_low_reward_box_seen = (state.reward_map[0][state.low_reward_idx] == agent_a_pos[0][0]) & (state.reward_map[1][state.low_reward_idx] == agent_a_pos[0][1])
        agent_2_high_reward_box_seen = (state.reward_map[0][state.high_reward_idx] == agent_b_pos[0][0]) & (state.reward_map[1][state.high_reward_idx] == agent_b_pos[0][1])
        agent_2_mid_reward_box_seen = (state.reward_map[0][state.mid_reward_idx] == agent_b_pos[0][0]) & (state.reward_map[1][state.mid_reward_idx] == agent_b_pos[0][1])
        agent_2_low_reward_box_seen = (state.reward_map[0][state.low_reward_idx] == agent_b_pos[0][0]) & (state.reward_map[1][state.low_reward_idx] == agent_b_pos[0][1])

        agent_1_low_reward_seen = (agent_1_low_reward_box_seen | state.agent_1_low_reward_seen)
        agent_2_low_reward_seen = (agent_2_low_reward_box_seen | state.agent_2_low_reward_seen)
        agent_1_mid_reward_seen = (agent_1_mid_reward_box_seen | state.agent_1_mid_reward_seen)
        agent_2_mid_reward_seen = (agent_2_mid_reward_box_seen | state.agent_2_mid_reward_seen)
        agent_1_high_reward_seen = (agent_1_high_reward_box_seen | state.agent_1_high_reward_seen)
        agent_2_high_reward_seen = (agent_2_high_reward_box_seen | state.agent_2_high_reward_seen)
        agent_1_reward_box_seen = jnp.int32(agent_1_low_reward_seen) + jnp.int32(agent_1_mid_reward_seen) + jnp.int32(agent_1_high_reward_seen)
        agent_2_reward_box_seen = jnp.int32(agent_2_low_reward_seen) + jnp.int32(agent_2_mid_reward_seen) + jnp.int32(agent_2_high_reward_seen)
        return state.replace(terminal_cond=terminal_cond, agent_b_pos_a_pov=agent_b_pos_a_pov, agent_a_pos_b_pov=agent_a_pos_b_pov, agent_steps=steps, 
                        full_obs1=full_obs1, full_obs2=full_obs2, obs_vec_1=obs_vec_1, obs_vec_2=obs_vec_2, 
                        agent_grid_1=agent_grid_1, agent_grid_2=agent_grid_2, agent_a_pos=agent_a_pos, agent_b_pos=agent_b_pos, total_reward=total_reward, last_non_stationary_move=last_non_stationary_move,
                        both_agent_coord_before_stationary=both_agent_coord_before_stationary, agent_1_reward_box_seen=agent_1_reward_box_seen, agent_2_reward_box_seen=agent_2_reward_box_seen,
                        agent_1_low_reward_seen=agent_1_low_reward_seen, agent_2_low_reward_seen=agent_2_low_reward_seen, agent_1_mid_reward_seen=agent_1_mid_reward_seen,
                        agent_2_mid_reward_seen=agent_2_mid_reward_seen, agent_1_high_reward_seen=agent_1_high_reward_seen, agent_2_high_reward_seen=agent_2_high_reward_seen, logging_reward=logging_reward, agent_1_key=agent_1_key, agent_2_key=agent_2_key), reward

    def terminal(self, state: GridState) -> bool:
        return state.terminal_cond

    @property
    def name(self) -> str:
        """Environment Name """
        return "GridEnvComplex"
    
    def action_space(self, agent: str):
        return self.action_spaces[agent]
    
    @property
    def num_actions(self) -> int:
        """Number of actions possible in environment."""
        return self.n_actions

    def observation_space(self, agent: str):
        return self.observation_spaces[agent]
        
    def deepcopy(self, x):
        return jax.tree_util.tree_map(lambda y: y, x)


