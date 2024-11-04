from multi_agent_env import MultiAgentEnv
import chex
import jax
import jax.numpy as jnp
from typing import Tuple, Dict
from gymnax.environments.spaces import Discrete
from functools import partial
from flax import struct
from jax import lax
from jax.lax import dynamic_update_slice

ACTION_TO_VECTOR = jnp.array([(dx, dy) for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0), (0, 0)]])

@struct.dataclass
class MirrorState:
    """
    Dataclass to store the state of the Mirror environment.
    
    Attributes:
        payoffs (chex.Array): Payoffs for boxes. Shape (num_cols, 3)
        terminal_cond (bool): True if the episode is terminated.
        agent_b_pos_a_pov (chex.Array): Position(x, y) of agent B from the perspective of agent A.
        agent_a_pos_b_pov (chex.Array): Position(x, y) of agent A from the perspective of agent B.
        terminal_agent (chex.Array): Not used currently.
        full_obs1 (chex.Array): Full observation vector for agent 1(includes agent 1 partial observability grid).
        full_obs2 (chex.Array): Full observation vector for agent 2(includes agent 2 partial observability grid).
        obs_vec_1 (chex.Array): Observation vector for agent 1 without the partial observability grid.
        obs_vec_2 (chex.Array): Observation vector for agent 2 without the partial observability grid.
        agent_reward_grid_1 (chex.Array): Reward grid view for agent 1.
        agent_reward_grid_2 (chex.Array): Reward grid view for agent 2.
        agent_other_pos_grid_1: grid view for agent 1 with other agent pos if other agent in view
        agent_other_pos_grid_2: grid view for agent 2 with other agent pos if other agent in view
        agent_steps (int): Number of steps taken by the agents.
        cur_player_idx (chex.Array): Index of the current player.
        reward_map (chex.Array): Mapping of box positions(x, y) and their rewards.
        grid_1 (chex.Array): Global grid for agent 1 with their noisy reward
        grid_2 (chex.Array): Global grid for agent 2 with their noisy reward
        non_prize_reward (chex.Array): Reward for not landing on a box.
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
    agent_reward_grid_1: chex.Array
    agent_reward_grid_2: chex.Array
    agent_other_pos_grid_1: chex.Array
    agent_other_pos_grid_2: chex.Array
    agent_steps: int
    cur_player_idx: chex.Array
    reward_map: chex.Array
    grid_1: chex.Array
    grid_2: chex.Array
    non_prize_reward: chex.Array
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
    both_agent_coord_before_stationary: int
    high_reward_idx: int
    mid_reward_idx: int
    low_reward_idx: int
    agent_0_last_non_stationary_move: int
    agent_1_last_non_stationary_move: int
    agent_1_view: int
    agent_2_view: int
    logging_reward: chex.Array

def max_of_n_gaussians(n, mean, sigma):
    """
    Utility funciton which gives the best possible performance.
    Logic from:
    https://math.stackexchange.com/questions/473229/expected-value-of-maximum-and-minimum-of-n-normal-random-variables/510580#510580
    
    Returns:
        float: The best possible performance.
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


class MirrorEnv(MultiAgentEnv):
    def __init__(self, num_agents=2, n_actions=5, agents=None, obs_size=None,
             non_coordinating_reward=1, non_prize_reward=-1, num_agent_steps=1024,
            width=10, height=10, agent_start_pos=[0, 0], r_mean=[[5, 5, 5], [6, 6, 6], [7, 7, 7], [8, 8, 8]],
            include_r_mean_noise_sigma=True, include_prev_acts_in_obs=True, include_noise_sigmas=True, include_other_agent_pos=True,
            action_spaces=None, observation_space=None, agent_view_size=3, sigma=2, sigma1=0, sigma2=0,
            agent_pos_other_play=False, include_agent_noise_sigmas=True, include_prev_reward_in_obs=False, override_obs_with_zeros=True,
            lever_other_play=False, include_reward_pos=False, include_agent_pos=False,
            max_rand_start_agent_x_pos=3, max_rand_start_agent_y_pos=3, include_time_step=False, agent_1_view_size=3, agent_2_view_size=3,
            reward_pos_other_play=True, sigma1_arr=[2, 2, 2, 2], sigma2_arr=[2, 2, 2, 2], include_reward=True, final_step_penalty=-10,
            agent_1_min_view_size=1, agent_1_max_view_size=3, agent_2_min_view_size=1, agent_2_max_view_size=3, include_rand_agent_view_size=True,
            agent_1_y_view_size=11, agent_2_y_view_size=11):
            super().__init__(num_agents)

            self.num_agents = num_agents
            self.agent_range = jnp.arange(num_agents)
            self.num_boxes_per_col = len(r_mean[0])
            self.r_mean = jnp.array(r_mean)
            self.n_actions = n_actions
            self.num_agent_steps = num_agent_steps
            self.width = width
            self.height = height
            self.agent_1_min_view_size = agent_1_min_view_size
            self.agent_1_max_view_size = agent_1_max_view_size
            self.agent_2_min_view_size = agent_2_min_view_size
            self.agent_2_max_view_size = agent_2_max_view_size
            self.include_rand_agent_view_size = include_rand_agent_view_size
            self.lever_other_play = lever_other_play
            self.reward_pos_other_play = reward_pos_other_play
            
            if agents is None:
                self.agents = [f"agent_{i}" for i in range(num_agents)]
            else:
                assert len(
                    agents) == num_agents, f"Number of agents {len(agents)} does not match number of agents {num_agents}"
                self.agents = agents

            if obs_size is None:
                obs_dim = 0
                # if include_prev_acts_in_obs:
                obs_dim += 1
                # if include_r_mean_noise_sigma:
                obs_dim += 3+1
                # if include_noise_sigmas:
                obs_dim += 1
                # include_agent_pos:
                obs_dim += 2
                # if include_other_agent_pos:
                obs_dim += 2 # for x and y pos
                # include obs view size
                obs_dim += 2
                obs_1_dim = obs_dim
                obs_2_dim = obs_dim
                self.agent_1_view_size = jax.lax.select(self.include_rand_agent_view_size, self.agent_1_max_view_size, agent_1_view_size) 
                self.agent_2_view_size = jax.lax.select(self.include_rand_agent_view_size, self.agent_2_max_view_size, agent_2_view_size)
                self.agent_1_y_view_size = agent_1_y_view_size
                self.agent_2_y_view_size = agent_2_y_view_size
                self.agent_1_view_size += 1
                self.agent_2_view_size += 1
                self.agent_1_obs_grid_size = self.agent_1_view_size * self.agent_1_y_view_size
                self.agent_2_obs_grid_size = self.agent_2_view_size * self.agent_2_y_view_size
                obs_1_dim += self.agent_1_obs_grid_size
                obs_2_dim += self.agent_2_obs_grid_size
                self.obs_1_size = obs_1_dim
                self.obs_2_size = obs_2_dim
                self.action_set = jnp.arange(self.n_actions)
                if action_spaces is None:
                    self.action_spaces = {i: Discrete(self.n_actions) for i in self.agents}
                if observation_space is None:
                    self.observation_spaces = {i: Discrete(self.obs_1_size if i == 'agent_0' else self.obs_2_size) for i in self.agents}

            self.agent_start_pos = agent_start_pos
            self.non_coordinating_reward = non_coordinating_reward
            self.sigma = sigma
            self.sigma1 = sigma1
            self.sigma2 = sigma2
            self.sigma1_arr = jnp.array(sigma1_arr)
            self.sigma2_arr = jnp.array(sigma2_arr)
            self.non_prize_reward = non_prize_reward
            best_reward_possible = max_of_n_gaussians(3,
                                              jnp.max(self.r_mean[:, 0]),
                                              self.sigma)*(1)
            self.best_reward_possible = best_reward_possible
            self.payoffs = jnp.zeros((self.width, self.num_boxes_per_col))
            self.agent_pos_other_play = agent_pos_other_play
            self.lever_pos = jnp.array([[self.width - 1, self.width - 1, self.width - 1], [1, 2, 3]])
            self.include_r_mean_noise_sigma = include_r_mean_noise_sigma
            self.include_prev_acts_in_obs = include_prev_acts_in_obs
            self.include_other_agent_pos = include_other_agent_pos
            self.include_agent_noise_sigmas = include_agent_noise_sigmas
            self.include_prev_reward_in_obs = include_prev_reward_in_obs
            self.override_obs_with_zeros = override_obs_with_zeros
            self.include_reward_pos = include_reward_pos
            self.include_reward = include_reward
            self.include_agent_pos = include_agent_pos
            self.max_rand_start_agent_x_pos = max_rand_start_agent_x_pos
            self.max_rand_start_agent_y_pos = max_rand_start_agent_y_pos
            self.include_time_step = include_time_step
            self.final_step_penalty = final_step_penalty * num_agent_steps

    
    def _gen_grid(self, key: chex.PRNGKey, reward_1, reward_2, default_reward_box_pos):
        """
        On reset(), generate the grid with the boxes and agent positions.
        
        Args:
            key (chex.PRNGKey): Random number generator key.
            reward (chex.Array): Rewards for the boxes.
            
        Returns:
            Tuple[chex.Array, Tuple[int, int]]: The generated grid and the initial agent position.
        """
        grid_1 = jnp.zeros((self.width, self.height))
        grid_2 = jnp.zeros((self.width, self.height))


        keys = jax.random.split(key, num=self.width)

        grid_1 = jnp.zeros((self.width, self.height))
        grid_2 = jnp.zeros((self.width, self.height))
        rand_idx = jnp.zeros((self.width, self.num_boxes_per_col))
        row_idx = jnp.arange(self.width).reshape(-1, 1) * jnp.ones((1, self.num_boxes_per_col))
        keys = jax.random.split(key, num=self.width)
        for i in jnp.arange(self.width):
            random_indices = jax.random.choice(keys[i], self.height, shape=(3,), replace=False)
            indices = jax.lax.select(self.reward_pos_other_play, random_indices, default_reward_box_pos[i])
            col_grid_1 = jnp.zeros(self.height)
            col_grid_2 = jnp.zeros(self.height)
            for j in jnp.arange(self.num_boxes_per_col):
                col_grid_1 = col_grid_1.at[indices[j]].set(reward_1[i, j])
                col_grid_2 = col_grid_2.at[indices[j]].set(reward_2[i, j])
            grid_1 = grid_1.at[i, :].set(col_grid_1)
            grid_2 = grid_2.at[i, :].set(col_grid_2)
            rand_idx = rand_idx.at[i, :].set(random_indices)
            

        key, agent_1_pos_x, agent_1_pos_y, agent_2_pos_x, agent_2_pos_y = jax.random.split(key, num=5)


        rand_1_x = jax.lax.select(self.agent_pos_other_play, (jax.random.randint(agent_1_pos_x, (1,), minval=0, maxval=self.max_rand_start_agent_x_pos)), jnp.array([self.agent_start_pos[0]]))
        rand_1_y = jax.lax.select(self.agent_pos_other_play, (jax.random.randint(agent_1_pos_y, (1,), minval=0, maxval=self.max_rand_start_agent_y_pos)), jnp.array([self.agent_start_pos[1]]))
        agent_1_pos = (rand_1_x, rand_1_y)

        rand_2_x = jax.lax.select(self.agent_pos_other_play, (jax.random.randint(agent_2_pos_x, (1,), minval=0, maxval=self.max_rand_start_agent_x_pos)), jnp.array([self.agent_start_pos[0]]))
        rand_2_y = jax.lax.select(self.agent_pos_other_play, (jax.random.randint(agent_2_pos_y, (1,), minval=0, maxval=self.max_rand_start_agent_y_pos)), jnp.array([self.agent_start_pos[1]]))
        agent_2_pos = (rand_2_x, rand_2_y)

        self.mission = "Coordinate with other agent to one of the boxes"
        return grid_1, agent_1_pos, grid_2, agent_2_pos, rand_idx, row_idx
    
    def reset(self, key: chex.PRNGKey) -> Tuple[Dict, MirrorState]:
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
        payoffs = jax.random.normal(subkey1, (self.width, self.num_boxes_per_col)) * self.sigma + self.r_mean
        sorted_indices = jnp.argsort(-payoffs, axis=1)


        ranked_array = jnp.empty_like(payoffs, dtype=jnp.int32)


        ranked_reward_idx = jnp.argsort(sorted_indices, axis=1)

        eta1 = jax.random.normal(subkey2, (self.width, self.num_boxes_per_col)) * self.sigma1_arr[:, None]
        eta2 = jax.random.normal(subkey3, (self.width, self.num_boxes_per_col)) * self.sigma2_arr[:, None]
    
        obs1_val = payoffs + eta1
        obs2_val = payoffs + eta2

        default_reward_box_pos = jnp.tile(jnp.array([0, 1, 2]), (self.width, 1))

        grid_1, agent_a_pos, grid_2, agent_b_pos, rand_reward_pos_idx, row_idx  = self._gen_grid(key=key, reward_1=obs1_val, reward_2=obs2_val, default_reward_box_pos=default_reward_box_pos)
        reward_map = jnp.array([row_idx, rand_reward_pos_idx, payoffs])
        agent_1_reward_map = jnp.array([rand_reward_pos_idx, obs1_val])
        agent_2_reward_map = jnp.array([rand_reward_pos_idx, obs2_val])

        agent_b_pos = (agent_b_pos[0][0], agent_b_pos[1][0])
        agent_a_pos = (agent_a_pos[0][0], agent_a_pos[1][0])

        agent_steps = 0
        
        key, view_key1, view_key2 = jax.random.split(key, num=3)
        self.agent_1_view = jax.lax.select(self.include_rand_agent_view_size, 1 + jax.random.randint(view_key1, (), minval=self.agent_1_min_view_size, maxval=self.agent_1_max_view_size + 1), self.agent_1_view_size)
        self.agent_2_view = jax.lax.select(self.include_rand_agent_view_size, 1 + jax.random.randint(view_key2, (), minval=self.agent_2_min_view_size, maxval=self.agent_2_max_view_size + 1), self.agent_2_view_size)

        min_max_values_agent_1 = jnp.array([self.agent_1_min_view_size, self.agent_1_max_view_size])
        min_max_values_agent_2 = jnp.array([self.agent_2_min_view_size, self.agent_2_max_view_size])
        self.agent_1_view = jax.lax.select(
            self.include_rand_agent_view_size,
            jax.random.choice(view_key1, min_max_values_agent_1),
            self.agent_1_view_size
        )

        self.agent_2_view = jax.lax.select(
            self.include_rand_agent_view_size,
            jax.random.choice(view_key2, min_max_values_agent_2),
            self.agent_2_view_size
        )
        
        agent_1_view = self.agent_1_view.astype(int)
        agent_2_view = self.agent_2_view.astype(int)
        
        a_action_b_pov, b_action_a_pov = -1, -1
        a_action_b_pov = jnp.array(a_action_b_pov).reshape(1, 1) 
        b_action_a_pov = jnp.array(b_action_a_pov).reshape(1, 1)
        agent_steps_arr = jnp.array(agent_steps).reshape(1, 1)
        obs1 = jax.lax.select(self.include_time_step, obs1.at[:, 0:1].set(agent_steps_arr), obs1)
        obs2 = jax.lax.select(self.include_time_step, obs2.at[:, 0:1].set(agent_steps_arr), obs2)

        agent_1_sigma = jnp.array(self.sigma1_arr[0]).reshape(1, 1) 
        agent_2_sigma = jnp.array(self.sigma2_arr[0]).reshape(1, 1) 
        obs1 = jax.lax.select(self.include_agent_noise_sigmas, obs1.at[:, 6:7].set(self.sigma1_arr[0]), obs1)
        obs2 = jax.lax.select(self.include_agent_noise_sigmas, obs2.at[:, 6:7].set(self.sigma2_arr[0]), obs2)

        agent_b_pos_a_pov = jnp.array(self._get_other_agent_pos(agent_a_pos, agent_b_pos, 1, agent_1_view, agent_2_view)).reshape(1, -1)
        agent_a_pos_b_pov = jnp.array(self._get_other_agent_pos(agent_b_pos, agent_a_pos, 2, agent_1_view, agent_2_view)).reshape(1, -1)
        obs1 = jax.lax.select(self.include_other_agent_pos, obs1.at[:, 7:9].set(agent_b_pos_a_pov), obs1)
        obs2 = jax.lax.select(self.include_other_agent_pos, obs2.at[:, 7:9].set(agent_a_pos_b_pov), obs2)

        obs1 = jax.lax.select(self.include_agent_pos, obs1.at[:, 9:11].set(jnp.array(agent_a_pos).reshape(1, -1)), obs1)
        obs2 = jax.lax.select(self.include_agent_pos, obs2.at[:, 9:11].set(jnp.array(agent_b_pos).reshape(1, -1)), obs2)

        obs1 = jax.lax.select(self.include_rand_agent_view_size, obs1.at[:, 11:12].set(self.agent_1_view), obs1)
        obs1 = jax.lax.select(self.include_rand_agent_view_size, obs1.at[:, 12:13].set(self.agent_2_view), obs1)

        obs2 = jax.lax.select(self.include_rand_agent_view_size, obs2.at[:, 11:12].set(self.agent_2_view), obs2)
        obs2 = jax.lax.select(self.include_rand_agent_view_size, obs2.at[:, 12:13].set(self.agent_1_view), obs2)

        obs_vec_1 = obs1
        obs_vec_2 = obs2

        agent_grid = self.cross_play_grid_view(agent_a_pos, agent_b_pos, grid_1, grid_2, agent_1_view, agent_2_view)
        agent_grid_1, agent_grid_2 = jnp.ravel(agent_grid['agent_0']).reshape(1, -1), jnp.ravel(agent_grid['agent_1']).reshape(1, -1)
        agent_other_pos_grid_1, agent_other_pos_grid_2 = jnp.ravel(agent_grid['agent_0_pos_grid']).reshape(1, -1), jnp.ravel(agent_grid['agent_1_pos_grid']).reshape(1, -1)

        start_indices = (0, 13)
        full_obs1 = dynamic_update_slice(obs_vec_1, agent_grid_1, start_indices)

        start_indices = (0, 13)
        full_obs2 = dynamic_update_slice(obs_vec_2, agent_grid_2, start_indices) 

        non_coordinating_reward = jnp.zeros((1, 1)).at[0].set(self.non_coordinating_reward)
        non_prize_reward = jnp.zeros((1, 1)).at[0].set(self.non_prize_reward)
        
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

        both_agent_coord_before_stationary = 0
        agent_pos_coord = ((agent_b_pos[0][0] == agent_a_pos[0][0]) & (agent_b_pos[0][1] == agent_a_pos[0][1]))
        both_agent_coord_before_stationary = jax.lax.select(agent_pos_coord, both_agent_coord_before_stationary + 1, both_agent_coord_before_stationary)
        agent_0_last_non_stationary_move = 0
        agent_1_last_non_stationary_move = 0
        state = MirrorState(payoffs=payoffs, terminal_cond=terminal_cond, agent_b_pos_a_pov=agent_b_pos_a_pov, agent_a_pos_b_pov=agent_a_pos_b_pov,
                        terminal_agent=terminal_agent, agent_steps=agent_steps, full_obs1=full_obs1, full_obs2=full_obs2, reward_map=reward_map,
                        obs_vec_1=obs_vec_1, obs_vec_2=obs_vec_2, agent_reward_grid_1=agent_grid_1, agent_reward_grid_2=agent_grid_2, cur_player_idx=cur_player_idx,
                        grid_1=grid_1, grid_2=grid_2, non_coordinating_reward=non_coordinating_reward, non_prize_reward=non_prize_reward, agent_a_pos=agent_a_pos,
                        agent_b_pos=agent_b_pos, agent_1_reward_map=agent_1_reward_map, agent_2_reward_map=agent_2_reward_map, total_reward=total_reward, last_non_stationary_move=last_non_stationary_move,
                        agent_1_start_pos=agent_a_pos, agent_2_start_pos=agent_b_pos, agent_1_reward_box_seen=agent_1_reward_box_seen, agent_2_reward_box_seen=agent_2_reward_box_seen, 
                        agent_1_low_reward_seen=agent_1_low_reward_seen, agent_1_mid_reward_seen=agent_1_mid_reward_seen, agent_1_high_reward_seen=agent_1_high_reward_seen,
                        agent_2_low_reward_seen=agent_2_low_reward_seen, agent_2_mid_reward_seen=agent_2_mid_reward_seen, agent_2_high_reward_seen=agent_2_high_reward_seen,
                        both_agent_coord_before_stationary=both_agent_coord_before_stationary, high_reward_idx=0, mid_reward_idx=1, low_reward_idx=2,
                        agent_other_pos_grid_1=agent_other_pos_grid_1, agent_other_pos_grid_2=agent_other_pos_grid_2, agent_0_last_non_stationary_move=agent_0_last_non_stationary_move, 
                        agent_1_last_non_stationary_move=agent_1_last_non_stationary_move, agent_1_view=agent_1_view, agent_2_view=agent_2_view, logging_reward=logging_reward)
        return self.get_obs(state), state
    
    @partial(jax.jit, static_argnums=[0])
    def get_pos_moves(self, state: MirrorState) -> chex.Array:
        """
        Get the legal moves for each agent based on their current position.
        
        Args:
            state (GridState): The current state of the environment.
            
        Returns:
            chex.Array: A dictionary mapping agent names to boolean array for legal moves out of 5 actions.
        """
        @partial(jax.vmap, in_axes=[0, None])
        def _legal_moves(aidx: int, state: MirrorState) -> chex.Array:
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
            return moves
        pos_moves = _legal_moves(self.agent_range, state)
        return {a: pos_moves[i] for i, a in enumerate(self.agents)}
    
    def cross_play_grid_view(self, agent_a_pos, agent_b_pos, grid_1, grid_2, agent_1_view, agent_2_view) -> chex.Array:
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
        def get_agent_reward_grid(agent_pos, view_size, agent_grid, grid):
            def cond_func(loop_carry):
                i, _ = loop_carry
                return i < view_size * (2 * view_size + 1)
            def body_func(loop_carry):
                i, agent_grid = loop_carry
                x = jnp.abs(agent_pos[0] - (i % view_size))
                y = agent_pos[1] - view_size + (i //view_size)
                in_bounds = (0 <= x) & (x < self.width) & (0 <= y) & (y < self.height)
                updated_grid_val = jax.lax.cond(in_bounds, lambda: grid[x, y], lambda: jnp.array([-1.0])[0])
                agent_grid = agent_grid.at[i // (2 * view_size + 1), i % (2 * view_size + 1)].set(updated_grid_val)
                return i + 1, agent_grid
            _, agent_grid = jax.lax.while_loop(cond_func, body_func, (0, agent_grid))
            return agent_grid
        
        def get_agent_pos_grid(agent_pos, view_size, other_agent_pos):
            agent_grid = jnp.zeros((view_size, 2 * view_size + 1))
            for i in range(view_size * (2 * view_size + 1)):
                x = jnp.abs(agent_pos[0] - (i % view_size))
                y = agent_pos[1] - view_size + (i // view_size)
                in_bounds = (0 <= x) & (x < self.width) & (0 <= y) & (y < self.height)
                def in_bounds_fn():
                    curr_pos = jnp.logical_and(agent_pos[0] == x, agent_pos[1] == y)
                    return jax.lax.select(
                        jnp.logical_and(curr_pos, jax.numpy.array_equal(agent_pos, other_agent_pos)),
                        jnp.array([2.0]),
                        jnp.array([0.0])
                    )

                def out_of_bounds_fn():
                    return jnp.array([-1.0])
                updated_grid_val = jax.lax.cond(in_bounds, in_bounds_fn, out_of_bounds_fn)[0]
                agent_grid = agent_grid.at[i // view_size, i % view_size].set(updated_grid_val)
            return agent_grid
        agent_a_grid = jnp.zeros((self.agent_1_view_size, self.agent_1_y_view_size))
        agent_a_grid = get_agent_reward_grid(agent_a_pos, self.agent_1_view_size, agent_a_grid, grid_1)
        agent_b_grid = jnp.zeros((self.agent_2_view_size, self.agent_2_y_view_size))
        agent_b_grid = get_agent_reward_grid(agent_b_pos, self.agent_2_view_size, agent_b_grid, grid_2)
        agent_a_pos_grid = get_agent_pos_grid(agent_a_pos, 3, agent_b_pos)
        agent_b_pos_grid = get_agent_pos_grid(agent_b_pos, 3, agent_a_pos)
        
        return {self.agents[0]: agent_a_grid, self.agents[1]: agent_b_grid, 'agent_0_pos_grid': agent_a_pos_grid, 'agent_1_pos_grid': agent_b_pos_grid}

    @partial(jax.jit, static_argnums=[0])
    def get_obs(self, state: MirrorState) -> Dict:
        """Get all the agents' observations(both agent a and b)"""
        def _observation(agent_id):
            if agent_id == 0:
                return state.full_obs1
            else:
                return state.full_obs2
        obs = {a: _observation(agent_id) for agent_id, a in enumerate(self.agents)}
        return obs


    @partial(jax.jit, static_argnums=[0])
    def step_env(self, key: chex.PRNGKey, state: MirrorState, actions: Dict,
                 ) -> Tuple[chex.Array, MirrorState, Dict, Dict, Dict]:
        actions = jnp.array([actions[i] for i in self.agents])
        action1, action2 = actions[0], actions[1]
        actions1 = jnp.asarray(action1, dtype=jnp.int32)
        actions2 = jnp.asarray(action2, dtype=jnp.int32)
        coor = actions1 == actions2

        aidx = jnp.nonzero(state.cur_player_idx, size=1)[0][0]
        state, reward = self.step_agent(key, state, aidx, actions, coor)

        max_reward = jnp.max(state.payoffs)

        done = self.terminal(state) 
        dones = {agent: done for agent in self.agents}
        dones["__all__"] = done

        rewards = {agent: reward for agent in self.agents}
        rewards["__all__"] = reward
        regret = (self.best_reward_possible - state.logging_reward) / self.best_reward_possible

        agent_pos_coord = ((state.agent_b_pos[0][0] == state.agent_a_pos[0][0]) & (state.agent_b_pos[0][1] == state.agent_a_pos[0][1]))
        agent_reward_box_coord = jnp.logical_and(
            agent_pos_coord,
            jnp.logical_and(state.logging_reward[0][0] != 2.0, state.logging_reward[0][0] != 0.0)
        )
        max_value = jnp.max(state.reward_map[2])
        max_index = jnp.argmax(state.reward_map[2])
        i, j = jnp.unravel_index(max_index, state.reward_map[2].shape)
        last_row = state.reward_map[2][-1, :]
        max_value_last_row = jnp.max(last_row)  
        max_value_last_row_idx = jnp.argmax(last_row) 

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
            "agent_reward_box_coord": agent_reward_box_coord,
            "total_reward": state.total_reward,
            "last_non_stationary_move": state.last_non_stationary_move,
            "agent_1_reward_box_stepped_on": state.agent_1_reward_box_seen,
            "agent_2_reward_box_stepped_on": state.agent_2_reward_box_seen,
            "both_agent_coord_before_stationary": state.both_agent_coord_before_stationary,
            "agent_0_last_non_stationary_move": state.agent_0_last_non_stationary_move,
            "agent_1_last_non_stationary_move": state.agent_1_last_non_stationary_move,
            "max_value": max_value,
            "index_of_max_value": max_index,
            "x_idx_max_value": i,
            "y_idx_max_value": state.reward_map[1][i][j],
            "max_value_last_row": max_value_last_row,
            "max_value_last_row_idx": max_value_last_row_idx,
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

    def _update_pos(self, state: MirrorState, actions: chex.Array) -> chex.Array:
        """Get updated players position based on their action"""
        @partial(jax.vmap, in_axes=[0, None])
        def _pos(agent_id: int, state: MirrorState) -> Tuple[int, int]:
            agent_x = jax.lax.select(agent_id==0, state.agent_a_pos[0][0], state.agent_b_pos[0][0])
            agent_y = jax.lax.select(agent_id==0, state.agent_a_pos[0][1], state.agent_b_pos[0][1])
            (dx, dy) = (ACTION_TO_VECTOR[actions[agent_id][0]][0], ACTION_TO_VECTOR[actions[agent_id][0]][1])
            return (agent_x + dx, agent_y + dy)
        pos = _pos(self.agent_range, state)
        return {a: pos[i] for i, a in enumerate(self.agents)}
    
    def _get_other_agent_pos(self, curr_agent_pos: Tuple[int, int], other_agent_pos: Tuple[int, int], agent_num: int, agent_1_view, agent_2_view) -> chex.Array:
        """Get updated players position if it is in the observable grid size"""
        agent_1_in_view = (-(curr_agent_pos[0] - agent_1_view + 1) > other_agent_pos[0]) & (jnp.abs(curr_agent_pos[1] - other_agent_pos[1]) <= (agent_1_view))
        agent_2_in_view = (-(curr_agent_pos[0] - agent_2_view + 1) > other_agent_pos[0]) & (jnp.abs(curr_agent_pos[1] - other_agent_pos[1]) <= (agent_2_view))
        in_view = jax.lax.select(agent_num==1, agent_1_in_view, agent_2_in_view)
        x = jax.lax.select(in_view, other_agent_pos[0], -1)
        y = jax.lax.select(in_view, other_agent_pos[1], -1)
        return (x, y)

    def step_agent(self, key: chex.PRNGKey, state: MirrorState, aidx: int, actions: chex.Array, coor: bool
                ) -> Tuple[MirrorState, int]:
        cur_player_idx = jnp.zeros(self.num_agents).at[aidx].set(0)
        aidx = (aidx + 1) % self.num_agents
        cur_player_idx = jnp.zeros(self.num_agents).at[aidx].set(1)
        actions1, actions2 = actions[0][0], actions[1][0]

        agent_a_pos = (state.agent_a_pos[0][0] + ACTION_TO_VECTOR[actions1][0], state.agent_a_pos[0][1] + ACTION_TO_VECTOR[actions1][1])
        agent_b_pos = (state.agent_b_pos[0][0] + ACTION_TO_VECTOR[actions2][0], state.agent_b_pos[0][1] + ACTION_TO_VECTOR[actions2][1])

        column_0 = state.reward_map[0].flatten()
        column_1 = state.reward_map[1].flatten()
        reward_values = state.reward_map[2].flatten()


        condition_a = (column_0 == agent_a_pos[0]) & (column_1 == agent_a_pos[1])
        condition_b = (column_0 == agent_b_pos[0]) & (column_1 == agent_b_pos[1])

        check_cond_a = jnp.any(condition_a)

        @jax.jit
        def find_index(cond):
            def true_case(cond):
                return jnp.argmax(cond)
    
            def false_case(cond):
                return -1
            
            return jax.lax.cond(jnp.any(cond), true_case, false_case, cond)
        idx_a = find_index(condition_a)
        idx_b = find_index(condition_b)


        agent_1_box = jnp.float32(jax.lax.select(idx_a==-1, state.non_prize_reward, state.non_coordinating_reward))

        agent_2_box = jnp.float32(jax.lax.select(idx_b==-1, state.non_prize_reward, state.non_coordinating_reward))

        terminal_cond = jax.lax.select(state.agent_steps + 1 >= self.num_agent_steps, True, False)

        reward = jax.lax.select((idx_a!=-1) & (idx_a==idx_b), jnp.zeros((1, 1)).at[0].set(reward_values[idx_a]), agent_1_box + agent_2_box) 
        logging_reward = reward
        reward = jax.lax.select(terminal_cond & ~((idx_a!=-1) & (idx_a==idx_b)), reward + self.final_step_penalty, reward)

        steps = state.agent_steps + 1
        last_non_stationary_move = jax.lax.select((actions1 == 4) & (actions2 == 4), state.last_non_stationary_move, steps)
        agent_0_last_non_stationary_move = jax.lax.select((actions1 == 4), state.agent_0_last_non_stationary_move, steps)
        agent_1_last_non_stationary_move = jax.lax.select((actions2 == 4), state.agent_1_last_non_stationary_move, steps)

        obs1 = jax.lax.select(self.override_obs_with_zeros, jnp.zeros((1, self.obs_1_size)), state.obs_vec_1)
        obs2 = jax.lax.select(self.override_obs_with_zeros, jnp.zeros((1, self.obs_2_size)), state.obs_vec_2)

        key, subkey1, subkey2, subkey3 = jax.random.split(key, num=4)

        provide_action = (-(agent_a_pos[0] - state.agent_1_view) > agent_b_pos[0]) & (jnp.abs(agent_a_pos[1] - agent_b_pos[1]) <= (state.agent_1_view // 2) + 1)
        a_action_b_pov = jax.lax.select(provide_action, actions1, -1)
        b_action_a_pov = jax.lax.select(provide_action, actions2, -1)

        a_action_b_pov = jnp.array(a_action_b_pov).reshape(1, 1) 
        b_action_a_pov = jnp.array(b_action_a_pov).reshape(1, 1)
        agent_steps_arr = jnp.array(steps).reshape(1, 1)
        obs1 = jax.lax.select(self.include_time_step, obs1.at[:, 0:1].set(agent_steps_arr), obs1)
        obs2 = jax.lax.select(self.include_time_step, obs2.at[:, 0:1].set(agent_steps_arr), obs2)

        r_mean_arr = jnp.array(self.r_mean[:, 0]).reshape(1, -1)
        r_mean_and_noise_sigma = jnp.concatenate((r_mean_arr, jnp.array(-1).reshape(1, 1)), axis=-1) 

        agent_1_sigma = jnp.array(self.sigma1_arr[0]).reshape(1, 1) 
        agent_2_sigma = jnp.array(self.sigma2_arr[0]).reshape(1, 1) 
        obs1 = jax.lax.select(self.include_agent_noise_sigmas, obs1.at[:, 6:7].set(agent_1_sigma), obs1)
        obs2 = jax.lax.select(self.include_agent_noise_sigmas, obs2.at[:, 6:7].set(agent_2_sigma), obs2)


        agent_b_pos_a_pov = jnp.array(self._get_other_agent_pos(agent_a_pos, agent_b_pos, 1, state.agent_1_view, state.agent_2_view)).reshape(1, -1) 
        agent_a_pos_b_pov = jnp.array(self._get_other_agent_pos(agent_b_pos, agent_a_pos, 2, state.agent_1_view, state.agent_2_view)).reshape(1, -1)

        obs1 = jax.lax.select(self.include_other_agent_pos, obs1.at[:, 7:9].set(agent_b_pos_a_pov), obs1)
        obs2 = jax.lax.select(self.include_other_agent_pos, obs2.at[:, 7:9].set(agent_a_pos_b_pov), obs2)

        obs1 = jax.lax.select(self.include_agent_pos, obs1.at[:, 9:11].set(jnp.array(agent_a_pos).reshape(1, -1)), obs1)
        obs2 = jax.lax.select(self.include_agent_pos, obs2.at[:, 9:11].set(jnp.array(agent_b_pos).reshape(1, -1)), obs2)

        obs1 = jax.lax.select(self.include_rand_agent_view_size, obs1.at[:, 11:12].set(state.agent_1_view), obs1)
        obs1 = jax.lax.select(self.include_rand_agent_view_size, obs1.at[:, 12:13].set(state.agent_2_view), obs1)

        obs2 = jax.lax.select(self.include_rand_agent_view_size, obs2.at[:, 11:12].set(state.agent_2_view), obs2)
        obs2 = jax.lax.select(self.include_rand_agent_view_size, obs2.at[:, 12:13].set(state.agent_1_view), obs2)

        obs_vec_1 = obs1
        obs_vec_2 = obs2

        agent_grid = self.cross_play_grid_view(agent_a_pos, agent_b_pos, state.grid_1, state.grid_2, state.agent_1_view, state.agent_2_view)
        agent_reward_grid_1, agent_reward_grid_2 = jnp.ravel(agent_grid['agent_0']).reshape(1, -1), jnp.ravel(agent_grid['agent_1']).reshape(1, -1)
        agent_other_pos_grid_1, agent_other_pos_grid_2 = jnp.ravel(agent_grid['agent_0_pos_grid']).reshape(1, -1), jnp.ravel(agent_grid['agent_1_pos_grid']).reshape(1, -1)


        start_indices = (0, 13)
        update_shape = (1, state.agent_1_view * state.agent_1_view)
        full_obs1 = dynamic_update_slice(obs_vec_1, agent_reward_grid_1, start_indices)


        start_indices = (0, 13)
        update_shape = (1, state.agent_2_view * state.agent_2_view)
        full_obs2 = dynamic_update_slice(obs_vec_2, agent_reward_grid_2, start_indices)

        agent_a_pos = jnp.array(agent_a_pos).reshape(1, -1)
        agent_b_pos = jnp.array(agent_b_pos).reshape(1, -1)
        total_reward = state.total_reward + logging_reward[0][0]

        agent_pos_coord = ((agent_b_pos[0][0] == agent_a_pos[0][0]) & (agent_b_pos[0][1] == agent_a_pos[0][1]))
        both_agent_coord_before_stationary = jax.lax.select((actions1 != 4) & (actions2 != 4) & agent_pos_coord, state.both_agent_coord_before_stationary + 1, state.both_agent_coord_before_stationary)

        return state.replace(terminal_cond=terminal_cond, agent_b_pos_a_pov=agent_b_pos_a_pov, agent_a_pos_b_pov=agent_a_pos_b_pov, agent_steps=steps, 
                        full_obs1=full_obs1, full_obs2=full_obs2, obs_vec_1=obs_vec_1, obs_vec_2=obs_vec_2, 
                        agent_reward_grid_1=agent_reward_grid_1, agent_reward_grid_2=agent_reward_grid_2, agent_a_pos=agent_a_pos, agent_b_pos=agent_b_pos, total_reward=total_reward, last_non_stationary_move=last_non_stationary_move,
                        both_agent_coord_before_stationary=both_agent_coord_before_stationary, agent_1_reward_box_seen=0, agent_2_reward_box_seen=0,
                        agent_1_low_reward_seen=False, agent_2_low_reward_seen=False, agent_1_mid_reward_seen=False,
                        agent_2_mid_reward_seen=False, agent_1_high_reward_seen=False, agent_2_high_reward_seen=False,
                        agent_other_pos_grid_1=agent_other_pos_grid_1, agent_other_pos_grid_2=agent_other_pos_grid_2,
                        agent_0_last_non_stationary_move=agent_0_last_non_stationary_move, agent_1_last_non_stationary_move=agent_1_last_non_stationary_move, logging_reward=logging_reward), reward

    def terminal(self, state: MirrorState) -> bool:
        return state.terminal_cond

    @property
    def name(self) -> str:
        """Environment Name """
        return "Mirror World"
    
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
