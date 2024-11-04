from typing import Callable, Any
from typing_extensions import Sequence

# import gym
import jax
import argparse
from jax import random
import jax.numpy as jnp
from flax import linen as nn
from flax.training import train_state
import wandb

from envs import NoisyRewardLG, SelfPlayNRLG, OtherPlayNZSC, GridEnv, MiniGridEnv, GridEnvLarge

env_id_to_cls = {
        'NoisyRewardLG': NoisyRewardLG,
        'SelfPlayNRLG': SelfPlayNRLG,
        'OtherPlay': OtherPlayNZSC,
        'GridEnv': GridEnv,
        'MiniGrid': MiniGridEnv,
        'GridEnvLarge': GridEnvLarge}

def make_env(env_id: str, env_kwargs: dict, seed: int, idx: int, capture_video: bool, run_name: str):
    def thunk():
        env = env_id_to_cls[env_id](**env_kwargs)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        if capture_video:
            if idx == 0:
                env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        print(seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env
    return thunk


def layer_init(layer: nn.Module, std: float = jnp.sqrt(2), bias_const: float = 0.0):
    layer.weight = jnp.random.normal(scale=std, shape=layer.weight.shape) # need to use a key for this I think
    layer.bias = jnp.full(shape=layer.bias.shape, fill_value=bias_const)
    return layer

class Agent(nn.Module):
    def __init__(self, envs: gym.Env, actuals_obs_dim: int):
        super().__init__()
        actual_obs_shape = (envs.single_observation_space.shape[0], actuals_obs_dim)
        self.critic = nn.Sequencial([
            layer_init(nn.Dense(64)),
            nn.tanh,
            layer_init(nn.Dense(64)),
            nn.tanh,
            layer_init(nn.Dense(1), std=1.0),
        ])
        self.actor = nn.Sequencial([
            layer_init(nn.Dense(64)),
            nn.tanh,
            layer_init(nn.Dense(64)),
            nn.tanh,
            layer_init(nn.Dense(envs.single_action_space.n), std=0.01)
        ])

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None, temp=1):
        logits = self.actor(x)
        probs = nn.softmax(logits * temp)
        if action is None:
            action = random.categorical(random.PRNGKey(0), logits, shape=(x.shape[0],))
        log_prob = jnp.log(probs[action])
        entropy = -jnp.sum(probs * jnp.log(probs), axis=1)
        return action, log_prob, entropy, self.critic(x)


def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    # Env parameters
    parser.add_argument("--env_id", type=str, default="OtherPlayLG",
        help="the id of the environment")
    parser.add_argument("--number_of_levers", type=int, default=3)
    parser.add_argument("--r_mean", type=float, default=5)
    parser.add_argument("--sigma", type=float, default=0)
    parser.add_argument("--sigma1", type=float, default=0)
    parser.add_argument("--sigma2", type=float, default=0)
    parser.add_argument("--horizon", type=int, default=1)
    parser.add_argument("--dense_reward", action='store_true')
    parser.add_argument("--dense_reward_value", type=float, default=0)
    parser.add_argument("--non_coordinating_reward", type=int, default=0)
    parser.add_argument("--disable_other_play", action='store_true')
    parser.add_argument("--include_r_mean", action='store_true')
    parser.add_argument("--include_noise_sigmas", action='store_true')
    parser.add_argument("--include_prev_reward_in_obs", action='store_true')
    parser.add_argument("--use_reward_delta", action='store_true',
        help="Use max-reward minus attained reward instead of attained reward only in obs.")
    parser.add_argument("--include_prev_acts_in_obs", action='store_true')
    parser.add_argument("--override_obs_with_zeros", action='store_true')
    parser.add_argument("--dont_resample_obs_noise", action='store_true',
        help="Do not resample obs noise at every timestep in iterated noisy lever game.")
    parser.add_argument("--dont_resample_permutation", action='store_true',
        help="Do not resample permutation at every timestep in iterated noisy lever game.")

    # wandb/general parameters
    parser.add_argument("--sweep", action='store_true',
        help="Use this when running sweep to indicate a sweep run.")
    parser.add_argument("--project", "-p", type=str, default=None,
        help="Wandb project")
    parser.add_argument("--group", '-g', type=str, default=None,
        help="Wandb group")
    parser.add_argument("--name", '-n', type=str, default=None,
        help="Wandb experiment name")
    parser.add_argument("--seed", type=int, default=random.randint(0,300),
        help="seed of the experiment")
    parser.add_argument("--torch_deterministic", action="store_true",
        help="if toggled, `torch.backends.cudnn.deterministic=False`")
    parser.add_argument("--dont_use_cuda", action='store_true', help="if toggled, cuda will not be used.")
    parser.add_argument("--wandb_project_name", type=str, default="noisyZSC",
        help="the wandb's project name")
    parser.add_argument("--wandb_entity", type=str, default="",
        help="the entity (team) of wandb's project")
    parser.add_argument("--eval_every", type=int, default=10000)
    parser.add_argument("--dont_save_policy", action='store_true',
        help="Disables saving of policy at end of training run.")

    # Algorithm specific arguments
    parser.add_argument("--tie_weights", action='store_true')
    parser.add_argument("--agent_type", type=str, default='normal',
        help="Determines agent neural network architecture.")
    parser.add_argument("--total_timesteps", type=int, default=500000,
        help="total timesteps of the experiments")
    parser.add_argument("--learning_rate", type=float, default=2.5e-4,
        help="the learning rate of the optimizer")
    parser.add_argument("--num_envs", type=int, default=1,
        help="the number of parallel game environments")
    parser.add_argument("--num_steps", type=int, default=128,
        help="the number of steps to run in each environment per policy rollout")
    parser.add_argument("--anneal_lr", action="store_true",
        help="Toggle learning rate annealing for policy and value networks")
    parser.add_argument("--gamma", type=float, default=0.99,
        help="the discount factor gamma")
    parser.add_argument("--gae_lambda", type=float, default=0.95,
        help="the lambda for the general advantage estimation")
    parser.add_argument("--num_minibatches", type=int, default=4,
        help="the number of mini_batches")
    parser.add_argument("--update_epochs", type=int, default=4,
        help="the K epochs to update the policy")
    parser.add_argument("--norm_adv", action="store_true",
        help="Toggles advantages normalization")
    parser.add_argument("--clip_coef", type=float, default=0.2,
        help="the surrogate clipping coefficient")
    parser.add_argument("--clip_vloss", action="store_true",
        help="Toggles whether or not to use a clipped loss for the value function, as per the paper.")
    parser.add_argument("--ent_coef", type=float, default=0.01,
        help="coefficient of the entropy")
    parser.add_argument("--anneal_ent_coef_every", type=int, default=None,
        help="Halves the entropy coefficient every x timesteps.")
    parser.add_argument("--vf-coef", type=float, default=0.5,
        help="coefficient of the value function")
    parser.add_argument("--max_grad_norm", type=float, default=0.5,
        help="the maximum norm for the gradient clipping")
    parser.add_argument("--target_kl", type=float, default=None,
        help="the target KL divergence threshold")
    args = parser.parse_args()
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    # fmt: on
    return args



class WandbLogger:
    def __init__(self):
        self.log_dict = {}

    def record(self, k, v):
        self.log_dict[k] = v

    def record_dict(self, d):
        self.log_dict.update(d)

    def dump(self, step):
        #table_print(self.log_dict)
        wandb.log(self.log_dict, step=step)
        self.log_dict = {}




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
