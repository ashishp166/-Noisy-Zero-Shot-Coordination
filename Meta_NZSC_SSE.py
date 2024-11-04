"""
Based on PureJaxRL Implementation of PPO
"""

import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np
import optax
from flax.linen.initializers import constant, orthogonal
from typing import Sequence, NamedTuple, Any, Dict
from flax.training.train_state import TrainState
import distrax
from baselines import LogWrapper
import jaxmarl
import wandb
import functools
import matplotlib.pyplot as plt
import hydra
from omegaconf import OmegaConf
from registration import make
import os
from jax import random
import pickle
from jax import lax

class ScannedLSTM(nn.Module):
    @functools.partial(
        nn.scan,
        variable_broadcast="params",
        in_axes=0,
        out_axes=0,
        split_rngs={"params": False},
    )
    @nn.compact
    def __call__(self, cell_state, x):
        (inputs, resets) = x
        lstm_carry, lstm_hidden = self.initialize_carry(inputs.shape[0], inputs.shape[1])
        lstm_state = (
            jnp.where(resets[:, np.newaxis], lstm_carry, cell_state[0]),
            jnp.where(resets[:, np.newaxis], lstm_hidden, cell_state[1])
        )
        (lstm_carry, lstm_hidden), y = nn.OptimizedLSTMCell(features=inputs.shape[1])(lstm_state, inputs)
        return (lstm_carry, lstm_hidden), y

    @staticmethod
    def initialize_carry(batch_size, hidden_size):
        cell = nn.OptimizedLSTMCell(features=hidden_size)
        return cell.initialize_carry(jax.random.PRNGKey(0), (batch_size, hidden_size))


class ActorCriticLSTM(nn.Module):
    action_dim: Sequence[int]
    config: Dict

    @nn.compact
    def __call__(self, carry, hidden, x):
        obs, dones, avail_actions = x
        embedding = nn.Dense(self.config["FC_DIM_SIZE"], kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(obs)
        embedding = nn.relu(embedding)

        lstm_in = (embedding, dones)
        (carry, hidden), embedding = ScannedLSTM()((carry, hidden), lstm_in)

        actor_mean = nn.Dense(self.config["LSTM_HIDDEN_DIM"], kernel_init=orthogonal(2), bias_init=constant(0.0))(embedding)
        actor_mean = nn.relu(actor_mean)
        actor_mean = nn.Dense(self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0))(actor_mean)

        unavail_actions = 1 - avail_actions
        action_logits = actor_mean - (unavail_actions * 1e10)
        pi = distrax.Categorical(logits=action_logits)

        critic = nn.Dense(self.config["FC_DIM_SIZE"], kernel_init=orthogonal(2), bias_init=constant(0.0))(embedding)
        critic = nn.relu(critic)
        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(critic)

        return carry, hidden, pi, jnp.squeeze(critic, axis=-1)


class Transition(NamedTuple):
    global_done: jnp.ndarray
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    info: jnp.ndarray
    avail_actions: jnp.ndarray


def batchify(x: dict, agent_list, num_actors):
    return jnp.stack([x[a] for a in agent_list]).reshape((num_actors, -1))


def unbatchify(x: jnp.ndarray, agent_list, num_envs, num_actors):
    x = x.reshape((num_actors, num_envs, -1))
    return {a: x[i] for i, a in enumerate(agent_list)}


def population_entropy(population_probs):
    mean_probs = jnp.mean(population_probs, axis=0)
    return -jnp.sum(mean_probs * jnp.log(mean_probs + 1e-8), axis=-1)


def make_train(config):
    env = make(config["ENV_NAME"], **config["ENV_KWARGS"])
    config["NUM_ACTORS"] = env.num_agents * config["NUM_ENVS"]
    config["NUM_UPDATES"] = config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    config["MINIBATCH_SIZE"] = config["NUM_ACTORS"] * config["NUM_STEPS"] // config["NUM_MINIBATCHES"]
    population_size = len(config["TRAINED_SEEDS"]) + 1
    env = LogWrapper(env)

    def linear_schedule(count):
        frac = 1.0 - (count // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"])) / config["NUM_UPDATES"]
        return config["LR"] * frac

    def train(rng):
        network = ActorCriticLSTM(env.action_space(env.agents[0]).n, config=config)
        rng, _rng = jax.random.split(rng)
        (init_cstate, init_hstate) = ScannedLSTM.initialize_carry(config["NUM_ENVS"], config["LSTM_HIDDEN_DIM"])

        partner_trained_agent_states = []
        train_states = []
        for i in range(population_size - 1):
            seed_num = config["TRAINED_SEEDS"][i]
            agent_weight_path = os.path.join(
                f"SelfPlay_SSE_",
                f"agent_{seed_num}_0_param_weights.npz"
            )
            with open(agent_weight_path, "rb") as f:
                loaded_params = jnp.load(f, allow_pickle=True)
            partner_train_state = TrainState.create(
                apply_fn=network.apply, params=loaded_params.item(), tx=optax.adam(1e-4)
            )
            partner_trained_agent_states.append(partner_train_state.params)
            train_state = TrainState.create(
                apply_fn=network.apply, params=loaded_params.item(), tx=optax.adam(1e-4)
            )
            train_states.append(train_state)

        tx = optax.chain(
            optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
            optax.adam(learning_rate=linear_schedule if config["ANNEAL_LR"] else config["LR"], eps=1e-5),
        )

        robust_seed_num = config["ROBUST_SEED"]
        agent_weight_path = os.path.join(
            f"SelfPlay_SSE_",
            f"agent_{robust_seed_num}_0_param_weights.npz"
        )
        with open(agent_weight_path, "rb") as f:
            loaded_params = jnp.load(f, allow_pickle=True)

        train_state = TrainState.create(apply_fn=network.apply, params=loaded_params.item(), tx=tx)
        partner_train_state = TrainState.create(
            apply_fn=network.apply, params=loaded_params.item(), tx=optax.adam(1e-4)
        )
        partner_trained_agent_states.append(partner_train_state.params)

        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, config["NUM_ENVS"])
        obsv, env_state = jax.vmap(env.reset, in_axes=(0,))(reset_rng)
        (init_cstate, init_hstate) = ScannedLSTM.initialize_carry(
            config["NUM_ACTORS"], config["LSTM_HIDDEN_DIM"]
        )
        (partner_c_state, partner_h_state) = ScannedLSTM.initialize_carry(
            config["NUM_ACTORS"], config["LSTM_HIDDEN_DIM"]
        )

        historical_performance = jnp.ones(population_size)

        def _update_step(update_runner_state, unused):
            runner_state, update_steps = update_runner_state
            (
                train_state,
                env_state,
                last_obs,
                last_done,
                cstate,
                hstate,
                rng,
                historical_performance,
            ) = runner_state

            sampling_probs = (
                jnp.power(1.0 / historical_performance, config["PRIORITIZATION_BETA"])
                / jnp.sum(jnp.power(1.0 / historical_performance, config["PRIORITIZATION_BETA"]))
            )
            rng, subkey = jax.random.split(rng)
            partner_idx = jax.random.choice(subkey, jnp.arange(population_size), p=sampling_probs)

            def fun(x, n):
                return jax.lax.switch(n, [lambda xi=xi: xi for xi in x])

            def _env_step(runner_state, unused):
                (
                    partner_train_state,
                    train_state,
                    env_state,
                    last_obs,
                    last_done,
                    cstate,
                    hstate,
                    partner_c_state,
                    partner_h_state,
                    partner_idx,
                    rng,
                ) = runner_state

                rng, _rng = jax.random.split(rng)
                avail_actions = jax.vmap(env.get_pos_moves)(env_state.env_state)
                avail_actions = jax.lax.stop_gradient(
                    batchify(avail_actions, env.agents, config["NUM_ACTORS"])
                )

                def process_single_pos(agent_a_pos_i, agent_b_pos_i, agent_1_view_i, agent_2_view, last_obs_agent_0_i, last_obs_agent_1_i):
                    last_obs_agent_0_i = last_obs_agent_0_i.at[:, 12:13].set(agent_2_view)
                    last_obs_agent_1_i = last_obs_agent_1_i.at[:, 11:12].set(agent_2_view)

                    curr_agent_a_pos = (agent_a_pos_i[0][0], agent_a_pos_i[0][1])
                    curr_agent_b_pos = (agent_b_pos_i[0][0], agent_b_pos_i[0][1])
                    agent_2_in_view = (-(curr_agent_b_pos[0] - agent_2_view + 1) > curr_agent_a_pos[0])
                    
                    x = jax.lax.select(agent_2_in_view, curr_agent_b_pos[0], -1)
                    y = jax.lax.select(agent_2_in_view, curr_agent_b_pos[1], -1)
                    agent_a_pos_b_pov = jnp.array((x, y)).reshape(1, -1)
                    
                    last_obs_agent_1_i = jax.lax.select(
                        agent_a_pos_b_pov[0][0] == -1,
                        last_obs_agent_1_i.at[:, 1:2].set(jnp.array(-1).reshape(1, 1)),
                        last_obs_agent_1_i
                    )
                    last_obs_agent_1_i = last_obs_agent_1_i.at[:, 7:9].set(agent_a_pos_b_pov)
                    return last_obs_agent_0_i, last_obs_agent_1_i

                vectorized_process = jax.vmap(process_single_pos, in_axes=(0, 0, 0, None, 0, 0))

                def parallel_process(env_state, config, last_obs, partner_idx):
                    agent_a_pos = env_state.env_state.agent_a_pos
                    agent_b_pos = env_state.env_state.agent_b_pos
                    agent_1_view = env_state.env_state.agent_1_view
                    partner_agent_view_sizes = jnp.array(config['partner_agent_view_sizes'])
                    agent_2_view = partner_agent_view_sizes[partner_idx] + 1

                    last_obs_agent_0, last_obs_agent_1 = vectorized_process(
                        agent_a_pos, agent_b_pos, agent_1_view, agent_2_view, last_obs['agent_0'], last_obs['agent_1']
                    )
                    last_obs['agent_0'] = last_obs_agent_0
                    last_obs['agent_1'] = last_obs_agent_1
                    return last_obs
                last_obs = parallel_process(env_state, config, last_obs, partner_idx)

                obs_batch = batchify(last_obs, env.agents, config["NUM_ACTORS"])
                ac_in = (obs_batch[np.newaxis, :], last_done[np.newaxis, :], avail_actions[np.newaxis, :])

                partner_cstate, partner_hstate, partner_pi, _ = network.apply(
                    partner_train_state, partner_c_state, partner_h_state, ac_in
                )
                partner_ac = partner_pi.sample(seed=_rng)
                partner_actions = unbatchify(
                    partner_ac, env.agents, config["NUM_ENVS"], env.num_agents
                )

                cstate, hstate, pi, value = network.apply(
                    train_state.params, cstate, hstate, ac_in
                )
                action = pi.sample(seed=_rng)
                log_prob = pi.log_prob(action)
                self_env_act = unbatchify(
                    action, env.agents, config["NUM_ENVS"], env.num_agents
                )

                env_act = {}
                env_act[env.agents[0]] = self_env_act[env.agents[0]]
                env_act[env.agents[1]] = partner_actions[env.agents[1]]


                rng, _rng = jax.random.split(rng)
                rng_step = jax.random.split(_rng, config["NUM_ENVS"])
                obsv, env_state, reward, done, info = jax.vmap(
                    env.step, in_axes=(0, 0, 0)
                )(rng_step, env_state, env_act)

                done_batch = batchify(done, env.agents, config["NUM_ACTORS"]).squeeze()
                transition = Transition(
                    jnp.tile(done["__all__"], env.num_agents),
                    last_done,
                    action.squeeze(),
                    value.squeeze(),
                    batchify(reward, env.agents, config["NUM_ACTORS"]).squeeze(),
                    log_prob.squeeze(),
                    obs_batch,
                    info,
                    avail_actions,
                )
                runner_state = (
                    partner_train_state,
                    train_state,
                    env_state,
                    obsv,
                    done_batch,
                    cstate,
                    hstate,
                    partner_c_state,
                    partner_h_state,
                    partner_idx,
                    rng,
                )
                return runner_state, transition

            initial_hstate = runner_state[-3]
            initial_cstate = runner_state[-4]

            partner_train_state = fun(partner_trained_agent_states, partner_idx)
            runner_state = (
                partner_train_state,
                train_state,
                env_state,
                last_obs,
                last_done,
                cstate,
                hstate,
                partner_c_state,
                partner_h_state,
                partner_idx,
                rng,
            )

            runner_state, traj_batch = jax.lax.scan(
                _env_step, runner_state, None, config["NUM_STEPS"]
            )
            (
                partner_train_state,
                train_state,
                env_state,
                last_obs,
                last_done,
                cstate,
                hstate,
                _,
                _,
                partner_idx,
                rng,
            ) = runner_state
            def process_single_pos(agent_a_pos_i, agent_b_pos_i, agent_1_view_i, agent_2_view, last_obs_agent_0_i, last_obs_agent_1_i):
                last_obs_agent_0_i = last_obs_agent_0_i.at[:, 12:13].set(agent_2_view)
                last_obs_agent_1_i = last_obs_agent_1_i.at[:, 11:12].set(agent_2_view)

                curr_agent_a_pos = (agent_a_pos_i[0][0], agent_a_pos_i[0][1])
                curr_agent_b_pos = (agent_b_pos_i[0][0], agent_b_pos_i[0][1])
                agent_2_in_view = (-(curr_agent_b_pos[0] - agent_2_view + 1) > curr_agent_a_pos[0])
                
                x = jax.lax.select(agent_2_in_view, curr_agent_b_pos[0], -1)
                y = jax.lax.select(agent_2_in_view, curr_agent_b_pos[1], -1)
                agent_a_pos_b_pov = jnp.array((x, y)).reshape(1, -1)
                
                last_obs_agent_1_i = jax.lax.select(
                    agent_a_pos_b_pov[0][0] == -1,
                    last_obs_agent_1_i.at[:, 1:2].set(jnp.array(-1).reshape(1, 1)),
                    last_obs_agent_1_i
                )
                last_obs_agent_1_i = last_obs_agent_1_i.at[:, 7:9].set(agent_a_pos_b_pov)
                return last_obs_agent_0_i, last_obs_agent_1_i

            def parallel_process(env_state, config, last_obs, partner_idx):
                agent_a_pos = env_state.env_state.agent_a_pos
                agent_b_pos = env_state.env_state.agent_b_pos
                agent_1_view = env_state.env_state.agent_1_view
                partner_agent_view_sizes = jnp.array(config['partner_agent_view_sizes'])
                agent_2_view = partner_agent_view_sizes[partner_idx] + 1
                vectorized_process = jax.vmap(process_single_pos, in_axes=(0, 0, 0, None, 0, 0))
                last_obs_agent_0, last_obs_agent_1 = vectorized_process(
                    agent_a_pos, agent_b_pos, agent_1_view, agent_2_view, last_obs['agent_0'], last_obs['agent_1']
                )
                last_obs['agent_0'] = last_obs_agent_0
                last_obs['agent_1'] = last_obs_agent_1
                return last_obs

            last_obs = parallel_process(env_state, config, last_obs, partner_idx)
            last_obs_batch = batchify(last_obs, env.agents, config["NUM_ACTORS"])
            avail_actions = jnp.ones((config["NUM_ACTORS"], env.action_space(env.agents[0]).n))
            ac_in = (last_obs_batch[np.newaxis, :], last_done[np.newaxis, :], avail_actions)

            _, _, _, last_val = network.apply(train_state.params, cstate, hstate, ac_in)
            last_val = last_val.squeeze()

            def _calculate_gae(traj_batch, last_val):
                def _get_advantages(gae_and_next_value, transition):
                    gae, next_value = gae_and_next_value
                    done, value, reward = (
                        transition.global_done,
                        transition.value,
                        transition.reward,
                    )
                    delta = reward + config["GAMMA"] * next_value * (1 - done) - value
                    gae = delta + config["GAMMA"] * config["GAE_LAMBDA"] * (1 - done) * gae
                    return (gae, value), gae

                _, advantages = jax.lax.scan(
                    _get_advantages,
                    (jnp.zeros_like(last_val), last_val),
                    traj_batch,
                    reverse=True,
                    unroll=16,
                )
                return advantages, advantages + traj_batch.value

            advantages, targets = _calculate_gae(traj_batch, last_val)

            def _update_epoch(update_state, unused):
                def _update_minibatch(train_state, batch_info):
                    init_cstate, init_hstate, traj_batch, advantages, targets = batch_info

                    def _loss_fn(params, init_cstate, init_hstate, traj_batch, gae, targets):
                        _, _, pi, value = network.apply(
                            params,
                            init_cstate.squeeze(),
                            init_hstate.squeeze(),
                            (traj_batch.obs, traj_batch.done, traj_batch.avail_actions),
                        )
                        log_prob = pi.log_prob(traj_batch.action)
                        value_pred_clipped = traj_batch.value + (value - traj_batch.value).clip(-config["CLIP_EPS"], config["CLIP_EPS"])

                        value_losses = jnp.square(value - targets)
                        value_losses_clipped = jnp.square(value_pred_clipped - targets)
                        value_loss = 0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()
                        logratio = log_prob - traj_batch.log_prob
                        ratio = jnp.exp(logratio)
                        gae = (gae - gae.mean()) / (gae.std() + 1e-8)

                        loss_actor1 = ratio * gae
                        loss_actor2 = jnp.clip(ratio, 1.0 - config["CLIP_EPS"], 1.0 + config["CLIP_EPS"]) * gae
                        loss_actor = -jnp.minimum(loss_actor1, loss_actor2).mean()

                        entropy = pi.entropy().mean()
                        approx_kl = ((ratio - 1) - logratio).mean()
                        clip_frac = jnp.mean(jnp.abs(ratio - 1) > config["CLIP_EPS"])

                        population_probs = []
                        for p in [ts.params for ts in train_states]:
                            _, _, pi, _ = network.apply(
                                p,
                                init_cstate.squeeze(),
                                init_hstate.squeeze(),
                                (traj_batch.obs, traj_batch.done, traj_batch.avail_actions),
                            )
                            population_probs.append(pi.probs)

                        population_probs = jnp.stack(population_probs)
                        pop_entropy = population_entropy(population_probs).mean()

                        total_loss = (
                            loss_actor + config["VF_COEF"] * value_loss - config["ENT_COEF"] * entropy + config["MEP_ENT_COEF"] * pop_entropy
                        )
                        return total_loss, (value_loss, loss_actor, entropy, ratio, approx_kl, clip_frac)

                    grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
                    (loss, aux), grads = grad_fn(train_state.params, init_cstate, init_hstate, traj_batch, advantages, targets)

                    def apply_grads_if_needed(approx_kl, target_kl, train_state, grads):
                        apply_grads = approx_kl < target_kl
                        return jax.lax.cond(apply_grads,
                            lambda _: train_state.apply_gradients(grads=grads),
                            lambda _: train_state,
                            operand=None)
                    train_state = apply_grads_if_needed(aux[4], config["TARGET_KL"], train_state, grads)

                    approx_kl_updated = jax.lax.select(aux[4] < config["TARGET_KL"], aux[4], 0*aux[4])
                    aux = aux[:4] + (approx_kl_updated,) + aux[5:]
                    total_loss = loss, aux
                    return train_state, total_loss

                (
                    train_state,
                    init_cstate,
                    init_hstate,
                    traj_batch,
                    advantages,
                    targets,
                    rng,
                ) = update_state
                rng, _rng = jax.random.split(rng)

                init_hstate = init_hstate.reshape((1, config["NUM_ACTORS"], -1))
                init_cstate = init_cstate.reshape((1, config["NUM_ACTORS"], -1))

                batch = (
                    init_cstate,
                    init_hstate,
                    traj_batch,
                    advantages.squeeze(),
                    targets.squeeze(),
                )
                permutation = jax.random.permutation(_rng, config["NUM_ACTORS"])
                shuffled_batch = jax.tree_util.tree_map(
                    lambda x: jnp.take(x, permutation, axis=1), batch
                )
                minibatches = jax.tree_util.tree_map(
                    lambda x: jnp.swapaxes(
                        x.reshape((x.shape[0], config["NUM_MINIBATCHES"], -1) + x.shape[2:]), 1, 0
                    ),
                    shuffled_batch,
                )

                train_state, total_loss = jax.lax.scan(
                    _update_minibatch, train_state, minibatches
                )
                update_state = (
                    train_state,
                    init_cstate.squeeze(),
                    init_hstate.squeeze(),
                    traj_batch,
                    advantages,
                    targets,
                    rng,
                )
                return update_state, total_loss

            update_state = (
                train_state,
                initial_cstate,
                initial_hstate,
                traj_batch,
                advantages,
                targets,
                rng,
            )
            update_state, loss_info = jax.lax.scan(
                _update_epoch, update_state, None, config["UPDATE_EPOCHS"]
            )

            train_state = update_state[0]
            metric = traj_batch.info
            rng = update_state[-1]
            ratio_0 = loss_info[1][3][0, 0].mean()
            loss_info = jax.tree_map(jnp.mean, loss_info)

            metric["loss"] = {
                "total_loss": loss_info[0],
                "value_loss": loss_info[1][0],
                "actor_loss": loss_info[1][1],
                "entropy": loss_info[1][2],
                "ratio": loss_info[1][3],
                "ratio_0": ratio_0,
                "approx_kl": loss_info[1][4],
                "clip_frac": loss_info[1][5],
            }
            metric["loss_info"] = loss_info

            def callback(metric):
                num_timesteps = len(metric["agent_0"]["reward"])
                wandb.log(
                    {
                        "returns": metric["returned_episode_returns"][-1, :].mean(),
                        "env_step": metric["update_steps"] * config["NUM_ENVS"] * config["NUM_STEPS"],
                        "raw_rewards": metric["agent_0"]["reward"][:, :].mean(),
                        "rewards_delta": metric["agent_0"]["reward_delta"][:, :].mean(),
                        "non_coord_action": metric["agent_0"]["non_coord"][:, :].mean(),
                        "true_action1": metric["agent_0"]["true_action1"][:, :].mean(),
                        "true_action2": metric["agent_0"]["true_action2"][:, :].mean(),
                        "max_reward": metric["agent_0"]["max_reward"][:, :].mean(),
                        "regret": metric["agent_0"]["regret"][:, :].mean(),
                        "agent_pos_coord": metric["agent_0"]["agent_pos_coord"][:, :].mean(),
                        "key_metrics/final_step_agent2_pos_x": metric["agent_0"]["agent2_pos_x"][-1, :].mean(),
                        "key_metrics/final_step_agent2_pos_y": metric["agent_0"]["agent2_pos_y"][-1, :].mean(),
                        "key_metrics/final_step_agent1_pos_x": metric["agent_0"]["agent1_pos_x"][-1, :].mean(),
                        "key_metrics/final_step_agent1_pos_y": metric["agent_0"]["agent1_pos_y"][-1, :].mean(),
                        "key_metrics/final_step_agent_pos_coord": metric["agent_0"]["agent_pos_coord"][-1, :].mean(),
                        "key_metrics/final_step_rewards_delta": metric["agent_0"]["reward_delta"][-1, :].mean(),
                        "key_metrics/final_step_regret": metric["agent_0"]["regret"][-1, :].mean(),
                        "key_metrics/final_step_reward_box_coord": metric["agent_0"]["agent_reward_box_coord"][-1, :].mean(),
                        "key_metrics/final_step_total_reward": metric["agent_0"]["total_reward"][-1, :].mean(),
                        "key_metrics/last_non_stationary_move": metric["agent_0"]["last_non_stationary_move"][-1, :].mean(),
                        "key_metrics/agent_0_last_non_stationary_move": metric["agent_0"]["agent_0_last_non_stationary_move"][-1, :].mean(),
                        "key_metrics/agent_1_last_non_stationary_move": metric["agent_0"]["agent_1_last_non_stationary_move"][-1, :].mean(),
                        "key_metrics/both_agent_coord_before_stationary": metric["agent_0"]["both_agent_coord_before_stationary"][-1, :].mean(),

                        "loss/total_loss": metric["loss"]["total_loss"],
                        "loss/value_loss": metric["loss"]["value_loss"],
                        "loss/actor_loss": metric["loss"]["actor_loss"],
                        "loss/entropy": metric["loss"]["entropy"],
                        "loss/ratio": metric["loss"]["ratio"],
                        "loss/ratio_0": metric["loss"]["ratio_0"],
                        "loss/approx_kl": metric["loss"]["approx_kl"],
                        "loss/clip_frac": metric["loss"]["clip_frac"],

                        "agent_1_num_up_actions": jnp.sum(metric["agent_0"]["true_action1"] == 0),
                        "agent_1_num_down_actions": jnp.sum(metric["agent_0"]["true_action1"] == 1),
                        "agent_1_num_right_actions": jnp.sum(metric["agent_0"]["true_action2"] == 2),
                        "agent_1_num_left_actions": jnp.sum(metric["agent_0"]["true_action1"] == 3),
                        "agent_1_num_stay_actions": jnp.sum(metric["agent_0"]["true_action1"] == 4),
                        "agent_2_num_up_actions": jnp.sum(metric["agent_0"]["true_action2"] == 0),
                        "agent_2_num_down_actions": jnp.sum(metric["agent_0"]["true_action2"] == 1),
                        "agent_2_num_right_actions": jnp.sum(metric["agent_0"]["true_action2"] == 2),
                        "agent_2_num_left_actions": jnp.sum(metric["agent_0"]["true_action2"] == 3),
                        "agent_2_num_stay_actions": jnp.sum(metric["agent_0"]["true_action2"] == 4),
                    }
                )



            metric["update_steps"] = update_steps

            new_performance = 0.9 * historical_performance[partner_idx] + 0.1 * (
                metric["returned_episode_returns"][-1].mean() + 2 * config["ENV_KWARGS"]["num_agent_steps"]
            )
            historical_performance = historical_performance.at[partner_idx].set(
                new_performance
            )


            partner_train_state = TrainState.create(
                apply_fn=network.apply, params=train_state.params, tx=optax.adam(1e-4)
            )
            partner_trained_agent_states[-1] = train_state.params

            jax.experimental.io_callback(callback, None, metric)
            update_steps = update_steps + 1
            runner_state = (
                train_state,
                env_state,
                last_obs,
                last_done,
                cstate,
                hstate,
                rng,
                historical_performance,
            )
            return (runner_state, update_steps), (
                metric["agent_0"]["regret"][-1, :].mean(),
                metric["agent_0"]["non_coord"][-1, :].mean(),
            )

        rng, _rng = jax.random.split(rng)
        runner_state = (
            train_state,
            env_state,
            obsv,
            jnp.zeros((config["NUM_ACTORS"]), dtype=bool),
            init_cstate,
            init_hstate,
            _rng,
            historical_performance,
        )
        runner_state, (regret, non_coord) = jax.lax.scan(
            _update_step, (runner_state, 0), None, config["NUM_UPDATES"]
        )
        return {"runner_state": runner_state, "regret": regret, "non_coord": non_coord}

    return train


@hydra.main(version_base=None, config_path="config", config_name="battleShip_small")
def main(config):
    config = OmegaConf.to_container(config)
    config['ENV_KWARGS']['agent_1_view_size'] = 6
    config['TRAINED_SEEDS'] = [40, 41, 42, 43, 44, 45, 46, 47] # Agent Seeds Trained via NZSC
    config['partner_agent_view_sizes'] = [2, 2, 2, 2, 4, 4, 4, 4] # Agents View Size
    for _ in range(5):
        run_name = "NZSC_SSE_" + str(config["SEED"] + counter)

        wandb.init(
            entity=config["ENTITY"],
            project=config["PROJECT"],
            tags=["IPPO", "LSTM", config["ENV_NAME"]],
            config=config,
            mode=config["WANDB_MODE"],
            name=run_name,
        )

        rng = jax.random.PRNGKey(config["SEED"])
        train_jit = jax.jit(make_train(config), device=jax.devices()[0])
        out = train_jit(rng)
        final_train_state = out["runner_state"][0][0]
        agent_params = final_train_state.params

        agent_weight = os.path.join(
            f"NZSC_SSE_",
            f"agent_{config['SEED']}_param_weights.npz",
        )
        with open(agent_weight, "wb") as f:
            jnp.save(f, agent_params)
        wandb.finish()
        counter += 1

if __name__ == "__main__":
    main()

