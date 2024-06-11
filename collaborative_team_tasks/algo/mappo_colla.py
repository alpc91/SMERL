# Copyright 2023 The Brax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Multi-agent proximal policy optimization training.

*This is branched from brax.v1lines/training/ppo.py, and will be folded back.*
"""

from collections import OrderedDict as odict
import functools
import os
import time
from typing import Any, Callable, Dict, Optional, Tuple, List

from absl import logging
from brax.v1 import envs
from brax.v1.io import model
from brax.v1.experimental import normalization
# from brax.v1.experimental.braxlines.training import env
from procedural_envs.misc import env
from brax.v1.experimental.composer import data_utils
from brax.training import distribution
from brax.training import networks
from brax.training import pmap
from brax.training.networks import FeedForwardNetwork
from brax.training.agents.ppo import losses as ppo_losses
from brax.training.types import Params
from brax.training.types import PRNGKey
# from brax.v1.experimental.braxlines.common import evaluators
from procedural_envs.misc import evaluators
from brax.v1.io import html

import flax
import jax
import jax.numpy as jnp
import optax
from flax import linen



from models.SHNN import make_sgnn_networks as SHNN ## final model

from models.MLP import make_networks as MLP




from jax import lax
from jax.experimental import host_callback

@flax.struct.dataclass
class StepData:
  """Contains data for one environment step."""
  obs: List[jnp.ndarray]
  rewards: jnp.ndarray
  dones: jnp.ndarray
  truncation: jnp.ndarray
  actions: jnp.ndarray
  logits: jnp.ndarray


@flax.struct.dataclass
class TrainingState:
  """Contains training state for the learner."""
  optimizer_state: optax.OptState
  params: Params
  key: PRNGKey
  normalizer_params: List[Params]


@flax.struct.dataclass
class Agent:
  parametric_action_distribution: distribution.ParametricDistribution
  policy_model: Any
  optimizer_state: Any
  init_params: Any
  grad_loss: Any


def compute_ppo_loss(
    models: Dict[str, Params],
    data: StepData,
    rng: PRNGKey,
    parametric_action_distribution: distribution.ParametricDistribution,
    policy_apply: Any,
    value_apply: Any,
    # num_limb: int,
    # local_state_size: int = 19,
    entropy_cost: float = 1e-4,
    discounting: float = 0.9,
    reward_scaling: float = 1.0,
    lambda_: float = 0.95,
    ppo_epsilon: float = 0.3,
    action_shapes: Dict[str, Dict[str, Any]] = None,
    agent_name: str = None
):
  # print(agent_name)
  """Computes PPO loss."""
  policy_params, value_params = models['policy'], models['value']
  # print("data.obs.shape",data.obs[0].shape) #(5, 1024, 382)
  # print("data.obs2.shape",data.obs[1].shape) #(5, 1024, 382)
  # print("data.rewards.shape",data.rewards.shape) #(6, 1024, 2)
  # print("data.dones.shape",data.dones.shape) #(6, 1024, 2)
  # print("data.truncation.shape",data.truncation.shape) #(6, 1024, 2)
  # print("data.actions.shape",data.actions[0].shape) #(6, 1024, 2, 32)
  # print("data.actions2.shape",data.actions[1].shape) #(6, 1024, 2, 32)

  agent_index = list(action_shapes.keys()).index(agent_name)
  R, B, A = data.actions[agent_index].shape
  policy_logits = policy_apply(policy_params, data.obs[agent_index][:-1].reshape(R*B,-1)).reshape(R,B,A*2)
  # print(policy_logits.shape) #(5, 1024, 32)
  baseline = value_apply(value_params, data.obs[agent_index].reshape((R+1)*B,-1)).reshape(R+1,B,1)
  # print(baseline.shape) #(6, 1024, 1)

  baseline = jnp.squeeze(baseline, axis=-1)

  # Use last baseline value (from the value function) to bootstrap.
  bootstrap_value = baseline[-1]
  baseline = baseline[:-1]

  # At this point, we have unroll length + 1 steps. The last step is only used
  # as bootstrap value, so it's removed.

  # already removed at data generation time
  # actions = actions[:-1]
  # logits = logits[:-1]

  
  rewards = data.rewards[1:, ..., agent_index] * reward_scaling
  truncation = data.truncation[1:]#jnp.zeros_like(data.dones[1:])#
  termination = data.dones[1:] * (1 - truncation)#jnp.zeros_like(data.dones[1:])#
  actions = data.actions[agent_index]
  logits = data.logits[agent_index]

  target_action_log_probs = parametric_action_distribution.log_prob(
      policy_logits, actions)
  behaviour_action_log_probs = parametric_action_distribution.log_prob(
      logits, actions)

  vs, advantages = ppo_losses.compute_gae(
      truncation=truncation,
      termination=termination,
      rewards=rewards,
      values=baseline,
      bootstrap_value=bootstrap_value,
      lambda_=lambda_,
      discount=discounting)
  rho_s = jnp.exp(target_action_log_probs - behaviour_action_log_probs)

  surrogate_loss1 = rho_s * advantages
  surrogate_loss2 = jnp.clip(rho_s, 1 - ppo_epsilon,
                             1 + ppo_epsilon) * advantages

  policy_loss = -jnp.mean(jnp.minimum(surrogate_loss1, surrogate_loss2))

  # Value function loss
  v_error = vs - baseline
  value_loss = jnp.mean(v_error * v_error) * 0.5 * 0.5

  # Entropy reward
  entropy = jnp.mean(parametric_action_distribution.entropy(policy_logits, rng))
  entropy_loss = entropy_cost * -entropy

  total_loss = policy_loss + value_loss + entropy_loss


  return total_loss, {
          'total_loss': total_loss,
          'policy_loss': policy_loss,
          'value_loss': value_loss,
          'entropy_loss': entropy_loss,
      }


def train(environment_fn: Callable[..., envs.Env],
          num_timesteps,
          episode_length: int,
          action_repeat: int = 1,
          num_envs: int = 1,
          max_devices_per_host: Optional[int] = None,
          num_eval_envs: int = 1024,
          learning_rate=1e-4,
          entropy_cost=1e-4,
          discounting=0.9,
          seed=0,
          unroll_length=10,
          batch_size=32,
          num_minibatches=16,
          num_update_epochs=2,
          log_frequency=10,
          normalize_observations=False,
          reward_scaling=1.,
          progress_fn: Optional[Callable[[int, Dict[str, Any]], None]] = None,
          # local_state_size: int = 19,
          gradient_clipping: float = 0.1,
          checkpoint_logdir: Optional[str] = None,
          goal_env=False,
          modelname=None,
          parametric_action_distribution_fn: Optional[Callable[[
              int,
          ], distribution.ParametricDistribution]] = distribution
          .NormalTanhDistribution,
          policy_params: Optional[Dict[str, jnp.ndarray]] = None,
          value_params: Optional[Dict[str, jnp.ndarray]] = None):
  """PPO training."""
  assert batch_size * num_minibatches % num_envs == 0
  xt = time.time()

  process_count = jax.process_count()
  process_id = jax.process_index()
  local_device_count = jax.local_device_count()
  local_devices_to_use = local_device_count
  if max_devices_per_host:
    local_devices_to_use = min(local_devices_to_use, max_devices_per_host)
  logging.info(
      'Device count: %d, process count: %d (id %d), local device count: %d, '
      'devices to be used count: %d',
      jax.device_count(), process_count, process_id, local_device_count, local_devices_to_use)
  num_envs_per_device = num_envs // local_devices_to_use // process_count

  # TODO: check key randomness
  key = jax.random.PRNGKey(seed)
  key, key_models, key_env, key_eval = jax.random.split(key, 4)
  # Make sure every process gets a different random key, otherwise they will be
  # doing identical work.
  key_env = jax.random.split(key_env, process_count)[process_id]
  key = jax.random.split(key, process_count)[process_id]
  # key_models should be the same, so that models are initialized the same way
  # for different processes

  core_env = environment_fn(
      action_repeat=action_repeat,
      batch_size=num_envs_per_device,
      episode_length=episode_length,
      auto_reset = True)
  key_envs = jax.random.split(key_env, local_devices_to_use)
  tmp_env_states = []
  for key in key_envs:
    first_state, step_fn = env.wrap(
        core_env, key, extra_step_kwargs=False)
    tmp_env_states.append(first_state)
  first_state = jax.tree_util.tree_map(lambda *args: jnp.stack(args),
                                  *tmp_env_states)

  component_env = core_env.unwrapped
  action_size = component_env.action_size
  obj_id = [component_env.obj_id,component_env.obj_id2]
  torso = [component_env.torso,component_env.torso2]
  local_state_size =  [component_env.local_state_size,component_env.local_state_size2]
  action_shapes = component_env.group_action_shapes
  obs_size=component_env.observation_size
  num_node = [component_env.num_node, component_env.num_node2]
  ctl_num = [component_env.ctl_num, component_env.ctl_num2]
  edge_index = [component_env.edge_index, component_env.edge_index2]
  edge_type = [component_env.edge_type, component_env.edge_type2]
  edge_index_inner = [component_env.edge_index_inner, component_env.edge_index_inner2]
  # print("num_node",num_node)
  # print("local_state_size", local_state_size)
  # print(action_shapes)
  # print(action_size, obs_size[0], obs_size[1])


  core_eval_env = environment_fn(
      action_repeat=action_repeat,
      batch_size=num_eval_envs,
      episode_length=episode_length,
      auto_reset = False)
  eval_first_state, eval_step_fn = env.wrap(
      core_eval_env, key_eval, extra_step_kwargs=False)

  normalizer_params, obs_normalizer_update_fn, obs_normalizer_apply_fn = [None] * len(action_shapes), [None] * len(action_shapes), [None] * len(action_shapes)

  agents = odict()
  policy_params = policy_params or [None] * len(action_shapes)
  value_params = value_params or [None] * len(action_shapes)

  if modelname == "MLP":
    make_models_fn = MLP
  elif modelname == "SHNN":
    make_models_fn = SHNN


  for i, (k, action_shape) in enumerate(action_shapes.items()):
    normalizer_params[i], obs_normalizer_update_fn[i], obs_normalizer_apply_fn[i] = (
    normalization.create_observation_normalizer(
        obs_size[i],
        normalize_observations,
        num_leading_batch_dims=2,
        pmap_to_devices=local_devices_to_use))
    parametric_action_distribution = parametric_action_distribution_fn(
        event_size=action_shape['size'])


    policy_model, value_model = make_models_fn(obs_size = obs_size[i], local_state_size=local_state_size[i],  num_node = num_node[i], ctl_num = ctl_num[i], torso = torso[i], edge_index = edge_index[i],  edge_type = edge_type[i], obj_id=obj_id[i], policy_params_size = parametric_action_distribution.param_size, edge_index_inner = edge_index_inner[i])
    key_policy, key_value, key_models = jax.random.split(key_models, 3)
    init_params = {
        'policy': policy_params[i] or policy_model.init(key_policy),
        'value': value_params[i] or value_model.init(key_value),
    }

    num_params = sum(jnp.prod(jnp.array(param.shape)) for param in jax.tree_util.tree_leaves(init_params['policy']['params']))
    print(f"Total number of parameters: {num_params}")


    # optimizer = optax.adam(learning_rate=learning_rate)
    optimizer = optax.chain(
      optax.clip(gradient_clipping),
      optax.adam(learning_rate=learning_rate),
    )

    optimizer_state = optimizer.init(init_params)
    optimizer_state, init_params = pmap.bcast_local_devices(
        (optimizer_state, init_params), local_devices_to_use)

    loss_fn = functools.partial(
        compute_ppo_loss,
        parametric_action_distribution=parametric_action_distribution,
        policy_apply=policy_model.apply,
        value_apply=value_model.apply,
        entropy_cost=entropy_cost,
        discounting=discounting,
        reward_scaling=reward_scaling,
        action_shapes=action_shapes,
        agent_name=k
        )

    grad_loss = jax.grad(loss_fn, has_aux=True)
    agents[k] = Agent(parametric_action_distribution, policy_model,
                      optimizer_state, init_params, grad_loss)

  key_debug = jax.random.PRNGKey(seed + 666)

  def do_one_step_eval(carry, unused_target_t):
    state, policy_params, normalizer_params, key = carry
    key, key_sample = jax.random.split(key)
    obs = [None]*len(action_shapes)
    actions = odict()
    for i, (k, agent) in enumerate(agents.items()):
      obs[i] = obs_normalizer_apply_fn[i](normalizer_params[i], state.core.obs[i])
      logits = agent.policy_model.apply(policy_params[i], obs[i])
      actions[k] = agent.parametric_action_distribution.sample(
          logits, key_sample)
    actions_arr = jnp.zeros(obs[i].shape[:-1] + (action_size,))
    actions = data_utils.fill_array(actions, actions_arr, action_shapes)
    nstate = eval_step_fn(state, actions)
    return (nstate, policy_params, normalizer_params, key), ()

  @jax.jit
  def run_eval(state, key, policy_params, normalizer_params):
    policy_params = jax.tree_util.tree_map(lambda x: x[0], policy_params)
    normalizer_params = [jax.tree_util.tree_map(lambda x: x[0], normalizer_params[i]) for i, _ in enumerate(agents.items())]
    (state, _, _, key), _ = jax.lax.scan(
        do_one_step_eval,
        (state, policy_params, normalizer_params, key), (),
        length=episode_length // action_repeat)
    return state, key

  def do_one_step(carry, unused_target_t):
    state, normalizer_params, policy_params, key = carry
    key, key_sample = jax.random.split(key)
    logits, actions, postprocessed_actions = [], [], odict()
    for i, (k, agent) in enumerate(agents.items()):
      normalized_obs = obs_normalizer_apply_fn[i](normalizer_params[i], state.core.obs[i])
      logits += [agent.policy_model.apply(policy_params[i], normalized_obs)]
      actions += [
          agent.parametric_action_distribution.sample_no_postprocessing(
              logits[-1], key_sample)
      ]
      postprocessed_actions[
          k] = agent.parametric_action_distribution.postprocess(actions[-1])
    postprocessed_actions_arr = jnp.zeros(normalized_obs.shape[:-1] +
                                          (action_size,))
    postprocessed_actions = data_utils.fill_array(postprocessed_actions,
                                                  postprocessed_actions_arr,
                                                  action_shapes)
    nstate = step_fn(state, postprocessed_actions)
    return (nstate, normalizer_params, policy_params,
            key), StepData(
                obs=state.core.obs,
                rewards=state.core.reward,
                dones=state.core.done,
                truncation=state.core.info['truncation'],
                actions=actions,
                logits=logits)

  def generate_unroll(carry, unused_target_t):
    state, normalizer_params, policy_params, key = carry
    (state, _, _, key), data = jax.lax.scan(
        do_one_step,
        (state, normalizer_params, policy_params, key), (),
        length=unroll_length)
    data = data.replace(
        obs=[jnp.concatenate([data.obs[i],
                             jnp.expand_dims(state.core.obs[i], axis=0)]) for i, _ in enumerate(agents.items())] ,
        rewards=jnp.concatenate(
            [data.rewards,
             jnp.expand_dims(state.core.reward, axis=0)]),
        dones=jnp.concatenate(
            [data.dones, jnp.expand_dims(state.core.done, axis=0)]),
        truncation=jnp.concatenate([
            data.truncation,
            jnp.expand_dims(state.core.info['truncation'], axis=0)
        ]))
    return (state, normalizer_params, policy_params, key), data

  def update_model(carry, data):
    optimizer_state, params, key = carry
    key, key_loss = jax.random.split(key)
    metrics = []
    for i, agent in enumerate(agents.values()):
      loss_grad, agent_metrics = agent.grad_loss(params[i], data,
                                                 key_loss)
      metrics.append(agent_metrics)
      loss_grad = jax.lax.pmean(loss_grad, axis_name='i')
      params_update, optimizer_state[i] = optimizer.update(
          loss_grad, optimizer_state[i])
      params[i] = optax.apply_updates(params[i], params_update)

    return (optimizer_state, params, key), metrics

  def minimize_epoch(carry, unused_t):
    optimizer_state, params, data, key = carry
    key, key_perm, key_grad = jax.random.split(key, 3)
    permutation = jax.random.permutation(key_perm, data.dones.shape[1])

    def convert_data(data, permutation):
      data = jnp.take(data, permutation, axis=1, mode='clip')
      data = jnp.reshape(data, [data.shape[0], num_minibatches, -1] +
                         list(data.shape[2:]))
      data = jnp.swapaxes(data, 0, 1)
      return data

    ndata = jax.tree_util.tree_map(lambda x: convert_data(x, permutation), data)#（32,6,1024,456）
    (optimizer_state, params, _), metrics = jax.lax.scan(
        update_model, (optimizer_state, params, key_grad), ndata,
        length=num_minibatches)
    return (optimizer_state, params, data, key), metrics

  def get_params(state, key, value=None):
    if value is not None:
      return [params.get(key, value) for params in state.params]
    return [params[key] for params in state.params]

  def run_epoch(carry, unused_t):
    training_state, state = carry
    key_minimize, key_generate_unroll, new_key = jax.random.split(
        training_state.key, 3)
    (state, _, _, _), data = jax.lax.scan(
        generate_unroll,
        (state, training_state.normalizer_params,
         get_params(training_state, 'policy'), key_generate_unroll), (),
        length=batch_size * num_minibatches // num_envs)
    # make unroll first
    # print(data.obs[0].shape)#(16length, 6unroll_length+1, 2048num_envs, 456obs_size)
    data = jax.tree_util.tree_map(lambda x: jnp.swapaxes(x, 0, 1), data)
    # print(data.obs[0].shape)#(6, 16, 2048, 456)
    data = jax.tree_util.tree_map(
        lambda x: jnp.reshape(x, [x.shape[0], -1] + list(x.shape[3:])), data)
    # print(data.obs[0].shape)#(6, 32768, 456)


    # Update normalization params and normalize observations.
    for i, _ in enumerate(agents.items()):
      normalizer_params[i] = obs_normalizer_update_fn[i](
          training_state.normalizer_params[i], data.obs[i][:-1])
    data = data.replace(
        obs=[obs_normalizer_apply_fn[i](normalizer_params[i], data.obs[i]) for i, _ in enumerate(agents.items())] )

    (optimizer_state, params, _, _), metrics = jax.lax.scan(
        minimize_epoch, (training_state.optimizer_state, training_state.params,
                         data, key_minimize), (),
        length=num_update_epochs)

    new_training_state = TrainingState(
        optimizer_state=optimizer_state,
        params=params,
        normalizer_params=normalizer_params,
        key=new_key)
    return (new_training_state, state), metrics

  num_epochs = num_timesteps // (
      batch_size * unroll_length * num_minibatches * action_repeat)

  def _minimize_loop(training_state, state):
    synchro = pmap.is_replicated(
        (training_state.optimizer_state, training_state.params,
         training_state.normalizer_params),
        axis_name='i')
    (training_state, state), losses = jax.lax.scan(
        run_epoch, (training_state, state), (),
        length=num_epochs // log_frequency)
    losses = jax.tree_util.tree_map(jnp.mean, losses)
    return (training_state, state), losses, synchro

  minimize_loop = jax.pmap(_minimize_loop, axis_name='i')

  inference = make_inference_fn(
      obs_size=obs_size, local_state_size=local_state_size, action_shapes=action_shapes, num_node=num_node, ctl_num = ctl_num, torso=torso, obj_id=obj_id, edge_index=edge_index, edge_index_inner = edge_index_inner, edge_type=edge_type, normalize_observations=normalize_observations,
      parametric_action_distribution_fn=parametric_action_distribution_fn, make_models_fn=make_models_fn)

  training_state = TrainingState(
      optimizer_state=[agent.optimizer_state for agent in agents.values()],
      params=[agent.init_params for agent in agents.values()],
      key=jnp.stack(jax.random.split(key, local_devices_to_use)),
      normalizer_params=normalizer_params)
  training_walltime = 0
  eval_walltime = 0
  sps = 0
  eval_sps = 0
  losses = []
  state = first_state
  metrics = {}

  for it in range(log_frequency + 1):
    num_timesteps = int(training_state.normalizer_params[0][0][0]) * action_repeat
    logging.info('starting iteration %s %s', it, time.time() - xt)
    t = time.time()

    if process_id == 0:
      num_timesteps = int(training_state.normalizer_params[0][0][0]) * action_repeat
      eval_state, key_debug = (
          run_eval(eval_first_state, key_debug,
                   get_params(training_state, 'policy'),
                   training_state.normalizer_params))
      print("key_debug", key_debug)
      key_debug, key_sample = jax.random.split(key_debug)
      eval_first_state = env.wrap(
          core_eval_env, key_sample, extra_step_kwargs=False, return_step=False)
      eval_state.success_episodes.block_until_ready()
      eval_state.total_episodes.block_until_ready()
      eval_walltime += time.time() - t
      eval_sps = (
          episode_length * eval_state.total_episodes /
          (time.time() - t))
      avg_episode_length = (
          eval_state.total_steps /
          eval_state.total_episodes)
      success_rate = eval_state.success_episodes / eval_state.total_episodes
      metrics = dict(
          **dict({
              f'eval/episode_{name}': value / eval_state.total_episodes
              for name, value in eval_state.total_metrics.items()
          }),
          **dict({
              f'{index}/losses/{k}': jnp.mean(v)
              for index, loss in enumerate(losses) for k, v in loss.items()
          }),
          **dict({
              'eval/total_episodes': eval_state.total_episodes,
              'eval/avg_episode_length': avg_episode_length,
              'eval/success_rate': success_rate,
              'speed/sps': sps,
              'speed/eval_sps': eval_sps,
              'speed/training_walltime': training_walltime,
              'speed/eval_walltime': eval_walltime,
              'speed/timestamp': training_walltime,
              'num_timesteps': num_timesteps,
          }))
      logging.info(metrics)
      if progress_fn:
        params = dict(
            normalizer=jax.tree_util.tree_map(lambda x: x[0],
                                    training_state.normalizer_params),
            policy=jax.tree_util.tree_map(lambda x: x[0],
                                get_params(training_state, 'policy')))
        # if not any(jnp.isnan(value).any() for value in metrics.values()):
        progress_fn(
            int(training_state.normalizer_params[0][0][0]) * action_repeat,
            metrics, params)
        # else:
        #   print("NaN")
      
      if checkpoint_logdir:
        # Save current policy.
        normalizer_params = jax.tree_util.tree_map(lambda x: x[0],
                                        training_state.normalizer_params)
        policy_params = jax.tree_util.tree_map(lambda x: x[0],
                                    get_params(training_state, 'policy'))

        params = dict(
            normalizer=normalizer_params, policy=policy_params)
        
        # filename = f'mappo_mlp_{num_timesteps}.pkl'
        # path = os.path.join(checkpoint_logdir, filename)
        # model.save_params(path, params)

        # evaluators.visualize_env(
        #   env_fn=environment_fn,
        #   inference_fn=inference,
        #   params=params,
        #   batch_size=1,
        #   seed=int(time.time()),
        #   output_path=checkpoint_logdir,
        #   output_name='videos'+str(num_timesteps),
        #   verbose=True)

    if it == log_frequency:
      break

    t = time.time()
    previous_step = training_state.normalizer_params[0][0][0]
    # optimization
    (training_state, state), losses, synchro = minimize_loop(training_state, state)
    assert synchro[0], (it, training_state)
    jax.tree_util.tree_map(lambda x: x.block_until_ready(), losses)
    sps = ((training_state.normalizer_params[0][0][0] - previous_step) /
           (time.time() - t)) * action_repeat
    training_walltime += time.time() - t

  # To undo the pmap.
  normalizer_params = jax.tree_util.tree_map(lambda x: x[0],
                                   training_state.normalizer_params)
  policy_params = jax.tree_util.tree_map(lambda x: x[0],
                               get_params(training_state, 'policy'))

  logging.info('total steps: %s', normalizer_params[0][0] * action_repeat)

  params = dict(
      normalizer=normalizer_params, policy=policy_params)

  # if process_count > 1:
  #   # Make sure all processes stay up until the end of main.
  #   x = jnp.ones([jax.local_device_count()])
  #   x = jax.device_get(jax.pmap(lambda x: jax.lax.psum(x, 'i'), 'i')(x))
  #   assert x[0] == jax.device_count()
  pmap.synchronize_hosts()

  return (inference, params, metrics)


def make_inference_fn(
    obs_size: List[int],
    local_state_size: List[int],
    action_shapes: Dict[str, Any],
    num_node: List[int],
    ctl_num: List[int],
    torso: List[List[int]],
    obj_id: List[jnp.ndarray],
    edge_index: List[jnp.ndarray],
    edge_index_inner: List[jnp.ndarray],
    edge_type: List[jnp.ndarray],
    normalize_observations: bool = False,
    parametric_action_distribution_fn: Optional[Callable[[
        int,
    ], distribution.ParametricDistribution]] = distribution
    .NormalTanhDistribution,
    make_models_fn: Optional[Callable[
        [int, int], Tuple[networks.FeedForwardNetwork]]] = networks.make_models):
  """Creates params and inference function for the multi-agent PPO agent."""
  action_size = sum([s['size'] for s in action_shapes.values()])
  obs_normalizer_apply_fn = [None] * len(action_shapes)
  agents = odict()
  for i, (k, action_shape) in enumerate(action_shapes.items()):
    _, obs_normalizer_apply_fn[i] = normalization.make_data_and_apply_fn(
      obs_size[i], normalize_observations=normalize_observations)
    parametric_action_distribution = parametric_action_distribution_fn(
        event_size=action_shape['size'])
 
    policy_model, _ = make_models_fn(obs_size = obs_size[i], local_state_size=local_state_size[i],  num_node = num_node[i], ctl_num = ctl_num[i], torso = torso[i], edge_index = edge_index[i], edge_type = edge_type[i], obj_id=obj_id[i], policy_params_size = parametric_action_distribution.param_size, edge_index_inner = edge_index_inner[i])

    agents[k] = (parametric_action_distribution, policy_model)

  def inference_fn(params, obs, key):
    normalizer_params, policy_params = params['normalizer'], params['policy']
    
    actions = odict()
    for i, (k, (parametric_action_distribution,
                policy_model)) in enumerate(agents.items()):
      obs[i] = obs_normalizer_apply_fn[i](normalizer_params[i], obs[i])
      actions[k] = parametric_action_distribution.sample(
          policy_model.apply(policy_params[i], obs[i]), key)
    actions_arr = jnp.zeros(obs[0].shape[:-1] + (action_size,))
    actions = data_utils.fill_array(actions, actions_arr, action_shapes)
    return actions

  return inference_fn
