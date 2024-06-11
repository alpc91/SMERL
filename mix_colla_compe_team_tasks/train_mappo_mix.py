import copy
import functools
import os, time
import pprint
import jax

from absl import app
from absl import flags
from brax.v1.io import html
from brax.v1.io import model
from brax.v1.experimental.braxlines import experiments
from brax.v1.experimental.braxlines.common import logger_utils
from datetime import datetime
import matplotlib.pyplot as plt

from algo import mappo_mix
from procedural_envs import composer
from procedural_envs.misc.observers import GraphObserver, EXTRA_ROOT_NODE_DICT
from procedural_envs.tasks.observation_config import obs_config_dict
# from brax.v1.experimental.braxlines.common import evaluators
from procedural_envs.misc import evaluators

import random
import numpy as np

FLAGS = flags.FLAGS
flags.DEFINE_string('env', 'ant_reach_4', 'Name of environment to train.')
flags.DEFINE_string('obs_config', 'amorpheus', 'Name of observation config to train.')
flags.DEFINE_integer('total_env_steps', 100000000,
                     'Number of env steps to run training for.')
flags.DEFINE_integer('eval_frequency', 20, 'How many times to run an eval.')
flags.DEFINE_integer('seed', 0, 'Random seed.')
flags.DEFINE_integer('num_envs', 2048, 'Number of envs to run in parallel.')
flags.DEFINE_integer('action_repeat', 1, 'Action repeat.')
flags.DEFINE_integer('batch_size', 128, 'Batch size.')
flags.DEFINE_float('reward_scaling', 1.0, 'Reward scale.')
flags.DEFINE_integer('episode_length', 1000, 'Episode length.')
flags.DEFINE_float('entropy_cost', 1e-2, 'Entropy cost.')
flags.DEFINE_integer('unroll_length', 5, 'Unroll length.')
flags.DEFINE_float('discounting', 0.97, 'Discounting.')
flags.DEFINE_float('learning_rate', 3e-4, 'Learning rate.')
flags.DEFINE_integer('num_minibatches', 32, 'Number')
flags.DEFINE_integer('num_update_epochs', 4,
                     'Number of times to reuse each transition for gradient '
                     'computation.')
flags.DEFINE_string('logdir', '', 'Logdir.')
flags.DEFINE_bool('normalize_observations', True, #env symmetry True, else False
                  'Whether to apply observation normalization.')
flags.DEFINE_integer('max_devices_per_host', None,
                     'Maximum number of devices to use per host. If None, '
                     'defaults to use as much as it can.')
flags.DEFINE_integer('num_save_html', 3, 'Number of Videos.')
flags.DEFINE_string('setname', '', 'set name.')
flags.DEFINE_string('modelname1', '', 'which model.')
flags.DEFINE_string('modelname2', '', 'which model.')


def main(unused_argv):
  np.random.seed(FLAGS.seed)
  random.seed(FLAGS.seed)
  # save dir 
  output_dir = os.path.join(
    FLAGS.logdir,
    f'{FLAGS.setname}_{FLAGS.env}_{datetime.now().strftime("%Y%m%d_%H%M%S")}')
  print(f'Saving outputs to {output_dir}')
  os.makedirs(output_dir, exist_ok=True)

  environment_params = {
      'env_name': FLAGS.env,
      'obs_config': FLAGS.obs_config,
  }
  obs_config = obs_config_dict[FLAGS.obs_config]

  if ('handsup2' in FLAGS.env) and ('ant' in FLAGS.env):
    obs_config['extra_root_node'] = EXTRA_ROOT_NODE_DICT['ant_handsup2']
  elif ('handsup2' in FLAGS.env) and ('centipede' in FLAGS.env):
    obs_config['extra_root_node'] = EXTRA_ROOT_NODE_DICT['centipede_handsup2']
  elif ('handsup' in FLAGS.env) and ('unimal' in FLAGS.env):
    obs_config['extra_root_node'] = EXTRA_ROOT_NODE_DICT['unimal_handsup']
  elif 'handsup' in FLAGS.env:
    obs_config['extra_root_node'] = EXTRA_ROOT_NODE_DICT['handsup']

  env_config = copy.deepcopy(environment_params)
  observer = GraphObserver(name=FLAGS.obs_config, **obs_config)
  observer2 = GraphObserver(name=FLAGS.obs_config, **obs_config)
  # create env
  env_fn = composer.create_fn(env_name=FLAGS.env, observer=observer, observer2=observer2)
  env = env_fn()


# #################model test#####################
#   from collections import OrderedDict as odict
#   from brax.v1.experimental import normalization
#   from brax.training import distribution
#   from models.SGNN_O_MLP_bipair import ModelTest

#   jax.config.update("jax_enable_x64", True) #add 2024.1.1
#   torso = [env.torso,env.torso2]
#   obj_id = [env.obj_id,env.obj_id2]
#   local_state_size =  [env.local_state_size,env.local_state_size2]
#   action_shapes = env.group_action_shapes
#   obs_size=env.observation_size
#   num_node = [env.num_node, env.num_node2]
#   ctl_num = [env.ctl_num, env.ctl_num2]
#   edge_index = [env.edge_index, env.edge_index2]
#   edge_index_inner = [env.edge_index_inner, env.edge_index_inner2]
#   edge_type = [env.edge_type, env.edge_type2]

#   normalizer_params, obs_normalizer_update_fn, obs_normalizer_apply_fn = [None] * len(action_shapes), [None] * len(action_shapes), [None] * len(action_shapes)



#   agents = odict()
#   policy_params = [None] * len(action_shapes)
#   value_params = [None] * len(action_shapes)

#   local_device_count = jax.local_device_count()
#   local_devices_to_use = local_device_count

#   key = jax.random.PRNGKey(30)
#   key, key_models, key_env, key_eval = jax.random.split(key, 4)
  

#   for i, (k, action_shape) in enumerate(action_shapes.items()):
#     normalizer_params[i], obs_normalizer_update_fn[i], obs_normalizer_apply_fn[i] = (
#     normalization.create_observation_normalizer(
#         obs_size[i],
#         num_leading_batch_dims=2,
#         pmap_to_devices=local_devices_to_use))
#     parametric_action_distribution = distribution.NormalTanhDistribution(
#         event_size=action_shape['size'])

#     mt = ModelTest()
#     mt.test_sgnn(local_state_size=local_state_size[i], torso = torso[i], edge_index = edge_index[i], edge_type = edge_type[i], num_node = num_node[i], ctl_num = ctl_num[i], obj_id = obj_id[i], policy_params_size = parametric_action_distribution.param_size, edge_index_inner = edge_index_inner[i])

#   print("Model Test Ok")
# ################model test#####################



  # logging
  logger_utils.save_config(
      f'{output_dir}/obs_config.txt', env_config, verbose=True)

  train_job_params = {
    'modelname1': FLAGS.modelname1,
    'modelname2': FLAGS.modelname2,
    'action_repeat': FLAGS.action_repeat,
    'batch_size': FLAGS.batch_size,
    'checkpoint_logdir': output_dir,
    'discounting': FLAGS.discounting,
    'entropy_cost': FLAGS.entropy_cost,
    'episode_length': FLAGS.episode_length,
    'learning_rate': FLAGS.learning_rate,
    'log_frequency': FLAGS.eval_frequency,
    # 'local_state_size': env_config['observation_size'] // env.num_node,
    'normalize_observations': FLAGS.normalize_observations,
    'num_envs': FLAGS.num_envs,
    'num_minibatches': FLAGS.num_minibatches,
    'num_timesteps': FLAGS.total_env_steps,
    'num_update_epochs': FLAGS.num_update_epochs,
    'max_devices_per_host': FLAGS.max_devices_per_host,
    'reward_scaling': FLAGS.reward_scaling,
    'seed': FLAGS.seed,
    'unroll_length': FLAGS.unroll_length,
    'goal_env': env.metadata.goal_based_task}

  config = copy.deepcopy(train_job_params)
  config['env'] = FLAGS.env
  pprint.pprint(config)

  # logging
  logger_utils.save_config(
      f'{output_dir}/config.txt', config, verbose=True)
  tab = logger_utils.Tabulator(
      output_path=f'{output_dir}/training_curves.csv', append=False)

  times = [datetime.now()]
  plotpatterns = []

  progress, _, _, _ = experiments.get_progress_fn(
      plotpatterns,
      times,
      tab=tab,
      max_ncols=5,
      xlim=[0, train_job_params['num_timesteps']],
      post_plot_fn=functools.partial(plt.savefig, f'{output_dir}/progress.png'))
  
  jit_env_reset = jax.jit(env.reset)
  state = jit_env_reset(rng=jax.random.PRNGKey(seed=FLAGS.seed))
  html_path = os.path.join(output_dir, 'show.html')
  html.save_html(html_path, env.sys, [state.qp])
  print(f'Saved {html_path}')
  # jit_env_reset = jax.jit(env.reset)
  # state = jit_env_reset(rng=jax.random.PRNGKey(seed=FLAGS.seed+int(time.time())))
  # html_path = os.path.join(output_dir, 'show1.html')
  # html.save_html(html_path, env.sys, [state.qp])
  # print(f'Saved {html_path}')

  inference_fn, params, _ = mappo_mix.train(
      environment_fn=env_fn,
      progress_fn=progress,
      **train_job_params)

  # Save to flax serialized checkpoint.
  filename = f'{FLAGS.setname}_{FLAGS.env}_final.pkl'
  path = os.path.join(output_dir, filename)
  model.save_params(path, params)

  for i in range(FLAGS.num_save_html):
    evaluators.visualize_env(
            env_fn=env_fn,
            inference_fn=inference_fn,
            params=params,
            batch_size=1,
            seed=i,
            output_path=output_dir,
            output_name=f'trajectory_{i}',
            verbose=True)


if __name__ == '__main__':
  app.run(main)
