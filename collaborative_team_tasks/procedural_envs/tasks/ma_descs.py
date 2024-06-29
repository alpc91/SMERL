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

"""multi-agent environments."""
import functools
import itertools
from typing import Any, Dict, Sequence, List
from brax.v1 import jumpy as jnp
from brax.v1 import math
# from brax.v1.experimental.composer import component_editor
from procedural_envs.misc import component_editor
# from brax.v1.experimental.composer import reward_functions
from brax.v1.experimental.composer.composer_utils import merge_desc
from procedural_envs.misc import reward_functions
# from brax.v1.experimental.composer.observers import SimObserver as so
from procedural_envs.misc.observers import SimObserver as so
import numpy as np
import jax
from procedural_envs.misc.random_init_functions import yaw_uniform_sampler, annulus_xy_sampler
from procedural_envs.misc.quaternion import eular2quat
from procedural_envs.misc.unimal_utils import get_end_effectors, get_agent_names
unimal_names = get_agent_names()

MAX_DIST = 20
MIN_DIST = 0.5


def get_n_agents_desc(agents: Sequence[str],
                      group: Sequence[str] = None,
                      agents_params: Sequence[str] = None,
                      init_r: float = 5):
  """Get n agents."""
  # angles_start = 0#np.random.uniform(0, 2 * np.pi)
  # angles = np.linspace(angles_start, angles_start+2 * np.pi, len(agents) + 1)
  agents_params = agents_params or ([None] * len(agents))
  components = {}
  edges = {}
  unimal_id = [20,22,20,22]
  for i, agent in enumerate(agents):
    # angle = angles[i]#np.random.uniform(0, 2 * np.pi)
    # r = init_r#np.random.uniform(0/2, init_r)
    # pos = (np.cos(angle) * r, np.sin(angle) * r, 0)
    components[f'{group[i]}_agent{i}'] = dict(component=agent, #pos=pos, quat= eular2quat((0,0,angles[i]))
              random_init='posquat',
              random_init_pos_fn=functools.partial(
                annulus_xy_sampler, r_min=0, r_max=init_r, init_z=0),
              random_init_quat_fn=yaw_uniform_sampler
              )
    if agent == 'unimal':
      component_params = dict(config_name=unimal_names[unimal_id[i]])
      components[f'{group[i]}_agent{i}'].update(dict(component_params=component_params))
    # if agents_params[i]:
    #   components[f'{group[i]}_agent{i}'].update(dict(component_params=agents_params[i]))
#   for k1, k2 in itertools.combinations(list(components), 2):
#     if k1 == k2:
#       continue
#     k1, k2 = sorted([k1, k2])  # ensure the name is always sorted in order
#     edge_name = component_editor.concat_comps(k1, k2)
#     edges[edge_name] = dict(
#         extra_observers=[dict(observer_type='root_vec', indices=(0, 1))])
  return dict(components=components, edges=edges)




def add_reach(env_desc: Dict[str, Any],
    torso_name = [],
    centering_scale: float = 5.,
    control_scale: float = 0.2,
    opp_scale: float = 0.2,  
    ring_size: float = 5.,
    win_bonus: float = 10000.,
    radius: float = 0.1):
  """Add chase task."""
  agents = sorted(env_desc['components'])
  num_agents = len(agents)
  groups = ['team0']
  agent_groups = {group: {'reward_names': ()} for group in groups}
  components = {}
  edges = {}
  

  balls = []
  for i in range(num_agents):
    balls.append(f'ball{i}')
    # angle = angles[i]#np.random.uniform(0, 2 * np.pi)
    # r = ring_size#np.random.uniform(0/2, init_r)
    # pos = (np.cos(angle) * r, np.sin(angle) * r, 0)
    components[f'ball{i}'] = dict(component='ball', #pos = pos,
              component_params=dict(
                  radius=radius,
                  frozen=True,
                  name=f"Ball_{i}"
                  ),
                  # pos = (3*(1-i*2),0,0),
                random_init='posquat',
                random_init_pos_fn=functools.partial(
                annulus_xy_sampler, r_min=0, r_max=ring_size, init_z=0),
                random_init_quat_fn=yaw_uniform_sampler,
    )

    for j, agent in enumerate(agents):
        edge_name = component_editor.concat_comps(f'ball{i}', agent)
        edges[edge_name] = dict(  ####sorted!!!
            reward_fns=dict(
                # move to goal's direction
                move_to_goal=dict(
                    reward_type=reward_functions.direction_reward,
                    vel0=lambda x, y: so('body', 'vel', y['root'], indices=(0, 1)), ####sorted!!!
                    vel1=lambda x, y: so('body', 'vel', x['root'], indices=(0, 1)),
                    pos0=lambda x, y: so('body', 'pos', y['root'], indices=(0, 1)),
                    pos1=lambda x, y: so('body', 'pos', x['root'], indices=(0, 1)),
                    scale=opp_scale,
                ),
            ))
        agent_groups['team0']['reward_names'] += (('move_to_goal', f'ball{i}', agent), )
                                            


  for agent in agents:  
    components[agent]=dict(
          reward_fns =
            dict(
                control_penalty=dict(
                    reward_type=reward_functions.control_reward,
                    scale=control_scale,
                ),
            ),
    )
    agent_groups['team0']['reward_names'] += (('control_penalty', agent),
                                              )
    
  if num_agents == 1:
      reach_reward_type = reward_functions.nearest_distance_reward_1
  elif num_agents == 2:
      reach_reward_type = reward_functions.nearest_distance_reward_2
  elif num_agents >= 3:
      reach_reward_type = reward_functions.nearest_distance_reward_3

  components['ground'] = dict(component='ground',
                      reward_fns = dict(distance = dict(
                      reward_type=reach_reward_type,
                      target=[
                          so(comp_name=f'ball{j}', sdname=f'Ball_{j}', indices=(0, 1)) for j,_ in enumerate(agents)],
                      obs=[
                          so(comp_name=f'team0_agent{j}', sdname=torso_name[j], indices=(0, 1)) for j,_ in enumerate(agents)],
                      min_dist=0,
                      done_bonus=0,
                      scale=centering_scale,),
                      win = dict(
                      reward_type=reach_reward_type,
                      target=[
                          so(comp_name=f'ball{j}', sdname=f'Ball_{j}', indices=(0, 1)) for j,_ in enumerate(agents)],
                      obs=[
                          so(comp_name=f'team0_agent{j}', sdname=torso_name[j], indices=(0, 1)) for j,_ in enumerate(agents)],
                      min_dist=radius*5,
                      done_bonus=win_bonus,
                      scale=0,)
                      ),
                )
    
  agent_groups['team0']['reward_names'] += (('distance','ground'),('win','ground'),)

  # env_desc['satisfy_all_cond'] = True
#   env_desc['goal_based_task'] = True
  
  # components.update(get_ring_components(radius=ring_size, num_segments=20))
  merge_desc(
      env_desc,
      dict(agent_groups=agent_groups, edges=edges, components=components))
  return env_desc

def get_ring_components(name: str = 'ring',
                        num_segments: int = 4,
                        radius: float = 3.0,
                        thickness: float = None,
                        collide: bool = False,
                        offset: Sequence[float] = None):
  """Draw a ring with capsules."""
  offset = offset or [0, 0, 0]
  offset = jnp.array(offset)
  thickness = thickness or radius / 40.
  components = {}
  angles = np.linspace(0, np.pi * 2, num_segments + 1)
  for i, angle in enumerate(angles[:-1]):
    k = f'{name}{i}'
    ring_length = radius * np.tan(np.pi / num_segments)
    components[k] = dict(
        component='singleton',
        component_params=dict(
            size=[thickness, ring_length * 2],
            collider_type='capsule',
            no_obs=True),
        pos=offset + jnp.array(
            (radius * np.cos(angle), radius * np.sin(angle), -ring_length)),
        quat=math.euler_to_quat(jnp.array([90, angle / jnp.pi * 180, 0])),
        quat_origin=(0, 0, ring_length),
        frozen=True,
        collide=collide)
  return components





TASK_MAP = dict(
    reach = add_reach)


def create_desc(team0_agent: List[str] = [], 
                team1_agent: List[str] = [],
                team0_agent_params: Dict[str, Any] = None,
                team1_agent_params: Dict[str, Any] = None,
                team0_num_agents: int = 1,
                team1_num_agents: int = 0,
                task: str = 'follow',
                init_r: float = 2.,
                **kwargs):
  """Creat env_desc."""
  # team0_agent_params = dict(num_legs=3)
  # other_agent_params = dict(num_legs=8)
  # if team0_agent_params or team1_agent_params:
  #   agents_params = [team0_agent_params] * team0_num_agents + [team1_agent_params] * team1_num_agents
  # else:
  #   agents_params = None
  env_desc = get_n_agents_desc(
      agents= team0_agent + team1_agent,
      group= ['team0'] * team0_num_agents + ['team1'] * team1_num_agents,
      # agents_params=agents_params,
      init_r=init_r)

  return TASK_MAP[task](env_desc=env_desc, **kwargs)


ENV_DESCS = {
             '1centipede':functools.partial(create_desc, team0_agent= ['centipede'], team0_num_agents = 1, init_r = 5, ring_size=5, torso_name = ['torso_0'], task='reach'),
             '1ant':functools.partial(create_desc, team0_agent= ['ant'], team0_num_agents = 1,  init_r = 5, ring_size=5,  torso_name = ['torso'],task='reach'),
             '2claws':functools.partial(create_desc, team0_agent= ['claw','claw'], team0_num_agents = 2,  init_r = 5, ring_size=5, torso_name = ['torso', 'torso'], task='reach'),
             '3acc':functools.partial(create_desc, team0_agent= ['ant','claw','centipede'], team0_num_agents = 3,  init_r = 5, ring_size=5, torso_name = ['torso', 'torso', 'torso_0'], task='reach'),
             '2unimals':functools.partial(create_desc, team0_agent= ['unimal']*2, team0_num_agents = 2,  init_r = 5, ring_size=5, torso_name = ['torso_0']*2, task='reach'),
             '2antclaw':functools.partial(create_desc, team0_agent= ['ant','claw'], team0_num_agents = 2,  init_r = 5, ring_size=5, torso_name = ['torso', 'torso'], task='reach'),
             '2ants':functools.partial(create_desc, team0_agent= ['ant','ant'], team0_num_agents = 2, init_r = 3, ring_size=3, torso_name = ['torso', 'torso'],  task='reach'),
             }

