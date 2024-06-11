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
                      init_r: float = 2):
  """Get n agents."""
  # angles_start = 0#np.random.uniform(0, 2 * np.pi)
  # angles = np.linspace(angles_start, angles_start+2 * np.pi, len(agents) + 1)
  agents_params = agents_params or ([None] * len(agents))
  components = {}
  edges = {}
  unimal_id = [9,28,9,28]
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
    # if agent == 'centipede':
    #   components[f'{group[i]}_agent{i}'] = dict(component=agent, pos=((4*i)-2, 0,0)
    #           )
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


def add_sumo(
    env_desc: Dict[str, Any],
    team0_torso_name: List[str] = None,
    team1_torso_name: List[str] = None,
    centering_scale: float = 5., #5
    control_scale: float = 0.1, #0.2
    draw_scale: float = 0.,
    knocking_scale: float = 1.,
    opp_scale: float = 5.0,  #0.2
    ring_size: float = 3.,
    win_bonus: float = 1000.,
):
  """Add a sumo task."""
  agents = sorted(env_desc['components'])
  groups = ['team0', 'team1']
  agent_groups = {group: {'reward_names': ()} for group in groups}
  components = {}
  edges = {}
  team0_agents = [team0 for team0 in agents if 'team0' in team0]
  team1_agents = [team1 for team1 in agents if 'team1' in team1]

  components['center'] = dict(component='ball',
              component_params=dict(
                  radius=0.1,
                  frozen=True,
                  name=f"Ball"
                  ),
                pos = (0,0,0),
                frozen=True,
                collide=False,
    )

  components['ground'] = dict(component='ground',
                      reward_fns = dict(

                      team1_distance_reward = dict(
                      reward_type=reward_functions.sumo_distance_reward,
                      target=so(comp_name='center', sdname=f'Ball', indices=(0, 1)),
                      obs=[
                          so(comp_name=team0, sdname=team0_torso_name[i], indices=(0, 1)) for i, team0 in enumerate(team0_agents)],
                      max_dist=ring_size,
                      done_bonus=0,
                      scale=centering_scale,),

                      team1_win = dict(
                      reward_type=reward_functions.sumo_distance_reward,
                      target=so(comp_name='center', sdname=f'Ball', indices=(0, 1)),
                      obs=[
                          so(comp_name=team0, sdname=team0_torso_name[i], indices=(0, 1)) for i, team0 in enumerate(team0_agents)],
                      max_dist=ring_size,
                      done_bonus=win_bonus,
                      scale=0,),

                      team0_distance_penalty = dict(
                      reward_type=reward_functions.sumo_distance_reward,
                      target=so(comp_name='center', sdname=f'Ball', indices=(0, 1)),
                      obs=[
                          so(comp_name=team0, sdname=team0_torso_name[i], indices=(0, 1)) for i, team0 in enumerate(team0_agents)],
                      max_dist=ring_size,
                      done_bonus=0,
                      scale=-centering_scale,),

                      team0_lose = dict(
                      reward_type=reward_functions.sumo_distance_reward,
                      target=so(comp_name='center', sdname=f'Ball', indices=(0, 1)),
                      obs=[
                          so(comp_name=team0, sdname=team0_torso_name[i], indices=(0, 1)) for i, team0 in enumerate(team0_agents)],
                      max_dist=ring_size,
                      done_bonus=-win_bonus,
                      scale=0,),
                      
                      #################################

                      team0_distance_reward = dict(
                      reward_type=reward_functions.sumo_distance_reward,
                      target=so(comp_name='center', sdname=f'Ball', indices=(0, 1)),
                      obs=[
                          so(comp_name=team1, sdname=team1_torso_name[i], indices=(0, 1)) for i, team1 in enumerate(team1_agents)],
                      max_dist=ring_size,
                      done_bonus=0,
                      scale=centering_scale,),

                      team0_win = dict(
                      reward_type=reward_functions.sumo_distance_reward,
                      target=so(comp_name='center', sdname=f'Ball', indices=(0, 1)),
                      obs=[
                          so(comp_name=team1, sdname=team1_torso_name[i], indices=(0, 1)) for i, team1 in enumerate(team1_agents)],
                      max_dist=ring_size,
                      done_bonus=win_bonus,
                      scale=0,),

                      team1_distance_penalty = dict(
                      reward_type=reward_functions.sumo_distance_reward,
                      target=so(comp_name='center', sdname=f'Ball', indices=(0, 1)),
                      obs=[
                          so(comp_name=team1, sdname=team1_torso_name[i], indices=(0, 1)) for i, team1 in enumerate(team1_agents)],
                      max_dist=ring_size,
                      done_bonus=0,
                      scale=-centering_scale,),

                      team1_lose = dict(
                      reward_type=reward_functions.sumo_distance_reward,
                      target=so(comp_name='center', sdname=f'Ball', indices=(0, 1)),
                      obs=[
                          so(comp_name=team1, sdname=team1_torso_name[i], indices=(0, 1)) for i, team1 in enumerate(team1_agents)],
                      max_dist=ring_size,
                      done_bonus=-win_bonus,
                      scale=0,)
                      ),
                )
    
  agent_groups['team0']['reward_names'] += (('team0_distance_reward','ground'),('team0_win','ground'),('team0_distance_penalty','ground'),('team0_lose','ground'),)
  agent_groups['team1']['reward_names'] += (('team1_distance_reward','ground'),('team1_win','ground'),('team1_distance_penalty','ground'),('team1_lose','ground'),)


  for team0 in team0_agents:
    for team1 in team1_agents:
        edge_name = component_editor.concat_comps(team0, team1)
        edges[edge_name] = dict(  ####sorted!!!
            reward_fns=dict(
                # move to opponent's direction
                team0_move_to_team1=dict(
                    reward_type=reward_functions.direction_reward,
                    vel0=lambda x, y: so('body', 'vel', x['root'], indices=(0, 1)),  ####sorted!!!
                    vel1=lambda x, y: so('body', 'vel', y['root'], indices=(0, 1)),
                    pos0=lambda x, y: so('body', 'pos', x['root'], indices=(0, 1)),
                    pos1=lambda x, y: so('body', 'pos', y['root'], indices=(0, 1)),
                    scale=opp_scale,
                ),
                team1_move_to_team0=dict(
                    reward_type=reward_functions.direction_reward,
                    vel0=lambda x, y: so('body', 'vel', y['root'], indices=(0, 1)), ####sorted!!!
                    vel1=lambda x, y: so('body', 'vel', x['root'], indices=(0, 1)),
                    pos0=lambda x, y: so('body', 'pos', y['root'], indices=(0, 1)),
                    pos1=lambda x, y: so('body', 'pos', x['root'], indices=(0, 1)),
                    scale=opp_scale,
                ),
            ))
        # jax.debug.breakpoint()
        agent_groups['team0']['reward_names'] += (
                                                ('team0_move_to_team1', team0, team1),
                                                )
        agent_groups['team1']['reward_names'] += (
                                                ('team1_move_to_team0',team0, team1),
                                                )
  for team0 in team0_agents:
    components[team0]=dict(
          reward_fns =
            dict(
                team0_control_penalty=dict(
                    reward_type=reward_functions.control_reward,
                    scale=control_scale,
                ),
            ))
    agent_groups['team0']['reward_names'] += (('team0_control_penalty', team0),
                                                )
  for team1 in team1_agents:
    components[team1]=dict(
          reward_fns =
            dict(
                team1_control_penalty=dict(
                    reward_type=reward_functions.control_reward,
                    scale=control_scale,
                ),
            ))
    agent_groups['team1']['reward_names'] += (('team1_control_penalty', team1),
                                                )
  # add sumo ring
  components.update(get_ring_components(radius=ring_size, num_segments=20))
  merge_desc(
      env_desc,
      dict(agent_groups=agent_groups, edges=edges, components=components))
  return env_desc





TASK_MAP = dict(
    sumo=add_sumo)


def create_desc(team0_agent: List[str] = [], 
                team1_agent: List[str] = [],
                team0_agent_params: Dict[str, Any] = None,
                team1_agent_params: Dict[str, Any] = None,
                team0_num_agents: int = 2,
                team1_num_agents: int = 2,
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
             '1centipede':functools.partial(create_desc, team0_agent= ['centipede'], team0_num_agents = 1, team1_agent= ['centipede'], team1_num_agents = 1, init_r = 2, ring_size=3,  team0_torso_name = ['torso_0'], team1_torso_name = ['torso_0'],task='sumo'),
             '1ant':functools.partial(create_desc, team0_agent= ['ant'], team0_num_agents = 1, team1_agent= ['ant'], team1_num_agents = 1,  init_r = 2, ring_size=3,  team0_torso_name = ['torso'], team1_torso_name = ['torso'], task='sumo'),
             '1claw':functools.partial(create_desc, team0_agent= ['claw'], team0_num_agents = 1, team1_agent= ['claw'], team1_num_agents = 1,  init_r = 2, ring_size=3,  team0_torso_name = ['torso'], team1_torso_name = ['torso'], task='sumo'),
             '2claws':functools.partial(create_desc, team0_agent= ['claw','claw'], team0_num_agents = 2, team1_agent= ['claw','claw'], team1_num_agents = 2,  init_r = 2, ring_size=3, team0_torso_name = ['torso', 'torso'], team1_torso_name = ['torso', 'torso'], task='sumo'),
             '2ants':functools.partial(create_desc, team0_agent= ['ant','ant'], team0_num_agents = 2, team1_agent= ['ant','ant'], team1_num_agents = 2,  init_r = 2, ring_size=3, team0_torso_name = ['torso', 'torso'], team1_torso_name = ['torso', 'torso'], task='sumo'),
             '3ants':functools.partial(create_desc, team0_agent= ['ant']*3, team0_num_agents = 3, team1_agent= ['ant']*3, team1_num_agents = 3,  init_r = 3, ring_size=4, team0_torso_name = ['torso']*3, team1_torso_name = ['torso']*3, task='sumo'),
             '2ac':functools.partial(create_desc, team0_agent= ['ant','centipede'], team0_num_agents = 2, team1_agent= ['ant','centipede'], team1_num_agents = 2,  init_r = 2, ring_size=3, team0_torso_name = ['torso', 'torso_0'],  team1_torso_name = ['torso', 'torso_0'], task='sumo'),
             '2cc':functools.partial(create_desc, team0_agent= ['claw','centipede'], team0_num_agents = 2, team1_agent= ['claw','centipede'], team1_num_agents = 2,  init_r = 2, ring_size=3, team0_torso_name = ['torso', 'torso_0'],  team1_torso_name = ['torso', 'torso_0'], task='sumo'),
             '2antclaw':functools.partial(create_desc, team0_agent= ['ant','claw'], team0_num_agents = 2, team1_agent= ['ant','claw'], team1_num_agents = 2,  init_r = 2, ring_size=3, team0_torso_name = ['torso', 'torso'],  team1_torso_name = ['torso', 'torso'], task='sumo'),
             '2cents':functools.partial(create_desc, team0_agent= ['centipede','centipede'], team0_num_agents = 2, team1_agent= ['centipede','centipede'], team1_num_agents = 2,  init_r = 2, ring_size=3, team0_torso_name = ['torso_0', 'torso_0'],  team1_torso_name = ['torso_0', 'torso_0'], task='sumo'),
             '2unimals':functools.partial(create_desc, team0_agent= ['unimal']*2, team0_num_agents = 2, team1_agent= ['unimal']*2, team1_num_agents = 2,  init_r = 3, ring_size=4, team0_torso_name = ['torso_0']*2, team1_torso_name = ['torso_0']*2, task='sumo'),
             }

