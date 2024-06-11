import functools
from procedural_envs.misc.observers import SimObserver
from procedural_envs.misc.random_init_functions import annulus_xy_sampler, uniform_z_sampler
from procedural_envs.misc.reward_functions import nearest_distance_reward, distance_reward


def load_desc(
    num_legs: int = 4,
    radius: float = 0.1,
    r_min: float = 1.25,
    r_max: float = 1.75,
    z_min: float = 0.5,
    z_max: float = 0.6,
    constant_z: bool = False,
    min_dist_z: float = 0.1,
    z_done_bonus: float = 0.0):
  # define random_init_fn for 'ball' component
  random_target_init_fn = functools.partial(
    annulus_xy_sampler, r_min=r_min, r_max=r_max)
  random_z_target_init_fn = functools.partial(
    uniform_z_sampler, z_min=z_min, z_max=z_max, const=constant_z)

  height_index = (0, 2)
  leg_indices = []
  for i in range(num_legs):
    if i not in height_index:
      leg_indices.append(i)

  return dict(
      components=dict(
          agent1=dict(
              component='ant',
              component_params=dict(num_legs=num_legs),
              pos=(0, 0, 0),
              reward_fns=dict(
                  distance=dict(
                      reward_type=nearest_distance_reward,
                      target=SimObserver(comp_name='cap1', sdname='Ball', indices=(0, 1)),
                      obs=[
                          SimObserver(comp_name='agent1', sdname=f'$ Body 4_{i}', indices=(0, 1)) for i in leg_indices],
                      min_dist=radius,
                      done_bonus=0.0),
                  leg_height=dict(
                      reward_type=distance_reward,
                      obs1=SimObserver(comp_name='cap2', sdname='Z_Target_1', indices=(2,)),
                      obs2=SimObserver(comp_name='agent1', sdname=f'$ Body 4_{height_index[0]}', indices=(2,)),
                      min_dist=min_dist_z,
                      done_bonus=z_done_bonus),
                  leg_height_2=dict(
                      reward_type=distance_reward,
                      obs1=SimObserver(comp_name='cap3', sdname='Z_Target_2', indices=(2,)),
                      obs2=SimObserver(comp_name='agent1', sdname=f'$ Body 4_{height_index[1]}', indices=(2,)),
                      min_dist=min_dist_z,
                      done_bonus=z_done_bonus)
              ),
          ),
          cap1=dict(
              component='ball',
              component_params=dict(
                  radius=radius,
                  frozen=True,
                  name="Ball"
                  ),
              pos=(0, 0, 0),
              random_init='pos',
              random_init_fn=random_target_init_fn,
          ),
          cap2=dict(
              component='ball',
              component_params=dict(
                  radius=min_dist_z,
                  frozen=True,
                  name="Z_Target_1"
                  ),
              pos=(0, 0, 0),
              random_init='pos',
              random_init_fn=random_z_target_init_fn,
          ),
          cap3=dict(
              component='ball',
              component_params=dict(
                  radius=min_dist_z,
                  frozen=True,
                  name="Z_Target_2"
                  ),
              pos=(0, 0, 0),
              random_init='pos',
              random_init_fn=random_z_target_init_fn,
          ),
        ),
      global_options=dict(dt=0.05, substeps=10),
      goal_based_task=True,
      satisfy_all_cond=True,
      task_edge=[
        ['cap1___Ball']+[f'agent1___$ Body 4_{i}' for i in leg_indices],
        ['cap2___Z_Target_1', f'agent1___$ Body 4_{height_index[0]}'],
        ['cap3___Z_Target_2', f'agent1___$ Body 4_{height_index[1]}'],
        ]
      )

ENV_DESCS = dict()

# add environments
for i in range(3, 7, 1):
  ENV_DESCS[f'ant_reach_handsup2_{i}'] = functools.partial(load_desc, num_legs=i)
