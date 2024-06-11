from procedural_envs.tasks.observation_config.one_hot_limb_id import AMORPHEUS

OBSERVATION_SIZE = 12 + 3 + 11

OBSERVATION_CONFIG = {
  'one_hot_limb_id': False,
  'one_hot_limb_id_dict': AMORPHEUS,
  'vel_clipping': True,
  'vel_clipping_range': (-10., 10),
  'angle_limit': 0.,
  'mass_clipping_range': 100.,
  'inertia_clip_range': 10.,
  'scale_stiffness': 0.001,
  'scale_angular_damping': 0.1,
  'scale_actuator_strength': 0.01,
  'rel_obs': {'pos': False, 'rot': False, 'vel': False, 'ang': False},
  'is_rel_obs_only': {'pos': False, 'rot': False, 'vel': False, 'ang': False},
  'add_joint_angle': True,
  'add_joint_range': True,
  'add_joint_vel': False,
  'add_egocentric_pos': True,
  'morphological_parameters': {
      'shape': True,  # 3
      'mass': True,  # 1
      'inertia': True,  # 3
      'friction': False,
      'frozen': False,
      'stiffness': False,
      'angular_damping': False,
      'actuator_strength': True,  # 1
      'actuator_dof_idx': True,  # 3
    },
  'num_goal': 3,
  'one_hot_goal_edge': False,
  'direct_goal_edge': True,
}
