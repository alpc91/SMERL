"""Simulator utilities."""
import collections
import functools
from typing import Callable, List
import brax.v1
import jax
from jax import numpy as jnp

lim_to_dof = {0: 1, 1: 1, 2: 2, 3: 3}


@functools.partial(jax.vmap, in_axes=[0, 0, None, None, None])
def transform_qp(qp, mask: jnp.ndarray, rot: jnp.ndarray, rot_vec: jnp.ndarray,
                 offset_vec: jnp.ndarray):
  """Rotates a qp by some rot around some ref_vec and translates it.

  Args:
    qp: QPs to be rotated
    mask: whether to transform this qp or not
    rot: Quaternion to rotate by
    rot_vec: point around which to rotate.
    offset_vec: relative displacement vector to translate qp by

  Returns:
    transformed QP
  """
  relative_pos = qp.pos - rot_vec
  new_pos = brax.math.rotate(relative_pos, rot) + rot_vec + offset_vec
  new_rot = brax.math.quat_mul(rot, qp.rot)
  new_vel = brax.math.rotate(qp.vel, rot) #add 2024.1.1
  new_ang = brax.math.rotate(qp.ang, rot) #add 2024.1.1
  return brax.v1.physics.base.QP(
      pos=jnp.where(mask, new_pos, qp.pos),
      vel=jnp.where(mask, new_vel, qp.vel), #modify 2024.1.1 
      ang=jnp.where(mask, new_ang, qp.ang), #modify 2024.1.1
      rot=jnp.where(mask, new_rot, qp.rot))



def sample_init_qp(qp,
                   pos_or_quat: str,
                   random_init_fn: Callable,
                   mask: jnp.ndarray):
  sample = random_init_fn()
  if pos_or_quat == 'pos':
    return brax.v1.physics.base.QP(
      pos=jnp.where(mask, sample, qp.pos),
      vel=qp.vel,
      ang=qp.ang,
      rot=qp.rot)
  elif pos_or_quat == 'quat':
    return brax.v1.physics.base.QP(
      pos=qp.pos,
      vel=qp.vel,
      ang=qp.ang,
      rot=jnp.where(mask, sample, qp.rot))


def get_names(config, datatype: str = 'body'):
  objs = {
      'body': config.bodies,
      'joint': config.joints,
      'actuator': config.actuators,
  }[datatype]
  return [b.name for b in objs]


def get_joint_value(sys, qp, info: collections.OrderedDict):
  """Get joint values."""
  values = collections.OrderedDict()
  angles_vels = {j.dof: j.angle_vel(qp) for j in sys.joints}
  for k, v in info.items():
    for i, type_ in zip((0, 1), ('pos', 'vel')):
      vvv = jnp.array([vv[v['index']] for vv in angles_vels[v['dof']][i]])
      values[f'joint_{type_}:{k}'] = vvv
  return values


def names2indices(config, names: List[str], datatype: str = 'body'):
  """Convert name string to indices for indexing simulator states."""

  if isinstance(names, str):
    names = [names]

  indices = {}
  info = {}

  objs = {
      'body': config.bodies,
      'joint': config.joints,
      'actuator': config.actuators,
  }[datatype]
  joint_counters = [0, 0, 0]
  actuator_counter = 0
  for i, b in enumerate(objs):
    if datatype == 'joint':
      dof = lim_to_dof[len(b.angle_limit)]
    elif datatype == 'actuator':
      joint = [j for j in config.joints if j.name == b.joint][0]
      dof = lim_to_dof[len(joint.angle_limit)]
    if b.name in names:
      indices[b.name] = i
      if datatype in ('joint',):
        info[b.name] = dict(dof=dof, index=joint_counters[dof - 1])
      if datatype in ('actuator',):
        info[b.name] = tuple(range(actuator_counter, actuator_counter + dof))
    if datatype in ('joint',):
      joint_counters[dof - 1] += 1
    if datatype in ('actuator',):
      actuator_counter += dof

  indices = [indices[n] for n in names]
  mask = jnp.array([b.name in names for b in objs])

  if datatype in ('actuator', 'joint'):
    info = collections.OrderedDict([(k, info[k]) for k in names])

  return indices, info, mask
