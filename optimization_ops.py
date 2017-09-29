# -*- coding: utf-8 -*-
"""
Optimization ops.

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


def optimize(cost,
             optimizer,
             global_step,
             max_grad_norm=5.0):
  """ Helper funciton to return the optimization op that will optimize
  the network during training.

  This function will obtain all trainable variables and clip their
  gradients by their global norm at max_grad_norm.

  After that, the optimizer object passed will get the (possibly) cplipped
  gradients and apply the updates defining the backprop step that will
  represent our training op.

  Args:
      cost: Tensor representing the cost obtained.
      optimizer: tf.train.Optimizer object. Valid objects are
          - GradientDescentOptimizer
          - AdagradOptimizer
          - AdamOptimizer
          - RMSPropOptimizer
      max_grad_norm: float, the maximum norm for the gradients before we
          clip them.

  Returns:
      train_op: the operation that we will run to optimize the network.
      grads: list of tensors, representing the gradients for each
          trainable variable
      tvars: list of Tensors, the trainable variables

  """
  assert optimizer is not None

  tvars = tf.trainable_variables()

  grads, _ = tf.clip_by_global_norm(
    tf.gradients(cost, tvars), max_grad_norm)
  train_op = optimizer.apply_gradients(zip(grads, tvars),
                                       global_step=global_step)

  return train_op, grads, tvars
