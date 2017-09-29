# -*- coding: utf-8 -*-
"""
utility functions to train the RNN-based VariationalAutoencoder

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import os

import numpy as np
import tensorflow as tf

import attention
import cells

# TODO: pydocs

TRAIN_INFO_LM = {
  "epoch": 0,
  "best_valid_ppx": np.inf,
  "best_epoch": 0,
  "estop_counter": 0,
  "current_cost": 0.0
}


def get_2d_tensor_shapes(tensor):
  """ """

  length = tensor.get_shape()[0].value

  if length is None:
    length = tf.shape(tensor)[0]

  dim = tensor.get_shape()[1].value

  return length, dim


def get_3d_tensor_shapes(tensor):
  """ """

  batch = tensor.get_shape()[0].value
  length = tensor.get_shape()[1].value

  if length is None:
    length = tf.shape(tensor)[2]

  dim = tensor.get_shape()[2].value

  return batch, length, dim


def reshape_attention(attention_states):
  """ """

  _, attn_length, attn_dim = get_3d_tensor_shapes(attention_states)

  # To calculate W1 * h_t we use a 1-by-1 convolution, need to reshape before.
  hidden = tf.reshape(attention_states, [-1, attn_length, 1, attn_dim])

  return hidden, attn_dim


def convolve(tensor, matrix):
  """ """
  return tf.nn.conv2d(tensor, matrix, [1, 1, 1, 1], "SAME")


def build_lm_layers(num_layers,
                    size,
                    is_training=False,
                    decoding_function_name=None,
                    keep_prob=1.0,
                    keep_attention_weights=False):
  """ Helper to build recurrent layers for he LM. """

  decoding_function = None

  # building the layers
  lstm_cell0 = tf.contrib.rnn.BasicLSTMCell(
      size, forget_bias=1.0, reuse=not is_training)
  # lstm_cell0 = tf.contrib.rnn.LSTMBlockCell(
  #     size, forget_bias=1.0)

  lstm_cell1 = tf.contrib.rnn.DropoutWrapper(
    lstm_cell0, output_keep_prob=keep_prob
  ) if is_training and keep_prob < 1.0  else lstm_cell0

  if decoding_function_name is not None:

    decoding_function = attention.get_decoder_content_function(
      decoding_function_name)

    lstm_cellA = cells.AttentionLSTMCell(
      size, forget_bias=1.0, state_is_tuple=True,
      init_constant_output_bias=False,
      decoding_function=decoding_function,
      keep_attention_weights=keep_attention_weights,
      reuse=tf.get_variable_scope().reuse)

    lstm_cellA = tf.contrib.rnn.DropoutWrapper(
      lstm_cellA, output_keep_prob=keep_prob
    ) if is_training and keep_prob < 1.0 else lstm_cellA

    # internal_cell = [lstm_cell1] * (num_layers - 1)
    internal_cell = [lstm_cell1 for _ in range(num_layers - 1)]
    internal_cell = internal_cell + [lstm_cellA]

  else:

    internal_cell = [lstm_cell1 for _ in range(num_layers)]

  cell = tf.contrib.rnn.MultiRNNCell(internal_cell, state_is_tuple=True)

  return cell

def create_queue(data_size,
                 num_steps,
                 capacity=128,
                 dtype=tf.int32):
  """ Create the queue and related ops and placeholders to be used to
  feed the network.
  """

  # Feeds for inputs.
  input_data = tf.placeholder(
    dtype, shape=[data_size, num_steps], name="input_data")

  targets = tf.placeholder(
    dtype, shape=[data_size, num_steps], name="targets")

  queue = tf.FIFOQueue(
    capacity=capacity,
    # min_after_dequeue=min_after_dequeue,
    dtypes=[dtype, dtype],
    shapes=[[num_steps]] * 2)

  enqueue_op = queue.enqueue_many(
    [input_data, targets])

  placeholders = {
    "input_data": input_data,
    "targets": targets
  }

  return queue, enqueue_op, placeholders
