# coding: utf-8 -*-
"""
This module contain content-based functions to neural Machine Translations of

    Bahdanau et al., (2014) - https://arxiv.org/pdf/1409.0473.pdf

and

    Luong et al (2015) - http://www.aclweb.org/anthology/D15-1166

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import cells

from utils import get_2d_tensor_shapes
from utils import reshape_attention, convolve

# pylint: disable=C0103

COMBINED = "combined"
SINGLE = "single"

# TODO: there is a better way of doing this so we can remove the first unused
# argument?
def score_single(unused_arg,  # not used - pylint: disable=W0613
                 decoder_previous,  # h_i
                 reuse_variables=False,
                 dtype=tf.float32):
  """ Applies a score function of the form

              v.(W.hi)

  to the hidden states of the decoder, where W is a weight matrix, v is a
  vector of parameters and hi is one of each of the decoder hidden states.

  The function performs a 1-by-1 convolution to calculate W.hi and the vector
  v is broadcasted with a multiplication step into the result of W.hi. After
  this step a reduce_sum is performed over axis=[2,3] so the correct results
  are obtained.

  Args:
      decoder_current: not used
      attn_size: the size of the attention vectors
      encoder_hiddens: 3-D Tensor [batch_size, timestep, hidden_dim]. It
          represents the hidden sattes of the decoder up to the current
          timestep.
      current_hidden: Tensor, representing the current hidden state at
          timestep t

  Returns:
      beta: decoder hidden states after applying the content function

  """

  with tf.variable_scope("score_salton_single") as scope:
    if reuse_variables:
      scope.reuse_variables()

    #
    decoder_previous, attn_dim = reshape_attention(decoder_previous)

    # we first get the correct weight matrix
    ws = tf.get_variable("AttnDecWs", [1, 1, attn_dim, attn_dim], dtype=dtype)

    # we apply a small convolution to the decoder states - it is more
    # efficient than performing a recurrent matrix * matrix
    hidden_features = convolve(decoder_previous, ws)

    # we then get the vector v that will be used on the second
    # multiplication op.
    vs = tf.get_variable("AttnDecV_%d" % 0, [attn_dim], dtype=dtype)

    scores = tf.reduce_sum((vs * tf.tanh(hidden_features)), [2, 3])

  return scores


# TODO: check a way of performing W1.hi beforehand so we avoid repeating the
# same multiplicaiton at each iteration
def score_combined(decoder_current,  # h_t
                   decoder_previous,  # h_i
                   reuse_variables=False,
                   dtype=tf.float32):
  """ Applies a score function of the form

              v.(W1.hi + W2.hs)

  where W is a weight matrix, v is a vector of parameters, hi is one
  of each of the decoder hidden states and hs is the current hidden state at
  timestep t.

  The function performs a 1-by-1 convolution to calculate W.hi  and performs
  the W2.hs step using a ``linear'' cell (see cells.linear for the
  documentation)  and broadcasted into the result of W1.hi (encoder_hiddens)
  via multiplication step.  After this step a reduce_sum is performed over
  axis=[2,3] so the correct results are obtained.

  Args:
      decoder_current: not used
      attn_size: the size of the attention vectors
      encoder_hiddens: 3-D Tensor [batch_size, timestep, hidden_dim]. It
          represents the hidden sattes of the decoder up to the current
          timestep.
      current_hidden: Tensor, representing the current hidden state at
          timestep t

  Returns:
      beta: decoder hidden states after applying the content function

  """
  with tf.variable_scope("score_salton_combined") as scope:
    if reuse_variables:
      scope.reuse_variables()

    _, output_size = get_2d_tensor_shapes(decoder_current)

    decoder_current = cells.linear(
      [decoder_current], output_size, bias=False, dtype=dtype)

    #
    decoder_previous, attn_dim = reshape_attention(decoder_previous)

    # we first get the correct weight matrix
    ws = tf.get_variable("AttnDecWs", [1, 1, attn_dim, attn_dim], dtype=dtype)

    # we apply a small convolution to the decoder states - it is more
    # efficient than performing a recurrent matrix * matrix
    hidden_features = convolve(decoder_previous, ws)

    hidden_features = hidden_features + decoder_current

    # we then get the vector v that will be used on the second
    # multiplication op.
    vs = tf.get_variable("AttnDecVs", [attn_dim], dtype=dtype)

    scores = tf.reduce_sum((vs * tf.tanh(hidden_features)), [2, 3])

  return scores


def get_decoder_content_function(name):
  """ Return the corresponding decoder scoring content function as defined
      by 'name'.
  """
  if name == SINGLE:
    return score_single

  elif name == COMBINED:
    return score_combined

  else:
    raise ValueError
