# -*- coding: utf-8 -*-
# pylint: disable=C0103,C0112
""" Extra cells and functions to be used with the attentive LM """
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.python.util import nest
from utils import get_2d_tensor_shapes

# RNNCell = tf.contrib.rnn.RNNCell


def linear(args, output_size, bias, bias_start=0.0, init_constant_bias=False,
           initializer=None, scope=None, dtype=tf.float32):
  """Linear map: sum_i(args[i] * W[i]), where W[i] is a variable.

  Args:
    args: a 2D Tensor or a list of 2D, batch x n, Tensors.
    output_size: int, second dimension of W[i].
    bias: boolean, whether to add a bias term or not.
    bias_start: starting value to initialize the bias; 0 by default.
    init_constant_bias: boolean. If False, the variable scope initializer will
      be used to initialize the bias parameter. If True, the bias Parameters
      will be initialized to a constant value as definied in bias_start.
    scope: VariableScope for the created subgraph; defaults to "Linear".

  Returns:
      A 2D Tensor with shape [batch x output_size] equal to
      sum_i(args[i] * W[i]), where W[i]s are newly created matrices.

  Raises:
    ValueError: if some of the arguments has unspecified or wrong shape.
  """
  assert args is not None
  if args is None or (nest.is_sequence(args) and not args):
    raise ValueError("`args` must be specified")
  if not nest.is_sequence(args):
    args = [args]

  # Calculate the total size of arguments on dimension 1.
  total_arg_size = 0
  shapes = [a.get_shape().as_list() for a in args]
  for shape in shapes:
    if len(shape) != 2:
      raise ValueError("Linear is expecting 2D arguments: {0}".format(
        str(shapes)))
    if not shape[1]:
      raise ValueError("Linear expects shape[1] of arguments: {0}".format(
        str(shapes)))
    else:
      total_arg_size += shape[1]

  # dtype = [a.dtype for a in args][0]

  # Now the computation.
  with tf.variable_scope(scope or "Linear"):  # , reuse=reuse_variables):
    matrix = tf.get_variable("Matrix", [total_arg_size, output_size],
                             dtype=dtype, initializer=initializer)
    if len(args) == 1:
      res = tf.matmul(args[0], matrix)
    else:
      res = tf.matmul(tf.concat(axis=1, values=args), matrix)
    if not bias:
      return res

    if init_constant_bias:
      init_bias = tf.constant_initializer(bias_start)
    else:
      init_bias = initializer
    bias_term = tf.get_variable("Bias", [output_size], dtype=dtype,
                                initializer=init_bias)
  return res + bias_term


class AttentionLSTMCell(tf.contrib.rnn.RNNCell):
  """Long short-term memory unit (LSTM) recurrent network cell.

  The default non-peephole implementation is based on:

    http://deeplearning.cs.cmu.edu/pdfs/Hochreiter97_lstm.pdf

  S. Hochreiter and J. Schmidhuber.
  "Long Short-Term Memory". Neural Computation, 9(8):1735-1780, 1997.

  The peephole implementation is based on:

    https://research.google.com/pubs/archive/43905.pdf

  Hasim Sak, Andrew Senior, and Francoise Beaufays.
  "Long short-term memory recurrent neural network architectures for
   large scale acoustic modeling." INTERSPEECH, 2014.
  """

  def __init__(self,
               num_units,
               #  input_size=None,
               initializer=None,
               forget_bias=1.0,
               state_is_tuple=True,
               activation=tf.tanh,
               decoding_function=None,
               init_constant_output_bias=True,
               keep_attention_weights=0,
               reuse=None):
    """Initialize the parameters for an LSTM cell.

    Args:
      num_units: int, The number of units in the LSTM cell
      input_size: Deprecated and unused.
      initializer: (optional) The initializer to use for the weight and
        projection matrices.
      num_proj: (optional) int, The output dimensionality for the projection
        matrices.  If None, no projection is performed.
      num_unit_shards: How to split the weight matrix.  If >1, the weight
        matrix is stored across num_unit_shards.
      num_proj_shards: How to split the projection matrix.  If >1, the
        projection matrix is stored across num_proj_shards.
      forget_bias: Biases of the forget gate are initialized by default to 1
        in order to reduce the scale of forgetting at the beginning of
        the training.
      state_is_tuple: If True, accepted and returned states are 2-tuples of
        the `c_state` and `m_state`.  If False, they are concatenated
        along the column axis.  This latter behavior will soon be
        deprecated.
      activation: Activation function of the inner states.
    """
    super(AttentionLSTMCell, self).__init__(_reuse=reuse)
    self.decoder_attentive = (True
                              if decoding_function is not None
                              else False)
    if self.decoder_attentive:
      assert state_is_tuple, "State must be tuple!"

    self._num_units = num_units
    self._initializer = initializer
    self._forget_bias = forget_bias
    self._state_is_tuple = state_is_tuple
    self._activation = activation
    # self._dtype = dtype

    self._state_size = (
      tf.contrib.rnn.LSTMStateTuple(num_units, num_units)
      if state_is_tuple else 2 * num_units)
    self._output_size = num_units

    # added properties for attention calculations
    self._decoding_function = decoding_function
    self._current_hidden = None
    self.previous_hiddens = []
    self._context = []
    self._init_constant_bias = init_constant_output_bias
    self.input_lengths = None
    self._attention_weights = []
    self._keep_attention_weights = True if keep_attention_weights > 0 else False

  @property
  def state_size(self):
    return self._state_size

  @property
  def output_size(self):

    return self._output_size

  @property
  def attention_weights(self):
    return self._attention_weights

  @property
  def contexts(self):
    return self._context

  def _compute_decoder_context(self, decoder_hidden, dtype=tf.float32):
    """ """
    reuse_variables = True if len(self.previous_hiddens) > 1 else None

    if len(self.previous_hiddens) == 0:
      batch, _ = get_2d_tensor_shapes(decoder_hidden)
      self.previous_hiddens = [
        tf.zeros([batch, 1, self.output_size], dtype=dtype)]

    previous_states = tf.reshape(
      tf.concat(axis=1, values=self.previous_hiddens),
      [-1, len(self.previous_hiddens), self.output_size])

    scores = self._decoding_function(decoder_hidden,
                                     decoder_previous=previous_states,
                                     reuse_variables=reuse_variables,
                                     dtype=dtype)

    _, timesteps = get_2d_tensor_shapes(scores)

    weights = tf.nn.softmax(scores)
    weights = tf.reshape(tf.squeeze(weights), [-1, timesteps, 1])

    context = tf.reduce_sum(weights * previous_states, [1])

    self.previous_hiddens += [self._current_hidden]

    self._attention_weights.append(weights)

    return context

  def __call__(self, inputs, state, scope=None):
    return self.call(inputs, state)

  def call(self, inputs, state):
    """Run one step of LSTM.

    Args:
      inputs: input Tensor, 2D, batch x num_units.
      state: if `state_is_tuple` is False, this must be a state Tensor,
        `2-D, batch x state_size`.  If `state_is_tuple` is True, this must
        be a tuple of state Tensors, both `2-D`, with column sizes
        `c_state` and `m_state`.
      scope: VariableScope for the created subgraph; defaults to "LSTMCell".

    Returns:
      A tuple containing:

      - A `2-D, [batch x output_dim]`, Tensor representing the output of
        the LSTM after reading `inputs` when previous state was `state`.
        Here output_dim is:
           num_proj if num_proj was set,
           num_units otherwise.
      - Tensor(s) representing the new state of LSTM after reading `inputs`
        when the previous state was `state`.  Same type and shape(s) as
        `state`.

    Raises:
      ValueError: If input size cannot be inferred from inputs via
        static shape inference.
    """
    if self._state_is_tuple:
      (c_prev, m_prev) = state
    else:
      c_prev = tf.slice(state, [0, 0], [-1, self._num_units])
      m_prev = tf.slice(state, [0, self._num_units],
                        [-1, self._num_units])

    dtype = inputs.dtype
    input_size = inputs.get_shape().with_rank(2)[1]
    if input_size.value is None:
      raise ValueError(
        "Could not infer input size from inputs.get_shape()[-1]")
    scope = tf.get_variable_scope()
    with tf.variable_scope(scope, initializer=self._initializer) as unit_scope:
      w_shape = [input_size.value + self._num_units, 4 * self._num_units]
      concat_w = tf.get_variable(
        "W", dtype=dtype, shape=w_shape)

      b = tf.get_variable(
        "B", shape=[4 * self._num_units],
        initializer=tf.zeros_initializer(), dtype=dtype)

      # i = input_gate, j = new_input, f = forget_gate, o = output_gate
      cell_inputs = tf.concat(axis=1, values=[inputs, m_prev])
      lstm_matrix = tf.nn.bias_add(
        tf.matmul(cell_inputs, concat_w), b)
      i, j, f, o = tf.split(axis=1, num_or_size_splits=4,
                            value=lstm_matrix)

      c = (tf.sigmoid(f + self._forget_bias) * c_prev + tf.sigmoid(i) *
           self._activation(j))

      m = tf.sigmoid(o) * self._activation(c)

      new_state = (tf.contrib.rnn.LSTMStateTuple(c, m)
                   if self._state_is_tuple
                   else tf.concat(axis=1, values=[c, m]))

      if self.decoder_attentive:
        self._current_hidden = tf.reshape(
          new_state[1], [-1, 1, self.output_size])

        with tf.variable_scope("decoder_attention") as decoder_attention:
          # ctx_decoder = self._compute_decoder_context(new_state[1])
          ctx_decoder = self._compute_decoder_context(m, dtype)

          reuse_variables = (True
                             if len(self.previous_hiddens) > 2
                             else None)

          if reuse_variables:
            decoder_attention.reuse_variables()

          m = linear(
            [m] + [ctx_decoder], self.output_size, True,
            init_constant_bias=self._init_constant_bias,
            dtype=dtype)

    return m, new_state
