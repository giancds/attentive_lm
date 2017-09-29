# -*- coding: utf-8 -*-
"""
Definition of the model.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

import rnn
from data_reader import PAD_ID
from optimization_ops import optimize
from utils import build_lm_layers
from utils import get_3d_tensor_shapes


# pylint: disable=R0903,W0108


class AttentiveLM(object):
  """The Attentive LM model. """

  def __init__(self,
               params,
               batch_size,
               num_steps,
               queue=None,
               is_training=True,
               keep_attention_weights=False,
               log_tensorboard=True):
    """  Create the AttentiveLM.

    Args:
        params: tensorflow.params. the set of parameters that will be used to
            build the model.
        input_: LMInput object. This object holds the data that weill be
            used to train or validate/test the model.
        is_training: boolean, default to True. Indicates whether the model
            will perform weight updates (learning) or will only be use to
            calculate losses for the dataset presented to it.

    """
    dtype = tf.float32
    self.batch_size = batch_size
    self.num_steps = num_steps

    if queue is not None:
      inputs = queue.dequeue_many(batch_size)
      self.input_data = inputs[0]
      self.targets = inputs[1]
    else:
      # Feeds for inputs.
      self.input_data = tf.placeholder(
        tf.int32, shape=[batch_size, num_steps], name="input_data")
      self.targets = tf.placeholder(
        tf.int32, shape=[batch_size, num_steps], name="targets")


    mask = tf.to_float(self.targets > PAD_ID)

    keep_prob = 1.0 - params.dropout_rate
    output_size = params.hidden_size

    # building the layers
    cell = build_lm_layers(
      params.num_layers, params.hidden_size,
      is_training=is_training, keep_prob=keep_prob,
      decoding_function_name=params.score_form,
      keep_attention_weights=keep_attention_weights)

    with tf.device("/cpu:0"):
      embedding = tf.get_variable(
        "embedding", [params.vocab_size, params.hidden_size], dtype=dtype)

    seq_lengths = tf.reduce_sum(mask, 1)
    if batch_size > 1:
      seq_lengths = tf.squeeze(seq_lengths)
    self.nwords = tf.reduce_sum(seq_lengths)

    inputs = tf.nn.embedding_lookup(embedding, self.input_data)
    if is_training and keep_prob < 1.0:
      inputs = tf.nn.dropout(inputs, keep_prob)

    outputs, self.final_state, weights = attentive_lm(
      inputs=inputs, cell=cell, is_training=is_training,
      seq_lengths=seq_lengths, keep_attention_weights=keep_attention_weights)

    self.attention_weights = weights
    outputs = tf.reshape(outputs, [-1, output_size])

    with tf.device("/cpu:0"):
      # we use the transposed embedding matrix as the softmax weights
      # or get a new matrix from scratch
      if params.tie_softmax_weights:
        softmax_w = tf.transpose(embedding)
      else:
        softmax_w = tf.get_variable(
          "softmax_w", [output_size, params.vocab_size], dtype=dtype)

      # softmax bias
      init_bias = tf.constant_initializer(0.0, dtype=dtype)
      softmax_b = tf.get_variable("softmax_b", [params.vocab_size],
                                  dtype=dtype, initializer=init_bias)

    logits = tf.matmul(outputs, softmax_w) + softmax_b
    loss = tf.contrib.legacy_seq2seq.sequence_loss_by_example(
      [logits], [tf.reshape(self.targets, [-1])],
      [tf.reshape(mask, [-1])], softmax_loss_function=None)

    # variables for training
    self.learning_rate = tf.get_variable(
      name="lr_rate", trainable=False, dtype=dtype,
      initializer=tf.convert_to_tensor(params.learning_rate, dtype=dtype))
    self.global_step = tf.contrib.framework.get_or_create_global_step()

    # calculating loss: first we sum all losses in the batch and normalize
    # it by the batch size. we then sum the mask to get the number of words
    # in this particular batch so we can calculate the validation and test
    # perplexities later
    sum_loss = tf.reduce_sum(loss)
    cost = (sum_loss / batch_size)

    # for the validation and testing we get the perplexity per word so we do
    # not normalize the loss per batch size but we get the total loss and
    # divide by the summation over the mask as it will give us the number of
    # words in the batch
    mask_sum = tf.reduce_sum(mask)
    self.batch_loss = (sum_loss / mask_sum) if is_training else sum_loss

    if not is_training:
      return

    # this function will perform weight updates and gradient clipping
    self.train_op, grads, tvars = optimize(
      cost=cost, global_step=self.global_step,
      optimizer=tf.train.GradientDescentOptimizer(self.lr_rate),
      max_grad_norm=params.max_grad_norm)

    # learning rate decay ops
    self._new_lr = tf.placeholder(dtype, shape=[],
                                  name="new_learning_rate")
    self._lr_update = tf.assign(self.learning_rate, self._new_lr)

  def assign_lr(self, session, lr_value):
    """ Op to update the learning rate when using SGD and learning rate
    decay.
    """
    session.run(self._lr_update,
                feed_dict={self._new_lr: lr_value})

  @property
  def lr_rate(self):
    """ The current learning rate for used to update the weight in each
    backprop step.
    """
    return self.learning_rate


def _get_attention_weights(cell, is_training):
  """ Obtain the attention weights if needed. """
  weights = None
  if is_training:
    weights = cell._cells[-1]._cell.attention_weights  # pylint: disable=W0212
  else:
    weights = cell._cells[-1].attention_weights  # pylint: disable=W0212
  return weights


def _reset_attention_state(cell, is_training):
  """ Reset the previous hiddens and attention weights in the rnn cell. """
  if is_training:
    cell._cells[-1]._cell.previous_hiddens = []  # pylint: disable=W0212
    cell._cells[-1]._cell._attention_weights = []  # pylint: disable=W0212
  else:
    cell._cells[-1].previous_hiddens = []  # pylint: disable=W0212
    cell._cells[-1]._attention_weights = []  # pylint: disable=W0212


def attentive_lm(inputs,
                 cell,
                 seq_lengths,
                 is_training=True,
                 keep_attention_weights=False):
  """ Run the AttentiveLM loop. """
  _reset_attention_state(cell, is_training)
  batch_size, _, _ = get_3d_tensor_shapes(inputs)
  final_state = cell.zero_state(batch_size, tf.float32)
  outputs = []

  with tf.variable_scope("attentive_lm") as attentive_scope:

    if not keep_attention_weights:
      final_outputs, final_state = tf.nn.dynamic_rnn(
        cell=cell, inputs=inputs, initial_state=final_state,
        swap_memory=True, sequence_length=seq_lengths)

    else:

      with tf.variable_scope("rnn") as scope:
        for  step, input_ in enumerate(tf.unstack(inputs, axis=1)):
          if step > 0:
            scope.reuse_variables()
          output, final_state = rnn.rnn_step(  # pylint: disable=E1101, W0212
            time=step,
            sequence_length=seq_lengths,
            min_sequence_length=tf.reduce_min(seq_lengths),
            max_sequence_length=tf.reduce_max(seq_lengths),
            zero_output=tf.zeros([batch_size, cell.output_size]),
            state=final_state,
            call_cell=lambda: cell(input_, final_state),  # pylint: disable=W0640
            state_size=cell.state_size,
            skip_conditionals=True)
          outputs.append(tf.reshape(output, [-1, 1, cell.output_size]))

        final_outputs = tf.concat(outputs, axis=1)

  weights = (_get_attention_weights(cell, is_training)
             if keep_attention_weights else None)

  return final_outputs, final_state, weights
