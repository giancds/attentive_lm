# -*- coding: utf-8 -*-
"""
This module contains functions to run the main loop of the Attentive LM.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import attention

# pylint: disable=C0111,R0903

def get_config(name):
  if name == "ptb_combined":
    return PTBAttentiveCombined
  elif name == "ptb_single":
    return PTBAttentiveSingle
  elif name == "wikitext2_combined":
    return Wiki2AttentiveCombined
  elif name == "wikitext2_single":
    return Wiki2AttentiveSingle
  else:
    raise ValueError("Invalid model: {0}".format(name))


class PTBAttentiveCombined(object):

  num_layers = 2
  hidden_size = 650
  init_scale = 0.05
  dropout_rate = 0.5
  max_grad_norm = 5.0
  num_steps = 35  # for training only
  score_form = attention.COMBINED
  tie_softmax_weights = True

  batch_size = 32
  learning_rate = 1.0
  start_decay = 12
  lr_decay = 0.5

  vocab_size = 10000 + 1  # add 1 to account for PAD
  data_files = ["data/ptb/ptb.train.txt",
                "data/ptb/ptb.valid.txt",
                "data/ptb/ptb.test.txt"]


class PTBAttentiveSingle(object):

  num_layers = 2
  hidden_size = 650
  init_scale = 0.05
  dropout_rate = 0.5
  max_grad_norm = 5.0
  num_steps = 35  # for training only
  score_form = attention.SINGLE
  tie_softmax_weights = True

  batch_size = 32
  learning_rate = 1.0
  start_decay = 12
  lr_decay = 0.5

  vocab_size = 10000 + 1  # add 1 to account for PAD
  data_files = ["data/ptb/ptb.train.txt",
                "data/ptb/ptb.valid.txt",
                "data/ptb/ptb.test.txt"]


class Wiki2AttentiveCombined(object):

  num_layers = 2
  hidden_size = 1000
  init_scale = 0.05
  dropout_rate = 0.65
  max_grad_norm = 10.0
  num_steps = 35  # for training only
  score_form = attention.COMBINED
  tie_softmax_weights = True

  batch_size = 32
  learning_rate = 1.0
  start_decay = 14
  lr_decay = 1.0/1.15

  vocab_size = 33278 + 1  # add 1 to account for PAD
  data_files = ["data/wikitext-2/wiki2.train.txt",
                "data/wikitext-2/wiki2.valid.txt",
                "data/wikitext-2/wiki2.test.txt"]

class Wiki2AttentiveSingle(object):

  num_layers = 2
  hidden_size = 1000
  init_scale = 0.05
  dropout_rate = 0.65
  max_grad_norm = 10.0
  num_steps = 35  # for training only
  score_form = attention.SINGLE
  tie_softmax_weights = True

  batch_size = 32
  learning_rate = 1.0
  start_decay = 14
  lr_decay = 1.0/1.15

  vocab_size = 33278 + 1  # add 1 to account for PAD
  data_files = ["data/wikitext-2/wiki2.train.txt",
                "data/wikitext-2/wiki2.valid.txt",
                "data/wikitext-2/wiki2.test.txt"]
