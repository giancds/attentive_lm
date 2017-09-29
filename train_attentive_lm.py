# -*- coding: utf-8 -*-

""" Entry point to train an attentive language model.

    This module defines the parameters for training such model. Check each
    parameter description for more details.

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os
import sys
import time
import numpy as np
import tensorflow as tf
from data_reader import lm_data_producer
from data_reader import read_lm_data, read_vocabulary
from model_ops import get_config
from attentive_lm import AttentiveLM
from utils import create_queue

# pylint: disable=W0613,C0103,C0112
seed = 1701

logging = tf.logging

BASE_DIR = os.path.expanduser("~")   # this will point to the user's home
TRAIN_DIR = "train_lms_new/ptb"


tf.flags.DEFINE_integer("max_epochs", 100,
                        "Max number of epochs to train the models")

tf.flags.DEFINE_boolean("log_tensorboard", True,
                        "Whether or not to log info using tensorboard")

tf.flags.DEFINE_integer("early_stop_patience", 10,
                        "How many training steps to monitor. Set to 0 to ignore.")

tf.flags.DEFINE_string("config", "ptb_single",
                       "Which configuration to use for training.")

tf.flags.DEFINE_string("model_name", "model.ckpt", "Model name")

tf.flags.DEFINE_string("train_dir", os.path.join(BASE_DIR, TRAIN_DIR) + "/",
                       "Train directory")

tf.flags.DEFINE_string("best_models_dir",
                       os.path.join(BASE_DIR, TRAIN_DIR, "best_models"),
                       "Best models directory.")

FLAGS = tf.flags.FLAGS


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


def train_lm():
  """ Loop that will perform the actual LM optimization. """

  print("\n\nTraining started at {} - {}\n\n".format(
    time.strftime("%d/%m/%Y"), time.strftime("%H:%M:%S")
  ))

  config = get_config(FLAGS.config)

  vocabulary = read_vocabulary(config.data_files, config.vocab_size)
  train_data, valid_data, test_data = read_lm_data(config.data_files,
                                                   vocabulary)

  with tf.Graph().as_default() as graph:

    # define a default initializer for the model
    initializer = tf.random_uniform_initializer(
      -config.init_scale, config.init_scale, seed=seed, dtype=tf.float32)

    # model for training
    print("\nBuilding Model for training...")
    with tf.name_scope("Train"):
      with tf.variable_scope("Model", reuse=None, initializer=initializer):

        train_data_producer = lm_data_producer(train_data,
                                               config.batch_size,
                                               config.num_steps)

        train_queue = tf.FIFOQueue(
          capacity=len(train_data_producer[0]), dtypes=[tf.int32, tf.int32],
          shapes=[[config.num_steps]] * 2)

        train_inputs = tf.convert_to_tensor(train_data_producer[0],
                                            dtype=tf.int32)
        train_targets = tf.convert_to_tensor(train_data_producer[1],
                                             dtype=tf.int32)
        enqueue_op_train = train_queue.enqueue_many([train_inputs,
                                                     train_targets])

        qr_train = tf.train.QueueRunner(train_queue, [enqueue_op_train] * 2)
        tf.train.add_queue_runner(qr_train)

        mtrain = AttentiveLM(is_training=True,
                             params=config,
                             batch_size=config.batch_size,
                             num_steps=config.num_steps,
                             queue=train_queue,
                             keep_attention_weights=False,
                             log_tensorboard=FLAGS.log_tensorboard)
      print("Batch size: {:d}".format(mtrain.batch_size))
      print("# of steps: {:d}".format(mtrain.num_steps))

    # model for validation
    print("\nBuilding Model for validation...")
    with tf.name_scope("Valid"):
      with tf.variable_scope("Model", reuse=True, initializer=initializer):

        num_valid_steps = max([len(sample) for sample in valid_data])
        valid_data_producer = lm_data_producer(
          valid_data, config.batch_size, num_valid_steps)

        valid_queue = tf.FIFOQueue(
          capacity=len(valid_data_producer[0]), dtypes=[tf.int32, tf.int32],
          shapes=[[num_valid_steps]] * 2)

        valid_inputs = tf.convert_to_tensor(
          valid_data_producer[0], dtype=tf.int32)
        valid_targets = tf.convert_to_tensor(
          valid_data_producer[1], dtype=tf.int32)
        enqueue_op_valid = valid_queue.enqueue_many(
          [valid_inputs, valid_targets])

        qr_valid = tf.train.QueueRunner(valid_queue, [enqueue_op_valid] * 2)
        tf.train.add_queue_runner(qr_valid)

        mvalid = AttentiveLM(
          is_training=False, params=config, batch_size=config.batch_size,
          num_steps=num_valid_steps, queue=valid_queue,
          keep_attention_weights=False)
        print("# of validation steps: {:d}".format(num_valid_steps))

    # configure the session
    proto_config = tf.ConfigProto(allow_soft_placement=True,
                                  log_device_placement=False)

    # save training and best models
    saver = tf.train.Saver(max_to_keep=3)
    saver_best = tf.train.Saver(max_to_keep=1)

    supervisor = tf.train.Supervisor(logdir=FLAGS.train_dir,
                                     saver=saver, save_model_secs=0)

    with supervisor.managed_session(config=proto_config) as session:

      # supervisor.

      best_valid_ppx = np.inf
      estop_counter = 0

      for epoch in range(FLAGS.max_epochs):

        lr_decay = config.lr_decay **  max(epoch - config.start_decay, 0.0)
        mtrain.assign_lr(session, config.learning_rate * lr_decay)

        # print info
        print("\nEpoch: {:d} - Learning rate: {:e}".format(
          epoch, session.run(mtrain.lr_rate)))

        _ = run_epoch(session, mtrain, train_data, is_train=True)

        # Save checkpoint
        print("\nSaving current model...")
        checkpoint_path = os.path.join(FLAGS.train_dir, FLAGS.model_name)
        saver.save(session, checkpoint_path, global_step=mtrain.global_step)

        print("\nRunning validation...")
        valid_ppx = run_epoch(session, mvalid, valid_data, is_train=False)
        print("Epoch {:d}: - Valid Perplexity: {:.8f}".format(epoch, valid_ppx))

        # check early stop
        if FLAGS.early_stop_patience > 0:

          if best_valid_ppx > valid_ppx:
            best_valid_ppx = valid_ppx
            estop_counter = 0
            print('\nSaving the best model so far...')
            model_name = FLAGS.model_name + '-best'
            best_model_path = os.path.join(FLAGS.best_models_dir, model_name)
            saver_best.save(session, best_model_path,
                            global_step=mtrain.global_step)
          else:
            estop_counter += 1

          print("\n\tbest valid. ppx: {:.8f}".format(best_valid_ppx))
          print("early stop patience: {:d} - max {:d}\n".format(
            estop_counter, FLAGS.early_stop_patience))

          if estop_counter >= FLAGS.early_stop_patience:
            print('\nEARLY STOP!\n')
            supervisor.request_stop()
            supervisor.coord.join(threads)
            break

  # when we ran the right number of epochs or we reached early stop we
  # finish training
  print("\n\nTraining finished at {} - {}\n\n".format(
    time.strftime("%d/%m/%Y"), time.strftime("%H:%M:%S")
  ))

  with tf.Graph().as_default() as test_graph:

    # model for testing
    print("\n\nBuilding Model for testing...\n")
    with tf.name_scope("Test"):
      with tf.variable_scope("Model", reuse=None, initializer=initializer):
        num_test_steps = max([len(sample) for sample in test_data])

        test_data_producer = lm_data_producer(
          test_data, config.batch_size, num_test_steps)

        test_queue = tf.FIFOQueue(
          capacity=len(test_data_producer[0]), dtypes=[tf.int32, tf.int32],
          shapes=[[num_test_steps]] * 2)

        test_inputs = tf.convert_to_tensor(
          test_data_producer[0], dtype=tf.int32)
        test_targets = tf.convert_to_tensor(
          test_data_producer[1], dtype=tf.int32)
        enqueue_op_test = test_queue.enqueue_many(
          [test_inputs, test_targets])

        qr_test = tf.train.QueueRunner(test_queue, [enqueue_op_test] * 2)
        tf.train.add_queue_runner(qr_test)
        mtest = AttentiveLM(is_training=False,
                            params=config,
                            batch_size=config.batch_size,
                            num_steps=num_test_steps,
                            keep_attention_weights=True)
        print("# of test steps: {:d}".format(num_test_steps))

    saver_test = tf.train.Saver(max_to_keep=1)
    test_supervisor = tf.train.Supervisor(
      logdir=FLAGS.best_models_dir, summary_writer=None,
      saver=saver_test, save_model_secs=0)

    with test_supervisor.managed_session(config=proto_config) as test_session:
      # eval on test
      print("\nRunning test...")
      test_ppx = run_epoch(
        test_session, mtest, test_data,
        is_train=False, plot_attention_weights=True)
      print("Test Perplexity: {:.8f}".format(test_ppx))

      test_supervisor.request_stop()
      test_supervisor.coord.join()

  sys.stdout.flush()


def run_epoch(session,
              model,
              dataset,
              is_train=False,
              plot_attention_weights=False):
  """ Run either a train or eval epoch on the given dataset. """
  assert dataset is not None
  n_words = len([word for sample in dataset for word in sample if word > 0])
  epoch_size = int(math.ceil(len(dataset) / model.batch_size))
  # producer = lm_data_producer(dataset, model.batch_size, model.num_steps)

  fetches = {"step_cost": model.batch_loss, "niters": model.nwords}
  if is_train:
    fetches["eval_op"] = model.train_op
  if plot_attention_weights:
    fetches["weights"] = model.attention_weights

  costs = 0.0
  iters = 0
  start_time = time.time()
  # for step, (x, y) in enumerate(producer):
  for step in range(epoch_size):
    step_time = time.time()
    vals = session.run(fetches, {})
    step_cost = vals["step_cost"]
    costs += step_cost
    # iters += np.sum(x > 0)
    iters += vals["niters"]

    # print information regarding the current training process
    if is_train:
      if step % (epoch_size // 20) == 10:
        print("{:.3f} - aprox. loss {:.8f} - approx. speed: {:.0f} wps".format(
          step * 1.0 / epoch_size, costs / (step + 1),
          iters / (time.time() - start_time)))
    # print information regarding the current training process
    else:
      if step % (epoch_size // 10) == 5:
        print("{:.3f} - approx. speed: {:.0f} wps".format(
          step * 1.0 / epoch_size, iters / (time.time() - start_time)))

  return np.exp(costs / n_words)


def main(unused_args):
  """ Start the training process. """

  if not os.path.exists(FLAGS.train_dir):
    os.makedirs(FLAGS.train_dir)

  if not os.path.exists(FLAGS.best_models_dir):
    os.makedirs(FLAGS.best_models_dir)

  train_lm()


if __name__ == "__main__":
  tf.app.run()
