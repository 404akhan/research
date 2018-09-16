# bootstrapped double dqn, https://arxiv.org/abs/1602.04621
# gym evaluation https://gym.openai.com/evaluations/eval_FlVYQ9MTfC8ECNzbPagTA

import itertools
import os
import time
import argparse
import numpy as np
import tensorflow as tf 
import matplotlib.pyplot as plt
import sys
import random
from collections import deque, namedtuple

import gym
from gym.wrappers import SkipWrapper
from ppaquette_gym_doom.wrappers.action_space import ToDiscrete
from scipy.misc import imresize
from gym.core import ObservationWrapper
from gym.spaces.box import Box

class PreprocessImage(ObservationWrapper):
    def __init__(self, env, height=64, width=64, grayscale=True,
                 crop=lambda img: img):
        """A gym wrapper that crops, scales image into the desired shapes and optionally grayscales it."""
        super(PreprocessImage, self).__init__(env)
        self.img_size = (height, width)
        self.grayscale = grayscale
        self.crop = crop

        n_colors = 1 if self.grayscale else 3
        self.observation_space = Box(0.0, 1.0, [n_colors, height, width])

    def _observation(self, img):
        """what happens to the observation"""
        img = self.crop(img)
        img = imresize(img, self.img_size)
        if self.grayscale:
            img = img.mean(-1, keepdims=True)
        img = np.transpose(img, (2, 0, 1))  # reshape from (h,w,colors) to (colors,h,w)
        img = img.astype('float32') / 255.
        return img

def make_env():
    env_spec = gym.spec('ppaquette/DoomBasic-v0')
    env_spec.id = 'DoomBasic-v0'
    env = env_spec.make()
    e = PreprocessImage(SkipWrapper(4)(ToDiscrete("minimal")(env)),
                                 width=80, height=80, grayscale=True)
    return e
env = make_env()

NUM_HEADS = 5
NOOP, SHOOT, RIGHT, LEFT = 0, 1, 2, 3
VALID_ACTIONS = [0, 1, 2, 3]

class StateProcessor():
    def __init__(self):
        pass

    def process(self, sess, state):
        return np.squeeze(state)

class Estimator():
    def __init__(self, scope="estimator"):
        self.scope = scope
        with tf.variable_scope(scope):
            self.X_pl = tf.placeholder(shape=[None, 80, 80, 4], dtype=tf.float32, name="X")
            self.y_pl = tf.placeholder(shape=[NUM_HEADS, None], dtype=tf.float32, name="y")
            self.actions_pl = tf.placeholder(shape=[None], dtype=tf.int32, name="actions")

            self.predictions_arr = []
            self.loss_arr = []
            self.train_op_arr = []

            self._build_shared()

            for agent_index in range(NUM_HEADS):
                self._build_rest(agent_index)

    def _build_shared(self):
        conv1 = tf.contrib.layers.conv2d(
            self.X_pl, 32, 5, 1, activation_fn=tf.nn.relu, padding="VALID")
        pool1 = tf.nn.max_pool(conv1, [1, 3, 3, 1], [1, 2, 2, 1], padding='VALID')

        conv2 = tf.contrib.layers.conv2d(
            pool1, 32, 3, 1, activation_fn=tf.nn.relu, padding="VALID")
        pool2 = tf.nn.max_pool(conv2, [1, 3, 3, 1], [1, 2, 2, 1], padding='VALID')

        conv3 = tf.contrib.layers.conv2d(
            pool2, 64, 2, 1, activation_fn=tf.nn.relu, padding="VALID")
        pool3 = tf.nn.max_pool(conv3, [1, 3, 3, 1], [1, 2, 2, 1], padding='VALID')

        flattened = tf.contrib.layers.flatten(pool3)
        self.fc1 = tf.contrib.layers.fully_connected(flattened, 64)

    def _build_rest(self, agent_index):
        predictions = tf.contrib.layers.fully_connected(self.fc1, len(VALID_ACTIONS), activation_fn=None)
        batch_size = tf.shape(self.X_pl)[0]

        gather_indices = tf.range(batch_size) * tf.shape(predictions)[1] + self.actions_pl
        action_predictions = tf.gather(tf.reshape(predictions, [-1]), gather_indices)

        losses = tf.squared_difference(self.y_pl[agent_index], action_predictions)
        loss = tf.reduce_mean(losses)

        optimizer = tf.train.RMSPropOptimizer(0.001 / NUM_HEADS, 0.99, 0.0, 1e-6, name='RMSProp')
        train_op = optimizer.minimize(loss)

        self.predictions_arr.append(predictions)
        self.loss_arr.append(loss)
        self.train_op_arr.append(train_op)

    def predict(self, sess, s, agent_index):
        return sess.run(self.predictions_arr[agent_index], { self.X_pl: s })

    def predict_all(self, sess, s):
        preds = sess.run(self.predictions_arr, { self.X_pl: s })

        return np.array(preds)

    def update(self, sess, s, a, y):
        feed_dict = { self.X_pl: s, self.y_pl: y, self.actions_pl: a }
        _, loss = sess.run(
            [self.train_op_arr, self.loss_arr],
            feed_dict)
        return np.mean(loss)

    def greedy_action(self, sess, state, agent_index):
        q_values = self.predict(sess, np.expand_dims(state, 0), agent_index)[0]
        best_action = np.argmax(q_values)

        return best_action

def copy_model_parameters(sess, estimator1, estimator2):
    e1_params = [t for t in tf.trainable_variables() if t.name.startswith(estimator1.scope)]
    e1_params = sorted(e1_params, key=lambda v: v.name)
    e2_params = [t for t in tf.trainable_variables() if t.name.startswith(estimator2.scope)]
    e2_params = sorted(e2_params, key=lambda v: v.name)

    update_ops = []
    for e1_v, e2_v in zip(e1_params, e2_params):
        op = e2_v.assign(e1_v)
        update_ops.append(op)

    sess.run(update_ops)

def deep_q_learning(sess,
                    env,
                    q_estimator,
                    target_estimator,
                    state_processor,
                    saver,
                    num_episodes,
                    replay_memory_size=10000,
                    replay_memory_init_size=1000,
                    update_target_estimator_every=500,
                    discount_factor=0.99,
                    batch_size=32):

    Transition = namedtuple("Transition", ["state", "action", "reward", "next_state", "done"])
    replay_memory = []
    total_t = 0

    for i_episode in range(num_episodes):

        state = env.reset()
        state = state_processor.process(sess, state)
        state = np.stack([state] * 4, axis=2)
        loss = None
        total_reward = 0
        actions_tracker = [0, 1, 2, 3]
        agent_index = np.random.randint(NUM_HEADS)

        for t in itertools.count():

            if total_t % update_target_estimator_every == 0:
                copy_model_parameters(sess, q_estimator, target_estimator)
                print("\nCopied model parameters to target network.")
                saver.save(sess, 'docs/dqn-doom-basic', global_step=total_t)

            action = q_estimator.greedy_action(sess, state, agent_index)
            actions_tracker.append(action)
            next_state, reward, done, _ = env.step(VALID_ACTIONS[action])
            next_state = state_processor.process(sess, next_state)
            next_state = np.append(state[:,:,1:], np.expand_dims(next_state, 2), axis=2)

            if len(replay_memory) == replay_memory_size:
                replay_memory.pop(0)

            replay_memory.append(Transition(state, action, reward, next_state, done))   
            total_reward += reward

            if total_t > replay_memory_init_size:
                samples = random.sample(replay_memory, batch_size)
                states_batch, action_batch, reward_batch, next_states_batch, done_batch = map(np.array, zip(*samples))

                indexes = np.array([[i, j] for i in range(NUM_HEADS) for j in range(batch_size)])
                indexes1 = np.reshape(indexes[:, 0], (NUM_HEADS, batch_size))
                indexes2 = np.reshape(indexes[:, 1], (NUM_HEADS, batch_size))

                q_values_next = q_estimator.predict_all(sess, next_states_batch)
                best_actions = np.argmax(q_values_next, axis=2)
                
                q_values_next_target = target_estimator.predict_all(sess, next_states_batch)
                targets_batch = reward_batch + np.invert(done_batch).astype(np.float32) * \
                    discount_factor * q_values_next_target[indexes1, indexes2, best_actions]

                states_batch = np.array(states_batch)
                loss = q_estimator.update(sess, states_batch, action_batch, targets_batch)

            state = next_state
            total_t += 1

            if done:
                print("Step {} ({}) @ Episode {}/{}, loss: {}".format(
                    t, total_t, i_episode + 1, num_episodes, loss), end=", ")
                print('reward %f, steps %d' % (total_reward, t), end=", ")
                counts_action = np.bincount(actions_tracker) - 1
                print('noop %d, shoot %d, right %d, left %d' % \
                    (counts_action[0], counts_action[1], counts_action[2], counts_action[3]))
                break

tf.reset_default_graph()
    
q_estimator = Estimator(scope="q")
target_estimator = Estimator(scope="target_q")

state_processor = StateProcessor()
saver = tf.train.Saver()

if not os.path.exists('docs'):
    os.makedirs('docs')

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    deep_q_learning(sess,
                    env,
                    q_estimator=q_estimator,
                    target_estimator=target_estimator,
                    state_processor=state_processor,
                    saver=saver,
                    num_episodes=1000)
