import gym
from gym.wrappers import Monitor
import itertools
import numpy as np
import os
import random
import sys
import tensorflow as tf

from collections import deque, namedtuple

env = gym.envs.make("SeaquestDeterministic-v4")

num_actions = env.action_space.n
gamma = 0.99


class StateProcessor():
    def __init__(self):
        with tf.variable_scope("state_processor"):
            self.input_state = tf.placeholder(shape=[210, 160, 3], dtype=tf.uint8)
            self.output = tf.image.rgb_to_grayscale(self.input_state)
            self.output = tf.image.crop_to_bounding_box(self.output, 34, 0, 160, 160)
            self.output = tf.image.resize_images(
                self.output, [84, 84], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
            self.output = tf.squeeze(self.output)

    def process(self, sess, state):
        return sess.run(self.output, { self.input_state: state })


class Estimator():
    def __init__(self, scope="estimator", summaries_dir=None, num_atoms=51):
        self.scope = scope
        self.summary_writer = None

        self.gamma = gamma
        self.N = num_atoms
        self.Vmax = 10
        self.Vmin = -self.Vmax
        self.dz = (self.Vmax - self.Vmin) / (self.N - 1)
        self.z = np.linspace(self.Vmin, self.Vmax, self.N)

        with tf.variable_scope(scope):
            # Build the graph
            self._build_model()
            if summaries_dir:
                summary_dir = os.path.join(summaries_dir, "summaries_{}".format(scope))
                if not os.path.exists(summary_dir):
                    os.makedirs(summary_dir)
                self.summary_writer = tf.summary.FileWriter(summary_dir)

    def _build_model(self):
        # Placeholders for our input
        # Our input are 4 RGB frames of shape 160, 160 each
        self.X_pl = tf.placeholder(shape=[None, 84, 84, 4], dtype=tf.uint8, name="X")
        # The TD target value
        self.target_prob = tf.placeholder(shape=[None, self.N], dtype=tf.float32, name="y")
        # Integer id of which action was selected
        self.actions_pl = tf.placeholder(shape=[None], dtype=tf.int32, name="actions")

        X = tf.to_float(self.X_pl) / 255.0
        batch_size = tf.shape(self.X_pl)[0]

        # Three convolutional layers
        conv1 = tf.contrib.layers.conv2d(
            X, 32, 8, 4, activation_fn=tf.nn.relu)
        conv2 = tf.contrib.layers.conv2d(
            conv1, 64, 4, 2, activation_fn=tf.nn.relu)
        conv3 = tf.contrib.layers.conv2d(
            conv2, 64, 3, 1, activation_fn=tf.nn.relu)

        # Fully connected layers
        flattened = tf.contrib.layers.flatten(conv3)
        fc1 = tf.contrib.layers.fully_connected(flattened, 512)

        self.logits = tf.contrib.layers.fully_connected(fc1, num_actions * self.N, activation_fn=None)
        self.logits = tf.reshape(self.logits, (-1, num_actions, self.N))
        
        self.probs = tf.nn.softmax(self.logits)
        self.q_values = tf.reduce_sum(self.probs * self.z, axis=2)

        # Get the predictions for the chosen actions only
        gather_indices = tf.range(batch_size) * num_actions + self.actions_pl
        self.actions_probs = tf.gather(tf.reshape(self.probs, [-1, self.N]), gather_indices)

        # Calcualte the loss
        self.loss = - tf.reduce_sum(self.target_prob * tf.log(self.actions_probs))

        # Optimizer Parameters from original paper
        self.optimizer = tf.train.RMSPropOptimizer(0.00025, 0.99, 0.0, 1e-6)
        self.train_op = self.optimizer.minimize(self.loss, global_step=tf.contrib.framework.get_global_step())

        # Summaries for Tensorboard
        self.summaries = tf.summary.merge([
            tf.summary.scalar("loss", self.loss),
            tf.summary.histogram("q_values_hist", self.q_values),
            tf.summary.scalar("max_q_value", tf.reduce_max(self.q_values))
        ])

    def predict(self, sess, s):
        return sess.run(self.q_values, { self.X_pl: s })

    def get_target_distr(self, sess, x2, rewards):
        # rewards           | (bsize)
        # x1 - cur_state    | (bsize x 84 x 84 x 4)
        # x2 - next_state   | (bsize x 84 x 84 x 4)
        # return            | (bsize x N)
        q_values, probs = sess.run([self.q_values, self.probs], { self.X_pl: x2 }) # (bsize, num_actions), (bsize, num_actions, num_atoms)
        a_star = np.argmax(q_values, axis=1) # (bsize)

        bsize = a_star.shape[0]
        probs_star = probs[range(bsize), a_star] # (bsize, N)

        r_rep = np.tile(np.expand_dims(rewards, 1), (1, self.N))
        Tz = np.clip(r_rep + self.gamma * self.z, self.Vmin, self.Vmax)
        # Tz | bsize x N
        b = (Tz - self.Vmin) / self.dz # b[batch, i] belongs [0, N-1]
        l, u = np.floor(b).astype(np.int32), np.ceil(b).astype(np.int32)

        m = np.zeros((bsize, self.N), dtype=np.float32)

        for i in range(self.N):
            m[range(bsize), l[:, i]] += probs_star[:, i] * (u[:, i] - b[:, i])
            m[range(bsize), u[:, i]] += probs_star[:, i] * (b[:, i] - l[:, i])

        return m

    def update(self, sess, x1, a, targ_prob):
        feed_dict = {
            self.X_pl: x1,
            self.actions_pl: a,
            self.target_prob: targ_prob
        }
        summaries, global_step, _, loss = sess.run(
            [self.summaries, tf.contrib.framework.get_global_step(), self.train_op, self.loss],
            feed_dict)
        if self.summary_writer:
            self.summary_writer.add_summary(summaries, global_step)
        return loss


def make_epsilon_greedy_policy(estimator, nA):
    def policy_fn(sess, observation, epsilon):
        A = np.ones(nA, dtype=float) * epsilon / nA
        q_values = estimator.predict(sess, np.expand_dims(observation, 0))[0]
        best_action = np.argmax(q_values)
        A[best_action] += (1.0 - epsilon)
        return A
    return policy_fn


def deep_q_learning(sess,
                    env,
                    q_estimator,
                    state_processor,
                    num_episodes,
                    experiment_dir,
                    replay_memory_size=500000,
                    replay_memory_init_size=50000,
                    discount_factor=0.99,
                    epsilon_start=1.0,
                    epsilon_end=0.1,
                    epsilon_decay_steps=500000,
                    batch_size=32,
                    record_video_every=50):

    Transition = namedtuple("Transition", ["state", "action", "reward", "next_state", "done"])

    # The replay memory
    replay_memory = []
    
    # Keeps track of useful statistics
    ep_rewards = np.zeros(num_episodes)
    
    # Create directories for checkpoints and summaries
    checkpoint_dir = os.path.join(experiment_dir, "checkpoints")
    checkpoint_path = os.path.join(checkpoint_dir, "model")
    monitor_path = os.path.join(experiment_dir, "monitor")
    
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    if not os.path.exists(monitor_path):
        os.makedirs(monitor_path)

    saver = tf.train.Saver()
    # Load a previous checkpoint if we find one
    latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
    if latest_checkpoint:
        print("Loading model checkpoint {}...\n".format(latest_checkpoint))
        saver.restore(sess, latest_checkpoint)
    
    # Get the current time step
    total_t = sess.run(tf.contrib.framework.get_global_step())

    # The epsilon decay schedule
    epsilons = np.linspace(epsilon_start, epsilon_end, epsilon_decay_steps)

    # The policy we're following
    policy = make_epsilon_greedy_policy(q_estimator, num_actions)

    # Populate the replay memory with initial experience
    print("Populating replay memory...")
    state = env.reset()
    state = state_processor.process(sess, state)
    state = np.stack([state] * 4, axis=2)
    for i in range(replay_memory_init_size):
        action_probs = policy(sess, state, epsilons[min(total_t, epsilon_decay_steps-1)])
        action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
        next_state, reward, done, _ = env.step(action)
        next_state = state_processor.process(sess, next_state)
        next_state = np.append(state[:,:,1:], np.expand_dims(next_state, 2), axis=2)
        replay_memory.append(Transition(state, action, reward, next_state, done))
        if done:
            state = env.reset()
            state = state_processor.process(sess, state)
            state = np.stack([state] * 4, axis=2)
        else:
            state = next_state


    # Record videos
    # Add env Monitor wrapper
    # env = Monitor(env, directory=monitor_path, video_callable=lambda count: count % record_video_every == 0, resume=True)

    for i_episode in range(num_episodes):

        # Save the current checkpoint
        saver.save(tf.get_default_session(), checkpoint_path)

        # Reset the environment
        state = env.reset()
        state = state_processor.process(sess, state)
        state = np.stack([state] * 4, axis=2)
        losses = []

        # One step in the environment
        for t in itertools.count():

            # Epsilon for this time step
            epsilon = epsilons[min(total_t, epsilon_decay_steps-1)]

            # Take a step
            action_probs = policy(sess, state, epsilon)
            action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
            next_state, reward, done, _ = env.step(action)
            next_state = state_processor.process(sess, next_state)
            next_state = np.append(state[:,:,1:], np.expand_dims(next_state, 2), axis=2)

            # If our replay memory is full, pop the first element
            if len(replay_memory) == replay_memory_size:
                replay_memory.pop(0)

            # Save transition to replay memory
            replay_memory.append(Transition(state, action, reward, next_state, done))   

            # Update statistics
            ep_rewards[i_episode] += reward

            # Sample a minibatch from the replay memory
            samples = random.sample(replay_memory, batch_size)
            states_batch, action_batch, reward_batch, next_states_batch, done_batch = map(np.array, zip(*samples))

            # Calculate q values and targets
            targ_prob = q_estimator.get_target_distr(sess, next_states_batch, reward_batch)

            # Perform gradient descent update
            loss = q_estimator.update(sess, states_batch, action_batch, targ_prob)
            losses.append(loss)

            state = next_state
            total_t += 1

            if done:
                print("Step {} ({}) @ Episode {}/{}, loss: {}".format(
                    t, total_t, i_episode + 1, num_episodes, np.mean(losses)), end=", ")
                print('reward %f, eps %f' % (ep_rewards[i_episode], epsilon))
                break

        # Add summaries to tensorboard
        episode_summary = tf.Summary()
        episode_summary.value.add(simple_value=epsilon, tag="episode/epsilon")
        episode_summary.value.add(simple_value=ep_rewards[i_episode], tag="episode/reward")
        episode_summary.value.add(simple_value=t, tag="episode/length")
        q_estimator.summary_writer.add_summary(episode_summary, i_episode)
        q_estimator.summary_writer.flush()


tf.reset_default_graph()

# Where we save our checkpoints and graphs
experiment_dir = os.path.abspath("./experiments/{}".format(env.spec.id))

# Create a glboal step variable
global_step = tf.Variable(0, name='global_step', trainable=False)
    
# Create estimators
q_estimator = Estimator(scope="q_estimator", summaries_dir=experiment_dir)

# State processor
state_processor = StateProcessor()

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.1)
tf_config = tf.ConfigProto(gpu_options=gpu_options)

# Run it!
with tf.Session(config=tf_config) as sess:
    sess.run(tf.global_variables_initializer())
    deep_q_learning(sess,
                    env,
                    q_estimator=q_estimator,
                    state_processor=state_processor,
                    experiment_dir=experiment_dir,
                    num_episodes=10000,
                    replay_memory_size=500000,
                    replay_memory_init_size=50000,
                    epsilon_start=1.0,
                    epsilon_end=0.1,
                    epsilon_decay_steps=500000,
                    discount_factor=gamma,
                    batch_size=32)
