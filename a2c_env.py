import tensorflow.compat.v1 as tf
import random
import numpy as np
import os


def fc(x, scope, nh, act=tf.nn.relu, init_scale=1.0):
    with tf.variable_scope(scope):
        nin = x.get_shape()[1]

        w = tf.get_variable("w", [nin, nh], initializer=ortho_init(init_scale))
        b = tf.get_variable("b", [nh], initializer=tf.constant_initializer(0.0))
        z = tf.matmul(x, w) + b
        h = act(z)
        return h


def ortho_init(scale=1.0):
    def _ortho_init(shape, dtype, partition_info=None):
        shape = tuple(shape)
        if len(shape) == 2:
            flat_shape = shape
        elif len(shape) == 4:  # assumes NHWC
            flat_shape = (np.prod(shape[:-1]), shape[-1])
        else:
            raise NotImplementedError
        np.random.seed(1)
        a = np.random.normal(0.0, 1.0, flat_shape)
        u, _, v = np.linalg.svd(a, full_matrices=False)
        q = u if u.shape == flat_shape else v  # pick the one with the correct shape
        q = q.reshape(shape)
        return (scale * q[:shape[0], :shape[1]]).astype(np.float32)

    return _ortho_init


class Estimator:
    """ build value network
    """

    def __init__(self,
                 sess,
                 action_dim,
                 state_dim,
                 env,
                 scope="estimator",
                 summaries_dir=None):
        self.sess = sess
        self.action_dim = action_dim
        self.state_dim = state_dim

        self.scope = scope
        self.env = env

        # Writes Tensorboard summaries to disk
        self.summary_writer = None
        with tf.variable_scope(scope):
            value_loss = self._build_value_model()

            with tf.variable_scope("policy"):
                actor_loss, entropy = self._build_mlp_policy()

            self.loss = actor_loss + .5 * value_loss - 10 * entropy

        # Summaries for Tensorboard
        self.summaries = tf.summary.merge([
            tf.summary.scalar("value_loss", self.value_loss),
            tf.summary.scalar("value_output", tf.reduce_mean(self.value_output)),
        ])

        self.policy_summaries = tf.summary.merge([
            tf.summary.scalar("policy_loss", self.policy_loss),
            tf.summary.scalar("adv", tf.reduce_mean(self.tfadv)),
            tf.summary.scalar("entropy", self.entropy),
        ])

        if summaries_dir:
            summary_dir = os.path.join(summaries_dir, "summaries_{}".format(scope))
            if not os.path.exists(summary_dir):
                os.makedirs(summary_dir)
            self.summary_writer = tf.summary.FileWriter(summary_dir)

    def _build_value_model(self):
        trainable = True
        tf.disable_eager_execution()
        self.state = X = tf.placeholder(shape=[None, self.state_dim], dtype=tf.float32, name="X")

        self.y_pl = tf.placeholder(shape=[None], dtype=tf.float32, name="y")

        self.loss_lr = tf.placeholder(tf.float32, None, "learning_rate")

        l1 = fc(X, "l1", 128, act=tf.nn.relu)
        l2 = fc(l1, "l2", 64, act=tf.nn.relu)
        l3 = fc(l2, "l3", 32, act=tf.nn.relu)
        self.value_output = fc(l3, "value_output", 1, act=tf.nn.relu)

        self.value_loss = tf.reduce_mean(tf.squared_difference(self.y_pl, self.value_output))

        self.value_train_op = tf.train.AdamOptimizer(self.loss_lr).minimize(self.value_loss)

        return self.value_loss

    def _build_mlp_policy(self):
        trainable = True
        tf.disable_eager_execution()
        self.policy_state = tf.placeholder(shape=[None, self.state_dim], dtype=tf.float32, name="P")
        self.ACTION = tf.placeholder(shape=[None, self.action_dim], dtype=tf.float32, name="action")
        self.tfadv = tf.placeholder(shape=[None, 1], dtype=tf.float32, name='advantage')

        l1 = fc(self.policy_state, "l1", 128, act=tf.nn.relu)
        l2 = fc(l1, "l2", 64, act=tf.nn.relu)
        l3 = fc(l2, "l3", 32, act=tf.nn.relu)

        self.valid_logits = logits = fc(l3, "logits", self.action_dim,
                                        act=tf.nn.relu) + 1  # avoid valid_logits are all zeros

        self.softmaxprob = tf.nn.softmax(tf.log(self.valid_logits + 1e-8))
        self.logsoftmaxprob = tf.nn.log_softmax(self.softmaxprob)

        self.neglogprob = - self.logsoftmaxprob * self.ACTION
        self.actor_loss = tf.reduce_mean(tf.reduce_sum(self.neglogprob * self.tfadv, axis=1))
        self.entropy = - tf.reduce_mean(self.softmaxprob * self.logsoftmaxprob)

        self.policy_loss = self.actor_loss - 0.01 * self.entropy

        self.policy_train_op = tf.train.AdamOptimizer(self.loss_lr).minimize(self.policy_loss)
        return self.actor_loss, self.entropy

    def predict(self, s):
        value_output = self.sess.run(self.value_output, {self.state: s})
        return value_output

    def action(self, s):
        value_output = self.sess.run(self.value_output, {self.state: s}).flatten()
        action_tuple = []
        valid_prob = []

        # for training policy gradient.
        action_choosen_mat = []
        policy_state = []
        curr_state_value = []
        action_probs = self.sess.run(self.softmaxprob, {self.policy_state: s})

        for fac_id in range(self.env.facsNum):
            action_prob = action_probs[fac_id]
            valid_prob.append(action_prob)  # action probability for state value function

            start_node_id = fac_id
            end_node_id = np.random.choice(self.action_dim, 1, p=action_prob / np.sum(action_prob))
            action_tuple.append((start_node_id, end_node_id))
            policy_state.append(s[fac_id])
            curr_state_value.append(value_output[fac_id])
            temp_a = np.zeros(self.action_dim)
            temp_a[end_node_id] = 1
            action_choosen_mat.append(temp_a)

        return action_tuple, np.stack(valid_prob), \
               np.stack(policy_state), np.stack(action_choosen_mat), curr_state_value

    def update(self, s, a, y, learning_rate, global_step):
        """
        Updates the estimator towards the given targets.

        Args:
          s: State input of shape [batch_size, state_dim]
          a: Chosen actions of shape [batch_size, action_dim], 0, 1 mask
          y: Targets of shape [batch_size]

        Returns:
          The calculated loss on the batch.
        """
        sess = self.sess
        feed_dict = {self.state: s, self.y_pl: y, self.ACTION: a, self.loss_lr: learning_rate}
        summaries, _, loss = sess.run([self.summaries, self.train_op, self.loss], feed_dict)

        if self.summary_writer:
            self.summary_writer.add_summary(summaries, global_step)
        return loss

    def compute_advantage(self, curr_state_value, next_state, node_reward, gamma):
        """for policy network"""
        advantage = []
        # node_reward = node_reward.flatten()
        qvalue_next = self.sess.run(self.value_output, {self.state: next_state}).flatten()

        temp_adv = node_reward + gamma * qvalue_next - curr_state_value
        advantage.append(temp_adv)

        return advantage

    def compute_targets(self, valid_prob, next_state, node_reward, gamma):
        targets = []
        # node_reward = node_reward.flatten()
        qvalue_next = self.sess.run(self.value_output, {self.state: next_state}).flatten()
        target = np.sum(node_reward + gamma * qvalue_next)
        targets.append(target)

        return targets

    def initialization(self, s, y, learning_rate):
        sess = self.sess
        feed_dict = {self.state: s, self.y_pl: y, self.loss_lr: learning_rate}
        _, value_loss = sess.run([self.value_train_op, self.value_loss], feed_dict)
        return value_loss

    def update_policy(self, policy_state, advantage, action_choosen_mat, learning_rate, global_step):
        sess = self.sess
        feed_dict = {self.policy_state: policy_state,
                     self.tfadv: advantage,
                     self.ACTION: action_choosen_mat,
                     # self.neighbor_mask: curr_neighbor_mask,
                     self.loss_lr: learning_rate}
        summaries, _, loss = sess.run([self.policy_summaries, self.policy_train_op, self.policy_loss], feed_dict)

        if self.summary_writer:
            self.summary_writer.add_summary(summaries, global_step)
            self.summary_writer.flush()
        return loss

    def update_value(self, s, y, learning_rate, global_step):
        sess = self.sess

        feed_dict = {self.state: s, self.y_pl: y, self.loss_lr: learning_rate}
        summaries, _, loss = sess.run([self.summaries, self.value_train_op, self.value_loss], feed_dict)

        if self.summary_writer:
            self.summary_writer.add_summary(summaries, global_step)
            self.summary_writer.flush()
        return loss


class stateProcessor:
    """
        Process a raw global state into the states of grids.
    """

    def __init__(self, action_dim, facsNum):
        self.action_dim = action_dim
        self.facsNum = facsNum

    def to_action_mat(self, action_tuple):
        action_mat = np.zeros((self.facsNum, self.action_dim))
        for action in action_tuple:
            start_node_id, action_id = action
            action_mat[start_node_id] = 0
            action_mat[start_node_id, action_id] = 1
        return action_mat

    def to_action_short_mat(self, action_tuple):
        action_mat = np.zeros((self.facsNum, self.action_dim))
        for action in action_tuple:
            start_node_id, action_id = action
            action_mat[start_node_id] = 0
            action_mat[start_node_id, action_id] = 1
        return action_mat


class policyReplayMemory:
    def __init__(self, memory_size, batch_size):
        self.states = []
        self.actions = []
        self.rewards = []  # advantages

        self.batch_size = batch_size
        self.memory_size = memory_size
        self.current = 0
        self.curr_lens = 0

    def add(self, s, a, r):
        if self.curr_lens == 0:
            self.states = s
            self.actions = a
            self.rewards = r
            self.curr_lens = self.states.shape[0]

        elif self.curr_lens <= self.memory_size:
            self.states = np.concatenate((self.states, s), axis=0)
            self.actions = np.concatenate((self.actions, a), axis=0)
            self.rewards = np.concatenate((self.rewards, r), axis=0)
            self.curr_lens = self.states.shape[0]

        else:
            new_sample_lens = s.shape[0]
            index = random.randint(0, self.curr_lens - new_sample_lens)

            self.states[index:(index + new_sample_lens)] = s
            self.actions[index:(index + new_sample_lens)] = a
            self.rewards[index:(index + new_sample_lens)] = r

    def sample(self):
        if self.curr_lens <= self.batch_size:
            return [self.states, self.actions, np.array(self.rewards)]
        indices = random.sample(range(0, self.curr_lens), self.batch_size)
        batch_s = self.states[indices]
        batch_a = self.actions[indices]
        batch_r = self.rewards[indices]
        return [batch_s, batch_a, batch_r]

    def reset(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.curr_lens = 0


class ReplayMemory:
    """ collect the experience and sample a batch for training networks.
        without time ordering
    """

    def __init__(self, memory_size, batch_size):
        self.states = []
        self.next_states = []
        self.actions = []
        self.rewards = []

        self.batch_size = batch_size
        self.memory_size = memory_size
        self.current = 0
        self.curr_lens = 0

    def add(self, s, a, r, next_s):
        if self.curr_lens == 0:
            self.states = s
            self.actions = a
            self.rewards = r
            self.next_states = next_s
            self.curr_lens = self.states.shape[0]

        elif self.curr_lens <= self.memory_size:
            self.states = np.concatenate((self.states, s), axis=0)
            self.next_states = np.concatenate((self.next_states, next_s), axis=0)
            self.actions = np.concatenate((self.actions, a), axis=0)
            self.rewards = np.concatenate((self.rewards, r), axis=0)
            self.curr_lens = self.states.shape[0]

        else:
            new_sample_lens = s.shape[0]
            index = random.randint(0, self.curr_lens - new_sample_lens)

            self.states[index:(index + new_sample_lens)] = s
            self.actions[index:(index + new_sample_lens)] = a
            self.rewards[index:(index + new_sample_lens)] = r
            self.next_states[index:(index + new_sample_lens)] = next_s

    def sample(self):
        if self.curr_lens <= self.batch_size:
            return [self.states, self.actions, self.rewards, self.next_states]
        indices = random.sample(range(0, self.curr_lens), self.batch_size)
        batch_s = self.states[indices]
        batch_a = self.actions[indices]
        batch_r = self.rewards[indices]
        batch_next_s = self.next_states[indices]
        return [batch_s, batch_a, batch_r, batch_next_s]

    def reset(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.curr_lens = 0


class ModelParametersCopier():
    def __init__(self, estimator1, estimator2):
        e1_params = [t for t in tf.trainable_variables() if t.name.startswith(estimator1.scope)]
        e1_params = sorted(e1_params, key=lambda v: v.name)
        e2_params = [t for t in tf.trainable_variables() if t.name.startswith(estimator2.scope)]
        e2_params = sorted(e2_params, key=lambda v: v.name)

        self.update_ops = []
        for e1_v, e2_v in zip(e1_params, e2_params):
            op = e2_v.assign(e1_v)
            self.update_ops.append(op)

    def make(self, sess):
        sess.run(self.update_ops)
