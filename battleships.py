import gym
import coding_challenge
import numpy as np
import tensorflow as tf
import random

using_batch = False
load = True
batch_size = 5
win_reward = 3
eps = 0.3
repetition_cost = -5


class DqnEpisode:
    def __init__(self, input_dim, output_dim, hidden_dim, conv_filters, lr=1e-4, gamma=0.9, win_reward=win_reward, eps=eps):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.conv_filters = conv_filters
        self.gamma = gamma
        self.win_reward = win_reward
        self.eps = eps
        x = self.x = tf.placeholder(tf.float32, shape=[None, input_dim])
        self.r = tf.placeholder(tf.float32, shape=[None])
        q = self.q = self.net(x)  # [None, output_dim]
        q_double = self.q_double = self.net(x)
        self.a_ph = tf.placeholder(tf.int32, shape=[None])
        self.greedy_a_double = tf.argmax(q_double, 1)
        # self.greedy_a = tf.argmax(self.q, 1)
        self.q_a = tf.gather_nd(q, tf.transpose([
            tf.range(tf.shape(q)[0]), self.a_ph]))
        self.q_a_double = tf.gather_nd(q_double, tf.transpose([
            tf.range(tf.shape(q)[0]), self.a_ph]))
        self.s_history = []
        self.a_history = []
        self.running_a = []
        loss = tf.losses.mean_squared_error(self.r, self.q_a)
        loss_double = tf.losses.mean_squared_error(self.r, self.q_a_double)
        self.train_op = tf.train.AdamOptimizer(lr).minimize(loss)
        self.train_op_double = tf.train.AdamOptimizer(lr).minimize(loss_double)
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()
        tf.get_default_graph().finalize()

    def net(self, x):
        conv = tf.layers.conv2d(tf.reshape(
            x, [-1, 10, 10, 1]), self.conv_filters, [3, 3],
            activation='relu', padding='valid')
        conv = tf.layers.conv2d(conv, self.conv_filters, [3, 3],
                                activation='relu', padding='valid')
        flat = tf.layers.flatten(conv)
        x = tf.layers.dense(x, self.hidden_dim, 'relu')
        x = tf.layers.dense(x, self.output_dim, 'linear') + \
            tf.layers.dense(flat, self.output_dim, 'linear')
        return x

    def act(self, s):
        self.s_history.append(s.copy())
        s = s.reshape([1, self.input_dim])
        a = self.sess.run(self.greedy_a_double, feed_dict={self.x: s})[0]
        if np.random.uniform() < self.eps:
            # if a in self.a_history:
            remain = [i for i in range(100) if i not in self.running_a]
            a = random.choice(remain)
        self.a_history.append(a)
        return a

    def train(self, r):
        q_a_next, greedy_a = self.sess.run([self.q_a, self.greedy_a_double], feed_dict={
            self.x: self.s_history,
            self.a_ph: self.a_history})
        for i in range(r.shape[0]):
            if np.abs(r[i]) == self.win_reward:
                continue
            r[i] += self.gamma * q_a_next[i+1]
        # r = (r - np.mean(r)) / np.std(r)
        self.sess.run([self.train_op, self.train_op_double], feed_dict={
                      self.x: self.s_history, self.r: r, self.a_ph: self.a_history})
        self.s_history.clear()
        self.a_history.clear()

    def save(self):
        self.saver.save(self.sess, "save/model.ckpt")

    def load(self):
        self.saver.restore(self.sess, "save/model.ckpt")


class DqnStep(DqnEpisode):
    def act_and_train(self, r, state):
        assert len(self.a_history) == 1
        assert len(self.s_history) == 1
        s = state.reshape([1, self.input_dim])
        a = self.sess.run(
            self.greedy_a_double, feed_dict={self.x: s})[0]
        qs = self.sess.run(self.q, feed_dict={self.x: s})
        q_a_next = self.sess.run(self.q_a, feed_dict={
                                 self.x: s, self.a_ph: [a]})
        if np.random.uniform() < self.eps:
            remain = [i for i in range(100) if i not in self.running_a]
            a = random.choice(remain)
        q_target = r + self.gamma * q_a_next
        self.sess.run([self.train_op, self.train_op_double], feed_dict={self.x: self.s_history, self.r:
                                                q_target, self.a_ph: self.a_history})
        self.s_history.clear()
        self.a_history.clear()
        self.a_history.append(a)
        self.s_history.append(state.copy())
        return a

    def train_last_step(self, r):
        self.sess.run(self.train_op, feed_dict={
            self.x: self.s_history, self.r: [r], self.a_ph: self.a_history})
        self.s_history.clear()
        self.a_history.clear()


def main():
    episode_num = 0
    env = gym.make('Battleship-v0')
    s = env.reset().reshape([-1])
    if using_batch:
        policy = DqnEpisode(input_dim=100, output_dim=100,
                              hidden_dim=256, conv_filters=64)
    else:
        policy = DqnStep(input_dim=100, output_dim=100,
                           hidden_dim=256, conv_filters=64)
    if load:
        policy.load()
    rewards = []
    win_rate = 0
    saved_win_rate = []
    avg_len = 0
    saved_avg_len = []
    while True:
        if using_batch or len(policy.running_a) == 0:
            a = policy.act(s)
        else:
            a = policy.act_and_train(r, s)
        action = np.array([int(a/10), a % 10])/10
        s, r, done, info = env.step(action)
        s = s.reshape([-1])
        avg_len += 1
        if a in policy.running_a:
            r = repetition_cost
        policy.running_a.append(a)
        if info['game_message'] == 'You win!':
            r = win_reward
            win_rate += 1/batch_size
        if info['game_message'] == 'You loose!':
            r = - win_reward
        if using_batch:
            rewards.append(r)
        if done:
            episode_num += 1
            s = env.reset().reshape([-1])
            policy.running_a.clear()
            if not using_batch:
                policy.train_last_step(r)
            if episode_num % batch_size == 0 and episode_num > 0:
                avg_len /= batch_size
                print("episode {}, win rate {}, average game length {}.".format(
                    episode_num, win_rate, avg_len))
                saved_win_rate.append(win_rate)
                saved_avg_len.append(avg_len)
                win_rate = 0
                avg_len = 0
                if using_batch:
                    policy.train(np.array(rewards))
                    rewards.clear()
            policy.save()
            np.save("save/winrate", np.array(saved_win_rate))
            np.save("save/avglen", np.array(saved_avg_len))


if __name__ == "__main__":
    main()
