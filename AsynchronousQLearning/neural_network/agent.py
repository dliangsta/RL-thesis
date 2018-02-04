import tensorflow as tf
import numpy as np
import threading
import gym
import sys
from network import Network
from time import time

import tensorflow as tf

class Agent(threading.Thread):

    def __init__(self, session, env, coord, name, global_network, input_shape, output_dim, lock, training, n_threads, thread_list):
        super(Agent, self).__init__()
        self.local = Network(name, input_shape, output_dim)
        self.global_to_local = self.copy_src_to_dst('global', name)
        self.global_network = global_network
        self.input_shape = input_shape
        self.output_dim = output_dim
        self.env = env
        self.sess = session
        self.coord = coord
        self.name = name
        self.episode = 0
        self.rewards = []
        self.epsilon = 1.
        self.end_epsilon = .1
        self.exploration_steps = 10000
        self.lock = lock
        self.training = training
        self.n_threads = n_threads
        self.thread_list = thread_list
        self.start_time = time()
        if training:
            self.local = global_network

    def run(self):
        while not self.coord.should_stop():
            self.episode += 1
            self.play_episode()
            self.epsilon = max(self.end_epsilon, 1. - self.n_threads * self.episode * (1. - self.end_epsilon) / self.exploration_steps)

            if self.episode % 1 == 0 and self.training:
                self.sess.run(self.global_to_local)

            if self.episode % 100 == 0 and not self.training:
                self.print_score()

            if not self.training:
                self.play_episode()
                if len(self.rewards) >= 100:
                    if np.mean(self.rewards[-100:]) >= .99:
                        print('Won!')
                        self.print_score()
                        self.coord.request_stop()
                        exit(0)

            elif self.episode % 100 == 0:
                self.print_episode()

    def train(self, experiences):
        states = []
        target_qs = []
        self.lock.acquire()
        np.random.shuffle(experiences)
        for i, experience in enumerate(experiences):
            states.append(np.reshape(experience[0][0], self.input_shape))
            s1 = np.reshape(experience[0][0], [-1, *self.input_shape])
            s2 = np.reshape(experience[0][1], [-1, *self.input_shape])
            action = experience[1]
            reward = experience[2]
            q_out = np.array(self.sess.run(self.local.q_out, {self.local.input: s1}))
            next_q_out = np.array(self.sess.run(self.local.q_out, {self.local.input: s2}))
            estimated_future_reward = np.max(next_q_out)
            target_q = q_out
            target_q[0][action] = reward + .99 * estimated_future_reward
            target_q = np.squeeze(target_q)
            target_qs.append(target_q)
        states = np.reshape(states, [-1, *self.input_shape])

        gradients = self.sess.run(self.local.gradients, feed_dict={
            self.local.input: states,
            self.local.target_q: target_qs,
        })

        feed = []
        for (grad, _), (placeholder, _) in zip(gradients, self.global_network.gradients_placeholders):
            feed.append((placeholder, grad))

        self.sess.run(self.global_network.apply_gradients, dict(feed))
        self.lock.release()

    def play_episode(self):
        self.sess.run(self.global_to_local)

        states = []
        actions = []
        rewards = []
        total_reward = 0

        s = self.env.reset()
        s = self.onehot(self.input_shape, s, 1)

        done = False
        while not done:
            a = self.choose_action(s)
            s2, r, done, _ = self.env.step(a)
            s2 = self.onehot(self.input_shape, s2, 1)
            
            states.append(s)
            actions.append(a)
            rewards.append(r)
            
            total_reward += r
            s = s2

        if self.training:
            states.append(s2)
            experiences = [((states[i], states[i+1]), actions[i], rewards[i]) for i in range(len(states) - 1)]
            self.train(experiences)
        else:
            self.rewards.append(total_reward)

    def choose_action(self, state):
        if np.random.random() < self.epsilon and self.training:
            return np.random.randint(self.output_dim)
        else:
            state = np.reshape(state, [-1, *self.input_shape])
            return np.argmax(self.sess.run(self.local.q_out, feed_dict={
                self.local.input: state
            }))

    def copy_src_to_dst(self, from_scope, to_scope):
        from_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, from_scope)
        to_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, to_scope)

        op_holder = []
        for from_var, to_var in zip(from_vars, to_vars):
            op_holder.append(to_var.assign(from_var))
        return op_holder

    def onehot(self, shape, s, value):
        return np.identity(*self.input_shape)[s:s+1] * value

    def print_score(self):
        if len(self.rewards) >= 100:
            message = 'Test: \tall rewards=%.2f \trecent rewards=%.2f \tavg episode: %.2f \telapsed time: %f)' % (
                sum(self.rewards) / len(self.rewards), sum(self.rewards[-100:]) / 100, np.mean([t.episode for t in self.thread_list[:-1]]), time() - self.start_time)
            print(message)

    def print_episode(self):
        message = '\t\t\t\t\t\t\t\t\t\t\t\t\t\tAgent(name=%s \tepisode=%d \tepsilon=%.2f)' % (
            self.name, self.episode, self.epsilon)
        print(message)