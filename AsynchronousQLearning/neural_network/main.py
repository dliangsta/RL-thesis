import tensorflow as tf
import numpy as np
import threading
import gym
import os
import sys
from network import Network
from agent import Agent

ENV_NAME = 'FrozenLake8x8-v0'


def main():
    try:
        tf.reset_default_graph()
        sess = tf.Session()
        coord = tf.train.Coordinator()

        checkpoint_dir = 'checkpoint'
        save_path = os.path.join(checkpoint_dir, 'model.ckpt')

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
            print('Directory {} was created'.format(checkpoint_dir))

        n_threads = 4 if len(sys.argv) <= 1 else int(sys.argv[1])
        input_shape = [gym.make(ENV_NAME).observation_space.n]
        output_dim = gym.make(ENV_NAME).action_space.n
        global_network = Network(name='global',
                                 input_shape=input_shape,
                                 output_dim=output_dim)

        lock = threading.Lock()
        thread_list = []
        env_list = []

        for id in range(n_threads + 1):
            env = gym.make(ENV_NAME)

            single_agent = Agent(env=env,
                                 session=sess,
                                 coord=coord,
                                 name='thread_{}'.format(id),
                                 global_network=global_network,
                                 input_shape=input_shape,
                                 output_dim=output_dim,
                                 lock=lock,
                                 training=id < n_threads,
                                 n_threads=n_threads,
                                 thread_list=thread_list)
            thread_list.append(single_agent)
            env_list.append(env)

        restored = False
        # if tf.train.get_checkpoint_state(os.path.dirname(save_path)):
        #     try:
        #         var_list = tf.get_collection(
        #             tf.GraphKeys.GLOBAL_VARIABLES, 'global')
        #         saver = tf.train.Saver(var_list=var_list)
        #         saver.restore(sess, save_path)
        #         print('Model restored to global')
        #         restored = True
        #     except:
        #         pass
        if not restored:
            init = tf.global_variables_initializer()
            sess.run(init)
            print('No model is found')

        for t in thread_list:
            t.start()

        print('All threads started.')
        coord.wait_for_stop()

    except Exception as e:
        print('Exception.')
        print(e)
        # var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'global')
        # saver = tf.train.Saver(var_list=var_list)
        # saver.save(sess, save_path)
        # print('Checkpoint Saved to {}'.format(save_path))

        print('Closing threads')
        coord.request_stop()
        coord.join(thread_list)

        print('Closing environments')
        for env in env_list:
            env.close()

        sess.close()


if __name__ == '__main__':
    main()
