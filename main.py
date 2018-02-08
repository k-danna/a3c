#!/usr/bin/env python

import sys, os
sys.dont_write_bytecode = True
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import gym
import threading
import tensorflow as tf

from a3c import A3C_Worker, A3C_Net

def main():
    
    #training vars
    episode_max = 10
    step_max = 100
    worker_max = 8

    #init environment
    env = gym.make('SpaceInvaders-v0')
    env.seed(42)

    #DEBUG: run with one worker only
    global_net = A3C_Net(env, 'global')
    worker = A3C_Worker(env, tf.train.Coordinator, global_net, 'worker1')
    worker.train(env)

    '''
    #init the global net, worker agents
    coordinator = tf.train.Coordinator()
    global_net = A3C_Net()
    workers = []
    for _ in range(worker_max):
        worker = A3C_worker(coordinator, global_net)
        work = lambda:
        thread = threading.Thread(target=work)
        coord.join(workers)
    '''

if __name__ == '__main__':
    main()

