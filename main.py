#!/usr/bin/env python3

import sys, os
from threading import Thread
sys.dont_write_bytecode = True
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import gym
import threading
import numpy as np
import tensorflow as tf

from a3c import A3C_Worker, A3C_Net

def main():
    
    #training vars
    worker_num = 4
    global_train_steps = 3000
    test = True
    env_name = 'CartPole-v0'
    #env_name = 'Pendulum-v0'
    #env_name = 'Pong-v0'
    #env_name = 'SpaceInvaders-v0'
    #env_name = 'FetchReach-v0'

    #env = gym.wrappers.Monitor(env, directory='./logs', force=True, 
    #        video_callable=lambda n: n == episode_max)

    #init global, workers
    workers = []
    coordinator = tf.train.Coordinator()
    sess = tf.Session()
    env = gym.make(env_name)
    global_net = A3C_Net(env, 'global', sess)
    for i in range(worker_num):
        name = 'worker_%s' % i
        local_net = A3C_Net(env, name, sess)
        workers.append(A3C_Worker(coordinator, global_net, local_net,
                name))

    print ('training for %s batches\n' % global_train_steps)

    #start training asynchronously
    threads = []
    for worker in workers:
        #lambda so we can avoid passing args to thread call
        env = gym.make(env_name)
        env.seed(42)
        work = lambda: worker.train(env, global_train_steps)
        thread = Thread(target=work)
        #thread.daemon = True
        thread.start()
        #thread.join() #block main until thread terminates
        #worker.train(env, steps=step_max)
        threads.append(thread)

    #apply gradients while threads do work
    global_net.update_loop(global_train_steps)
    coordinator.join(threads) #wait for threads to finish

    #output training info
    print
    for worker in workers:
        print ('[*] %s trained for: %s' % (worker.scope, 
                worker.local_net.get_step()))
    print ('[*] global trained for: %s' % global_net.get_step())

    #worker2 = A3C_Worker(env, tf.train.Coordinator, global_net, 'worker2')
    #worker1.train(env)
    #worker2.train(env)

    #test
    if test:
        for worker in workers:
            worker.test(gym.make(env_name))

    env.close()

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

