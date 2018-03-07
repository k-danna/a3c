
import sys
from queue import Queue
import random
import tensorflow as tf
import numpy as np


#FIXME: move these to the net
#some quick wrapper methods for the state
    #easily enables handling a large variety of inputs

def process_state(state):

    #pad state if 1d with odd number of observations
    dims = len(state.shape)
    state = np.asarray(state, dtype=np.float32)

    #handle rgb inputs
    if dims == 3:
        #convert rgb to greyscale
        r, g, b = state[:, :, 0], state[:, :, 1], state[:, :, 2]
        state = 0.2989 * r + 0.5870 * g + 0.1140 * b
        state = state.reshape(state.shape + (1,))
    
    #handle list of observations
    elif dims == 1:
        #convert to a 2d square 'image'
        if not state.shape[0] % 2 == 0:
            state = state.append(0.0) #pad
        w = int(state.shape[0] / 2)
        state = state.reshape((w, w, 1))
    
    #error for any unsupported sizes
    elif dims < 1 or dims > 3:
        print('error: state size unsupported: %s' % dims)
        sys.exit(1)

    #downsample to ?x?
    #state = state[::2, ::2]

    return state

def get_initial_state(env):
    return process_state(env.reset())

def get_action_space(env):
    return env.action_space

def get_successor_state(env, action):
    next_state, reward, done, _ = env.step(action)
    return process_state(next_state), reward, done


#the prediction model

class A3C_Net(object):
    def __init__(self, env, scope, sess, path='', seed=42):
        self.path = path
        self.seed = seed
        self.scope = scope
        self.sess = sess

        #trained for x batches
        self.steps = 0

        #set seeds
        tf.set_random_seed(self.seed)
        random.seed(self.seed)

        #threadsafe queue
        self.update_queue = Queue()

        #spaceinvaders input is (210, 160, 3)
        height, width, channels = get_initial_state(env).shape
        n_actions = get_action_space(env).n

        #ensure local copies of the net
        with tf.name_scope(self.scope):

            #preprocess raw inputs
            with tf.name_scope('preprocess_input'):
                #rgb input to square dimensions
                batchsize = None #None #FIXME: hardcoded value
                self.state_in = tf.placeholder(tf.float32, 
                        [batchsize, height, width, channels], 
                        name='state_in')
                dim = height if height > width else width
                state_square = tf.image.resize_image_with_crop_or_pad(
                        self.state_in, dim, dim)
                #action input to onehot
                self.action_in = tf.placeholder(tf.int32, [batchsize],
                        name='action_in')
                action_in = tf.one_hot(self.action_in, n_actions)
                #value input
                self.value_in = tf.placeholder(tf.float32, [batchsize],
                        name='value_in')

            #3x3 conv2d, relu, 2x2 maxpool
            with tf.name_scope('conv_pool'):
                #filter shape = [height, width, in_channels, 
                        #out_channels]
                #FIXME: out_channels hardcoded
                out_channels = 32
                filter_shape = [3, 3, channels, out_channels]
                conv_w = tf.Variable(tf.truncated_normal(filter_shape, 
                        stddev=0.1), name='weight')
                conv_b = tf.Variable(tf.constant(0.1, 
                        shape=[out_channels]), name='bias')
                conv = tf.nn.conv2d(state_square, conv_w, 
                        strides=[1,1,1,1], padding='SAME')
                relu = tf.nn.relu(conv + conv_b)
                pool = tf.nn.max_pool(relu, ksize=[1,2,2,1], 
                        strides=[1,2,2,1], padding='SAME')

            #FIXME: add dynamic lstm?

            #fully connected with dropout
            with tf.name_scope('dense_dropout'):
                #flatten input
                flat = tf.contrib.layers.flatten(pool)

                #FIXME: n
                n = 1024
                w_shape = [int(flat.shape[1]), n]
                fc_w = tf.Variable(tf.truncated_normal(w_shape, 
                        stddev=0.1), name='weight')
                fc_b = tf.Variable(tf.constant(0.1, 
                        shape=[n]), name='bias')

                fc_relu = tf.nn.relu(tf.matmul(flat, fc_w) + fc_b)
                #pool2 = tf.reshape() #flatten
                self.keep_prob = tf.placeholder(tf.float32)
                drop = tf.nn.dropout(fc_relu, self.keep_prob)

            #policy out, aka action values
            with tf.name_scope('action_prediction'):
                a_w = tf.Variable(tf.truncated_normal([n, n_actions], 
                        stddev=0.1), name='weight')
                a_b = tf.Variable(tf.constant(0.1, 
                        shape=[n_actions]), name='bias')
                actions = tf.matmul(drop, a_w) + a_b
                self.a_prob = tf.nn.softmax(actions)
                a_logprob = tf.nn.log_softmax(actions)
                a_pred =  tf.reduce_sum(tf.multiply(a_logprob, 
                        action_in))

            #value out, predicting the state reward essentially
            with tf.name_scope('value_prediction'):
                v_w = tf.Variable(tf.truncated_normal([n, 1], 
                        stddev=0.1), name='weight')
                v_b = tf.Variable(tf.constant(0.1, 
                        shape=[1]), name='bias')
                self.v_pred = tf.matmul(drop, v_w) + v_b

            #loss and optimization
                #functions from openai universe starter agent
                #gradient = log (policy) * (v - v_pred) + beta * entropy
            with tf.name_scope('loss'):
                advantage = self.v_pred - self.value_in
                
                #value loss
                v_loss  = 0.5 * tf.reduce_sum(tf.square(advantage))
                
                #policy loss
                a_loss = - tf.reduce_sum(a_pred * advantage)

                #entropy
                entropy = - tf.reduce_sum(self.a_prob * a_logprob)

                #loss used for gradients
                beta = 0.01 #FIXME: this is hardcoded
                self.loss = a_loss + 0.5 * v_loss - beta * entropy

            #calc and clip gradients for just local variables
                #clip to 40 idea from openai universe starter agent
                #should be set to 1/10 of max value that allows net 
                    #to converge
            with tf.name_scope('calc_gradients'):
                #optimizer
                learn_rate = 1e-4
                self.optimizer = tf.train.AdamOptimizer(learn_rate)
                #get local collection
                self.variables = tf.get_collection(
                        tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)
                #compute returns a tuple list (grad, var)
                grad, var = zip(*self.optimizer.compute_gradients(
                        self.loss, self.variables))
                self.gradients, _ = tf.clip_by_global_norm(grad, 40.0)

            with tf.name_scope('apply_gradients'):
                #number of steps model has been trained
                    #note that batch input is considered 1 step
                self.step_count = tf.Variable(0, name='step_count', 
                        trainable=False)
                self.inc_step = tf.assign_add(self.step_count, 1)

                #input gradients are the same shape as trainiable vars
                self.gradient_in = [tf.placeholder(tf.float32, x.shape) 
                        for x in self.variables]

                #zip with vars for optimizer
                grads_vars = zip(self.gradient_in, self.variables)

                self.optimize = self.optimizer.apply_gradients(
                        grads_vars, global_step=self.step_count)
            
            with tf.name_scope('replace_vars'):
                #create a placeholder for each trainable variable
                self.vars_in = [tf.placeholder(tf.float32, x.shape) 
                        for x in self.variables]
                var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 
                        self.scope)
                vars_list = zip(var, self.vars_in)
                op_list = [v.assign(w) for v,w in vars_list]
                self.put_vars = tf.group(*op_list)

                #self.sync = tf.group(*[v1.assign(v2) for v1, v2 in 
                        #zip(pi.var_list, self.network.var_list)])

            #tensorboard visualization
            with tf.name_scope('summaries'):
                all_summaries = [
                    tf.summary.scalar('loss', self.loss),
                    tf.summary.scalar('v_loss', v_loss),
                    tf.summary.scalar('a_loss', a_loss),
                    #tf.summary.scalar('value_pred', self.v_pred),
                    tf.summary.scalar('value_max', tf.reduce_max(
                            self.value_in)),
                    tf.summary.scalar('value_min', tf.reduce_min(
                            self.value_in)),
                    tf.summary.scalar('value_avg', tf.reduce_mean(
                            self.value_in)),
                ]

            #tensorboard data
            self.summaries = tf.summary.merge(all_summaries)
            
            #separate summary dirs
            self.writer = tf.summary.FileWriter('./logs/%s_data' % (
                    self.scope,), self.loss.graph, flush_secs=1)
            
        #self.sess = tf.Session()
        all_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 
                scope=self.scope)
        init_op = tf.variables_initializer(all_vars)
        self.sess.run(init_op)

        print ('[+] %s net initialized' % self.scope)

        '''
        #FIXME: ref
        process_rollout(rollout, gamma, lambda_=1.0)
            batch_si = states
            batch_a = actions
            
            rewards = rewards
            rewards_plus_v = rewards + [rollout.r]
            batch_r = discount(rewards_plus_v, gamma)[:-1]
            
            delta_t = rewards + gamma * values - values
            batch_adv = discount(delta_t, gamma * lambda_)

            discount(x, gamma)
                return lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]

        feed
            self.local_network.x: batch.si
            self.ac: batch.a
            self.adv: batch.adv
            self.r: batch.r

        paper advantage function
            sum of discounted rewards?
            gamma = discount (0,1]
            sum i=0 to k [gamma^i * reward_i+1 + gamma^k * V(s_t+k) - V(s_t)]
            tf.reduce_sum()
        '''

    def process_batch(self, batch):
        #FIXME: this is dumb, move to using an object to store batch
        #split batch 
        imgs = []
        actions = []
        rewards = []
        values = []
        dones = []
        for elem in batch:
            img, action, reward, value, done = elem
            imgs.append(img)
            actions.append(action)
            rewards.append(int(reward))
            values.append(value)
            dones.append(int(done)) #convert from bool
        imgs = np.asarray(imgs).astype(np.float32)
        actions = np.asarray(actions).astype(np.int32)
        rewards = np.asarray(rewards).astype(np.float32)
        values = np.asarray(values).astype(np.float32)
        dones = np.asarray(dones).astype(np.int32)
        return imgs, actions, rewards, values, dones

    def get_weights(self):
        #need to convert tensors to numpy arrays
        weights = [x.eval(session=self.sess) for x in self.variables]
        return weights

    def put_weights(self, weights):
        self.sess.run([self.put_vars], feed_dict={
                ph: v for ph,v in zip(self.vars_in, weights)})

    def apply_gradients(self, gradients):
        self.update_queue.put(gradients)

    def update_loop(self, steps, print_interval=100):
        #apply gradients in order given (fifo)
        step = self.get_step()
        while step < steps:
            while not self.update_queue.empty():
                #update msg
                if step % print_interval == 0 or step == steps - 1:
                    print ('%s applying grad %s' % (self.scope, step))

                gradients = self.update_queue.get()
                self.sess.run([self.optimize], feed_dict={
                        ph: g for ph,g in zip(self.gradient_in, gradients)})
                step = self.get_step()

    def calc_gradients(self, batch):
        imgs, actions, rewards, values, _ = self.process_batch(batch)
        loss, gradients, summary, step = self.sess.run([self.loss, 
                self.gradients, self.summaries, self.inc_step], 
                feed_dict={
                    self.state_in: imgs, 
                    self.action_in: actions, 
                    self.value_in: values, 
                    self.keep_prob: 0.5})
        #step = self.step_count.eval(session=self.sess)
        #print ('%s step: %s' % (self.scope, step))
        self.writer.add_summary(summary, step)

        return gradients

    def get_action_value(self, state):
        #FIXME: choose action according to policy
            #choose best action with probability that varies based on
            #how close the highest rewards are
                #ie explore more when there are similar high rated actions
                #otherwise low chance to explore, choose best always
                #measure entropy...
        #epsilon = (% based on top_k values)
        epsilon = 0.14
            #actions,stddev = sess.run(...)
                #run a top_k tensor and calc the stddev?
            #choose argmax with prob 1-epsilon
            #choose random with prob epsilon

        action, value = self.sess.run([self.a_prob, self.v_pred],
                feed_dict={self.state_in: [state], self.keep_prob: 1.0})
        return np.argmax(action[0]), value[0][0]

    def get_step(self):
        return self.step_count.eval(session=self.sess)

    def save(self):
        saver = tf.train.Saver()
        saver.save(self.sess, '%s/model' % self.path)


class A3C_Worker(object):
    def __init__(self, coordinator, global_net, local_net, scope): 
        self.scope = scope
        self.global_net = global_net
        self.local_net = local_net
        self.pull_weights()

        self.update_interval = 20

    def train(self, env, global_step_max=10):
        #train for specified number of steps
        #t = self.global_net.get_step()
        #t_max = steps + t
        batch = []

        state = get_initial_state(env)

        while self.global_net.get_step() < global_step_max:
            action, value = self.local_net.get_action_value(state)
            next_state, reward, done = get_successor_state(env, action)
            reward = 0 if done else reward

            #DEBUG
            #env.render()

            #add example to batch
            example = (state, action, reward, value, done)
            batch.append(example)

            #reset if terminal state, else continue
            if done:
                state = get_initial_state(env)

            state = next_state

            if len(batch) >= self.update_interval:
                #push gradients to global_net
                self.push_gradients(batch)

                #FIXME: pull gradients from global_net
                self.pull_weights()
                
                #reset experience batch
                batch = []

        #DEBUG FIXME
        #print 'trained for %s steps (%s)' % (t, self.global_net.get_step())
        #print 'global trained: %s' % self.global_net.get_step()
        #print '%s trained: %s' % (self.scope, self.local_net.get_step())
        print ('%s quit after training for %s' % (self.scope, 
                self.local_net.get_step()))

    def push_gradients(self, batch):
        gradients = self.local_net.calc_gradients(batch)
        self.global_net.apply_gradients(gradients)

    def pull_weights(self):
        self.local_net.put_weights(self.global_net.get_weights())

    def test(self, env, episodes=100, records=4, out_dir='./records'):
        #wrap env, record x episodes and eval scores

        #wrapper that records episodes
        #vc = lambda n: n in [int(x) for x in np.linspace(episodes, 0, 
        #        records)] #func that indicates which episodes to record
        #env = wrappers.Monitor(env, directory=out_dir, force=True, 
        #        video_callable=vc)

        #FIXME: play for x episodes
        stats = {}
        stats['steps'] = []
        for i in range(episodes):
            steps = 0
            done = False
            state = get_initial_state(env)
            while not done:
                action, _ = self.local_net.get_action_value(state)
                state, _, done = get_successor_state(env, action)
                steps += 1
            stats['steps'].append(steps)
        total_steps = np.asarray(stats['steps'])
        print('%s tested for %s episodes' % (self.scope, episodes))
        print('max: %s' % np.nanmax(total_steps))
        print('min: %s' % np.nanmin(total_steps))
        print('avg: %s' % np.average(total_steps))
        print
        print(total_steps)
        print


class batch(object):
    def __init__(self):
        #self.rewards = []
        #...
        #self.gradients = []
        pass

    def reset(self):
        #clear all values
        pass

    def get_data(self):
        #returns vectorized input to be fed to graph
        pass

    def add(self, example):
        #adds a train experience to batch
        pass


























