
import sys
import random
import tensorflow as tf
import numpy as np

class A3C_Net(object):
    def __init__(self, env, scope, path='', seed=42):
        self.path = path
        self.seed = seed
        self.scope = scope

        #set seeds
        tf.set_random_seed(self.seed)
        random.seed(self.seed)

        #restore model
        if path:
            pass
            #self.sess = tf.Session()
            #saver = tf.train.import_meta_graph('%s/model.meta' % path)
            #graph = tf.get_default_graph()
            #self.tensor = graph.get_tensor_by_name('scope/name:0')
            '''
                    self.img_in = tf.placeholder(tf.float32, [None, height, 
                    self.action_in = tf.placeholder(tf.int32)
                    self.value_in = tf.placeholder(tf.float32)
                    self.loss = a_loss + 0.5 * v_loss - beta * entropy
                    self.variables = tf.get_collection(
                    self.gradients = tf.clip_by_global_norm(gradients, 40.0)
                    self.optimizer = tf.train.AdamOptimizer(learn_rate)
                    self.step_count = tf.Variable(0, name='global_step', 
            '''
        
        #create the graph
        else:

            #spaceinvaders input is (210, 160, 3)
            height, width, channels = env.reset().shape
            n_actions = env.action_space.n

            #ensure local copies of the net
            with tf.name_scope(self.scope):

                #preprocess raw inputs
                with tf.name_scope('preprocess_input'):
                    #rgb input to square dimensions
                    self.img_in = tf.placeholder(tf.float32, [None, height, 
                            width, channels])
                    dim = height if height < width else width
                    img_square = tf.image.resize_image_with_crop_or_pad(
                            self.img_in, dim, dim)
                    #action input to onehot
                    self.action_in = tf.placeholder(tf.int32)
                    action_in = tf.one_hot(self.action_in, n_actions)
                    #value input
                    self.value_in = tf.placeholder(tf.float32)

                #3x3 conv2d, relu, 2x2 maxpool
                with tf.name_scope('conv_pool'):
                    #filter shape = [height, width, in_channels, out_channels]
                    #FIXME: out_channels
                    out_channels = 32
                    filter_shape = [3, 3, channels, out_channels]
                    conv_w = tf.Variable(tf.truncated_normal(filter_shape, 
                            stddev=0.1), name='weight')
                    conv_b = tf.Variable(tf.constant(0.1, 
                            shape=[out_channels]), name='bias')
                    conv = tf.nn.conv2d(img_square, conv_w, strides=[1,1,1,1], 
                            padding='SAME')
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
                    a_prob = tf.nn.softmax(actions)
                    a_logprob = tf.nn.log_softmax(actions)
                    a_pred =  tf.reduce_sum(tf.multiply(a_logprob, 
                            action_in))

                #value out, predicting the state reward essentially
                with tf.name_scope('value_prediction'):
                    v_w = tf.Variable(tf.truncated_normal([n, 1], 
                            stddev=0.1), name='weight')
                    v_b = tf.Variable(tf.constant(0.1, 
                            shape=[1]), name='bias')
                    value_pred = tf.matmul(drop, v_w) + v_b

                #loss and optimization
                    #functions from openai universe starter agent
                    #gradient = log (policy) * (v - v_pred) + beta * entropy
                with tf.name_scope('loss'):
                    beta = 0.01
                    advantage = self.value_in - value_pred
                    
                    #value loss
                    v_loss  = 0.5 * tf.square(advantage)
                    
                    #policy loss
                    #FIXME: a_loss shape error - [20,6] vs [20]
                    a_loss = - tf.reduce_sum(a_pred * advantage)

                    #entropy
                    entropy = - tf.reduce_sum(a_prob * a_logprob)

                    #loss used for gradients
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
                    gradients, _ = tf.clip_by_global_norm(grad, 40.0)
                    
                with tf.name_scope('apply_gradients'):
                    #number of steps model has been trained
                    self.step_count = tf.Variable(0, name='global_step', 
                            trainable=False)
                    self.optimize = self.optimizer.apply_gradients(
                            zip(gradients, self.variables), 
                            global_step=self.step_count)


                    #for grad, var in self.gradients: #shape is (8,2)
                    #    print grad
                    #    print var
                    #    print
                    
                    #FIXME: solve exploding gradient problem
                    #self.gradients = tf.clip_by_global_norm(gradients, 40.0)

                #tensorboard visualization
                with tf.name_scope('summaries'):
                    tf.summary.scalar('loss', self.loss)

        #tensorboard data
        self.summaries = tf.summary.merge_all()
        self.writer = tf.summary.FileWriter('./%s_data' % self.scope, 
                self.loss.graph, flush_secs=1)

        #initialize vars if not restoring model
        if not path:
            self.sess = tf.Session()
            all_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 
                    scope=self.scope)
            init_op = tf.variables_initializer(all_vars)
            self.sess.run(init_op)

            #uninit = self.sess.run(tf.report_uninitialized_variables())
            #self.sess.run(init)
            #print '\nuninitialized: %s' % uninit

        print '[+] %s %s model at %s steps' % ('recovered' if path 
                else 'created', self.scope, 
                self.step_count.eval(session=self.sess))

    def apply_gradients(self, gradients):
        #self.sess.run(self.apply_grads)
        pass


    def calc_gradients(self, batch):
        #split batch #FIXME: this is dumb
        imgs = []
        actions = []
        rewards = []
        values = []
        dones = []
        for elem in batch:
            img, action, reward, value, done = elem
            imgs.append(img)
            actions.append(action)
            rewards.append(reward)
            values.append(value)
            dones.append(done)
        imgs = np.asarray(imgs)
        actions = np.asarray(actions)
        rewards = np.asarray(rewards)
        values = np.asarray(values)
        dones = np.asarray(dones)

        self.sess.run([self.optimize],
                feed_dict={self.img_in: imgs,
                            self.action_in: actions,
                            self.value_in: rewards,
                            self.keep_prob: 0.5})
        #return gradients

    def get_step(self):
        return self.step_count.eval(session=self.sess)

    def save(self):
        saver = tf.train.Saver()
        saver.save(self.sess, '%s/model' % self.path)


class A3C_Worker(object):
    def __init__(self, env, coordinator, global_net, scope): 
        self.env = env
        self.global_net = global_net
        self.local_net = A3C_Net(env, scope)
        self.pull_weights()

        self.update_interval = 20

    def train(self, env, steps=100):
        #FIXME: move to global net, make global net train for x steps and
            #kill
        #train for specified number of steps
        t = self.global_net.get_step()
        t_max = steps + t
        state = env.reset()
        batch = []

        while t < t_max: #and not coordinator.should_stop():
            action, value = self.get_action_value(state)
            value = 0 #get from policy.act(state)
            next_state, reward, done, _ = env.step(action)
            reward = 0 if done else reward

            #DEBUG
            #env.render()

            #add example to batch
            example = (state, action, reward, value, done)
            batch.append(example)

            #reset if terminal state, else continue
            if done:
                state = env.reset()

            if len(batch) >= self.update_interval:
                #FIXME: update global net, add batch to queue
                #push gradients to global_net
                self.push_gradients(batch)

                #pull gradients from global_net
                self.pull_weights()
                
                #reset experience batch
                batch = []

            else:
                state = next_state

            #FIXME: stop when we have trained for specified number of steps
            if self.global_net.get_step() >= t_max or t >= t_max:
                #send last partial batch?
                #coordinator.request_stop()
                break
            else: #FIXME
                t += 1

        #DEBUG FIXME
        print 'trained for %s steps (%s)' % (t, self.global_net.get_step())

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
        return self.env.action_space.sample(), 1

    def push_gradients(self, batch):
        gradients = self.local_net.calc_gradients(batch)
        self.global_net.apply_gradients(gradients)

    def pull_weights(self):
        #FIXME: pull gradients from gobal_net
        #f_scope = 'global'
        #t_scope = self.name
        #from = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
        #to = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)

        #ops = []
        #for f,t in zip(from, to):
            #ops.append(t.assign(f))
        #return ops
        pass

    def test(self, env, episodes=100, records=4, out_dir='./records'):
        #wrap env, record x episodes and eval scores

        #wrapper that records episodes
        vc = lambda n: n in [int(x) for x in np.linspace(episodes, 0, 
                records)] #func that indicates which episodes to record
        env = wrappers.Monitor(env, directory=out_dir, force=True, 
                video_callable=vc)

        #FIXME: play for x episodes
        done = False
        for _ in range(episodes):
            while not done:
                action = env.action_space.sample()
                _, reward, done, _ = env.step(action)




























