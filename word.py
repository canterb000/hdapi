#!/usr/bin/python
# Author: Clara Vania

import tensorflow as tf
from tensorflow.contrib import rnn 

class WordLM(object):
    """
    RNNLM with LSTM + Dropout
    Code based on tensorflow tutorial on building a PTB LSTM model.
    https://www.tensorflow.org/versions/r0.7/tutorials/recurrent/index.html
    """
    def __init__(self, args, is_training, is_testing=False):
        self.batch_size = batch_size = args.batch_size
        self.num_steps = num_steps = args.num_steps
        self.model = model = args.model
        self.optimizer = args.optimization

        rnn_size = args.rnn_size
        rnn_cell = tf.contrib.rnn
        word_vocab_size = args.word_vocab_size
        out_vocab_size = args.out_vocab_size
        # tf_device = "/gpu:" + str(args.gpu)

        if is_testing:
            self.batch_size = batch_size = 1
            self.num_steps = num_steps = 1

        cell_fn = rnn_cell.BasicLSTMCell

        # placeholders for data
        self._input_data = tf.placeholder(tf.int32, shape=[batch_size, num_steps])
        self._targets = tf.placeholder(tf.int32, shape=[batch_size, num_steps])

        # ********************************************************************************
        # RNNLM
        # ********************************************************************************
        # you can add this if using gpu
        # with tf.device(tf_device):
        cell = cell_fn(rnn_size, forget_bias=0.0, state_is_tuple=False, reuse = tf.get_variable_scope().reuse)
        cell2 = cell_fn(rnn_size, forget_bias=0.0, state_is_tuple=False, reuse = tf.get_variable_scope().reuse)

        if is_training and args.keep_prob < 1:
            cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=args.keep_prob)
            cell2 = tf.contrib.rnn.DropoutWrapper(cell2, output_keep_prob=args.keep_prob)
#        lm_cell = tf.contrib.rnn.MultiRNNCell([cell for _ in range(args.num_layers)])
        lm_cell = tf.contrib.rnn.MultiRNNCell([cell, cell2])
       
        self._initial_lm_state = lm_cell.zero_state(batch_size, tf.float32)
	print("wordlm init lm state:*************") 
	print(self._initial_lm_state)
        with tf.device("/cpu:0"):
            embedding = tf.get_variable("embedding", [word_vocab_size, rnn_size])
            inputs = tf.nn.embedding_lookup(embedding, self._input_data)

        if is_training and args.keep_prob < 1:
            inputs = tf.nn.dropout(inputs, args.keep_prob)

        # split input into a list
        lm_inputs = tf.split(inputs, num_steps, 1)
        lm_inputs = [tf.squeeze(input_, [1]) for input_ in lm_inputs]
        #lm_outputs, lm_state = tf.contrib.rnn(lm_cell, lm_inputs, initial_state=self._initial_lm_state)
        lm_outputs, lm_state = tf.contrib.rnn.static_rnn(lm_cell, lm_inputs, initial_state=self._initial_lm_state)
     
        lm_outputs = tf.concat(lm_outputs, 1)
        lm_outputs = tf.reshape(lm_outputs, [-1, rnn_size])

        self.softmax_w = tf.get_variable("softmax_w", [rnn_size, out_vocab_size])
        softmax_b = tf.get_variable("softmax_b", [out_vocab_size])

        logits = tf.matmul(lm_outputs, self.softmax_w) + softmax_b

        # compute log perplexity
        loss = tf.contrib.legacy_seq2seq.sequence_loss_by_example(
       # loss = tf.nn.seq2seq.sequence_loss_by_example(
            [logits],
            [tf.reshape(self._targets, [-1])],
            [tf.ones([batch_size * num_steps])])

        # cost is the log perplexity
        self._cost = cost = tf.reduce_sum(loss) / batch_size
        self._final_state = lm_state

        if not is_training:
            return

        self._lr = tf.Variable(0.0, trainable=False)
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars),
                                          args.grad_clip)
        if self.optimizer == "momentum":
            optimizer = tf.train.MomentumOptimizer(self.lr, 0.95)
        else:
            optimizer = tf.train.GradientDescentOptimizer(self.lr)
        self._train_op = optimizer.apply_gradients(zip(grads, tvars))
        #self._train_op = optimizer.minimize(self._cost)

    def assign_lr(self, session, lr_value):
        session.run(tf.assign(self.lr, lr_value))

    def assign_w(self, session, w_value):
        session.run(tf.assign(self.softmax_w, w_value))


    @property
    def output_weights(self):
        return self.softmax_w

    @property
    def input_data(self):
        return self._input_data

    @property
    def targets(self):
        return self._targets

    @property
    def initial_lm_state(self):
        return self._initial_lm_state

    @property
    def cost(self):
        return self._cost

    @property
    def final_state(self):
        return self._final_state

    @property
    def lr(self):
        return self._lr

    @property
    def train_op(self):
        return self._train_op
