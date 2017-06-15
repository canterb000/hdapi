import tensorflow as tf
from tensorflow.contrib import rnn

class RNNMODEL(object):
	def __init__(self, args, is_training, is_testing=False):
		self.learning_rate = args.learning_rate
		n_input = args.n_input
		NUM_CLASS = args.num_class
		n_hidden = args.n_hidden
		vocab_size = args.vocab_size

		# tf Graph input
		self._x = tf.placeholder("float", [None, n_input])
		self._y = tf.placeholder("float", [None, NUM_CLASS])



		'''
		# RNN output node weights and biases
		self.weights = {
		    'out': tf.Variable(tf.random_normal([n_hidden, vocab_size]))
		}
		self.biases = {
		    'out': tf.Variable(tf.random_normal([vocab_size]))
		}
		'''	
	        self.weights = tf.get_variable("softmax_w", [n_hidden, vocab_size])
	        self.biases = tf.get_variable("softmax_b", [vocab_size])



		print ("init1")
		# reshape to [1, n_input]
		self.x_reshape = tf.reshape(self._x, [-1, n_input])

		# Generate a n_input-element sequence of inputs
		# (eg. [had] [a] [general] -> [20] [6] [33])
		self.x_split = tf.split(self.x_reshape, n_input, 1)

		cell_fn = tf.contrib.rnn.BasicLSTMCell

		cell = cell_fn(n_hidden, forget_bias=0.0, state_is_tuple=False, reuse = tf.get_variable_scope().reuse)
        	cell2 = cell_fn(n_hidden, forget_bias=0.0, state_is_tuple=False, reuse = tf.get_variable_scope().reuse)

		lm_cell = tf.contrib.rnn.MultiRNNCell([cell, cell2])

		#rnn_cell = rnn.MultiRNNCell([rnn.BasicLSTMCell(n_hidden, state_is_tuple=False, reuse = tf.get_variable_scope().reuse),rnn.BasicLSTMCell(n_hidden, state_is_tuple=False, reuse = tf.get_variable_scope().reuse)])
		'''
	        cell_fn = rnn.BasicLSTMCell

        	cell = cell_fn(n_hidden, forget_bias=0.0, state_is_tuple=False, reuse = tf.get_variable_scope().reuse)
        	cell2 = cell_fn(n_hidden, forget_bias=0.0, state_is_tuple=False, reuse = tf.get_variable_scope().reuse)

        	if is_training and args.keep_prob < 1:
            		cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=args.keep_prob)
            		cell2 = tf.contrib.rnn.DropoutWrapper(cell2, output_keep_prob=args.keep_prob)
        	rnn_cell = tf.contrib.rnn.MultiRNNCell([cell, cell2])
		'''
		print(tf.get_variable_scope().reuse)
		batch_size = 10
		self._initial_lm_state = lm_cell.zero_state(batch_size, tf.float32)
		print ("init2")
		print(self._initial_lm_state)

		# generate prediction
		self.outputs, self.states = rnn.static_rnn(lm_cell, self.x_split, dtype=tf.float32)

		# we only want the last output
		#self.pred = tf.matmul(self.outputs[-1], self.weights['out']) + self.biases['out']
		self.pred = tf.matmul(self.outputs[-1], self.weights) + self.biases

		self.correct_pred = tf.equal(tf.argmax(self.pred,1), tf.argmax(self._y,1))
		self.accuracy = tf.reduce_mean(tf.cast(self.correct_pred, tf.float32))

		self._final_state = self.states
		self._cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.pred, labels=self._y))


	        if not is_training:
        		return
		
		# Loss and optimizer
		#optimizer = tf.train.RMSPropOptimizer(learning_rate = self.learning_rate)
        	#optimizer = tf.train.GradientDescentOptimizer(self.lr)



	        tvars = tf.trainable_variables()
        	grads, _ = tf.clip_by_global_norm(tf.gradients(self._cost, tvars),
                                          5.0)
                                          #args.grad_clip)
            	#optimizer = tf.train.MomentumOptimizer(self._lr, 0.95)
        	optimizer = tf.train.GradientDescentOptimizer(1.0)
         	self._train_op = optimizer.apply_gradients(zip(grads, tvars))
        	#self._train_op = optimizer.minimize(self._cost)


		print ("init3")

	@property
	def input_data(self):
		return self._x

	@property
	def targets(self):
		return self._y

	@property
	def cost(self):
		return self._cost

        @property
        def acc(self):
                return self.accuracy


	@property
	def train_op(self):
		#return self.optimizer
		return self._train_op

	@property
	def initial_lm_state(self):
		return self._initial_lm_state

	@property
	def final_state(self):
		return self._final_state


	@property
	def output_weights(self):
		return self.weights
