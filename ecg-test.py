#!/usr/bin/env python


from __future__ import print_function

import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn
import random
import collections
import time
import argparse
from ecgmodel import RNNMODEL


NUM_CLASS = 12
FEATURE_SIZE = 500

start_time = time.time()
def elapsed(sec):
    if sec<60:
        return str(sec) + " sec"
    elif sec<(60*60):
        return str(sec/60) + " min"
    else:
        return str(sec/(60*60)) + " hr"


# Target log path
logs_path = '/tmp/tensorflow/logs'

def read_data(fname):
    content = []
    with open(fname) as f:
        for l in f.readlines():
            content.append(float(l))
    return content

def read_and_decode(filename_queue):
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example,
    features={
        "label": tf.FixedLenFeature([NUM_CLASS], tf.int64),
        "signal": tf.FixedLenFeature([FEATURE_SIZE], tf.float32),
    })

    label_out = features["label"]
    feature_out = features["signal"]

    return label_out, feature_out


'''
#input 
filename_queue = tf.train.string_input_producer(["train.tfrecords"])
 

batchSize = 10
min_after_dequeue = 8
capacity = min_after_dequeue + 3 * batchSize
num_threads = 1
label, features = read_and_decode(filename_queue)
'''



# Parameters
learning_rate = 0.001
training_iters = 12000
display_step = 1000
n_input = 500

# number of units in RNN cell
vocab_size = NUM_CLASS


parser = argparse.ArgumentParser()
parser.add_argument('--n_input', type=int, default='500',
                help="input length")
parser.add_argument('--n_hidden', type=int, default='10',
                help="hidden layer")
parser.add_argument('--num_class', type=int, default='12',
                help="label class")
parser.add_argument('--vocab_size', type=int, default='12',
                help="label class")


parser.add_argument('--output', '-o', type=str, default='train.log',
                help='output file')
parser.add_argument('--save_dir', type=str, default='rnn-model-2',
                help='directory to store checkpointed models')
parser.add_argument('--rnn_size', type=int, default=400,    #200,
                help='size of RNN hidden state')
parser.add_argument('--num_layers', type=int, default=2,#2
                help='number of layers in the RNN')
parser.add_argument('--model', type=str, default='lstm',
                help='rnn, gru, or lstm')
parser.add_argument('--batch_size', type=int, default=10,
                help='minibatch size')
parser.add_argument('--num_steps', type=int, default=20,
                help='RNN sequence length')
parser.add_argument('--out_vocab_size', type=int, default=20000,
                help='size of output vocabulary')
parser.add_argument('--num_epochs', type=int, default=3,
                help='number of epochs')
parser.add_argument('--validation_interval', type=int, default=1,
                help='validation interval')
parser.add_argument('--init_scale', type=float, default=0.1,
                help='initial weight scale')
parser.add_argument('--grad_clip', type=float, default=5.0,
                help='maximum permissible norm of the gradient')
parser.add_argument('--learning_rate', type=float, default=1.0,
                help='initial learning rate')
parser.add_argument('--decay_rate', type=float, default=0.5,
                help='the decay of the learning rate')
parser.add_argument('--keep_prob', type=float, default=0.5,
                help='the probability of keeping weights in the dropout layer')
parser.add_argument('--optimization', type=str, default='sgd',
                help='sgd, momentum, or adagrad')
args = parser.parse_args()



m = RNNMODEL

#input 
#filename_queue = tf.train.string_input_producer(["test.tfrecords"])
filename_queue = tf.train.string_input_producer(["test.tfrecords"])


batchSize = 10
min_after_dequeue = 8
capacity = min_after_dequeue + 3 * batchSize
num_threads = 1
label, features = read_and_decode(filename_queue)

batch_labels, batch_features = tf.train.shuffle_batch([label, features], batch_size= batchSize, num_threads= num_threads, capacity= capacity, min_after_dequeue = min_after_dequeue)



print ("Begin testing...")
# If using gpu:
# gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)
# gpu_config = tf.ConfigProto(log_device_placement=False, gpu_options=gpu_options)
# add parameters to the tf session -> tf.Session(config=gpu_config)
with tf.Graph().as_default(), tf.Session() as sess:
	initializer = tf.random_uniform_initializer(-args.init_scale, args.init_scale)
	with tf.variable_scope("model", reuse=None, initializer=initializer):
		mtest = m(args, is_training=False, is_testing=True)

	# save only the last model
	saver = tf.train.Saver(tf.all_variables())
	tf.initialize_all_variables().run()
	ckpt = tf.train.get_checkpoint_state(args.save_dir)
	print("ckpt...........args.save_dir: %s"%args.save_dir)
	print(ckpt)
	if ckpt and ckpt.model_checkpoint_path:
		saver.restore(sess, ckpt.model_checkpoint_path)
		print("saver restoring...")
	#***************************************************************JUST FOR DEBUG, WITH ALL VAR DELETED
	tf.initialize_all_variables().run()
	zero_weights = tf.zeros([args.n_hidden, args.vocab_size], dtype=tf.float32, name=None)
	random_weights = tf.random_normal([args.n_hidden, args.vocab_size], mean=0.0, stddev=1.0, dtype=tf.float32, seed=None, name=None)
	#mtest.assign_w(sess, random_weights)
	print(mtest.output_weights.eval())


	state = mtest.initial_lm_state
        state_input0 = state[0].eval()
        state_input1 = state[1].eval()
        state = [state_input0, state_input1]
	print("*****************************initial state:")
	#print(mtest.initial_lm_state[1].eval())
	step = 0

	coord = tf.train.Coordinator()
	threads = tf.train.start_queue_runners(sess=sess, coord=coord)
	acc_total = 0
	try:
	    while not coord.should_stop():
		x_r, y_r = sess.run([batch_features, batch_labels])

		costs = 0.0
		iters = 0
		acc, state = sess.run([mtest.acc, mtest.final_state],
				             {mtest.input_data: x_r,
				              mtest.targets: y_r,
				              mtest.initial_lm_state: state})


	        acc_total += acc

                if (step+1) % display_step == 0:
                	print("TEST:***  Iter= " + str(step+1) + \
                 	  ", Average Accuracy= " + \
                   	  "{:.2f}%".format((100 * acc_total / display_step)))
                	acc_total = 0
		step += 1
            
		'''
            if dev_pp > dev_perplexity:
                print "Achieve highest perplexity on dev set, save model."
                checkpoint_path = os.path.join(save_dir, 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step=e)
                print "model saved to {}".format(checkpoint_path)
                dev_pp = dev_perplexity

		'''




		if step > training_iters:
			break

	except tf.errors.OutOfRangeError:
		print ('Done training -- epoch limit reached')
	finally:
		coord.request_stop()
		sess.close()
		print("Test Finished!")
		print("Elapsed time: ", elapsed(time.time() - start_time))

