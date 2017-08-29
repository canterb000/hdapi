#!/usr/bin/python
# Author: Clara Vania

import numpy as np
import tensorflow as tf
import argparse
import time
import os
import cPickle
from utils import TextLoader
from word import WordLM


def main():
    parser = argparse.ArgumentParser()
    #parser.add_argument('--test_file', type=str, default='data/tinyshakespeare/test.txt',
    parser.add_argument('--test_file', type=str, default='data/tinyshakespeare/test.txt',
                        help="test file")
    parser.add_argument('--save_dir', type=str, default='model',
                        help='directory of the checkpointed models')

    parser.add_argument('--train_file', type=str, default='data/tinyshakespeare/train.txt',
                        help="training data")
    parser.add_argument('--dev_file', type=str, default='data/tinyshakespeare/dev.txt',
                        help="development data")
    parser.add_argument('--output', '-o', type=str, default='train.log',
                        help='output file')
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
    test(args)


def run_epoch(session, m, data, data_loader, eval_op):
    costs = 0.0
    iters = 0
    state0 = m.initial_lm_state[0].eval()
    state1 = m.initial_lm_state[1].eval()
    state = [state0, state1]
    for step, (x, y) in enumerate(data_loader.data_iterator(data, m.batch_size, m.num_steps)):
        cost, state, _ = session.run([m.cost, m.final_state, eval_op],
                                     {m.input_data: x,
                                      m.targets: y,
                                      m.initial_lm_state: state})
        costs += cost
        iters += m.num_steps
    return np.exp(costs / iters)


def test(test_args):
    start = time.time()
    #with open(os.path.join(test_args.save_dir, 'config.pkl')) as f:
    #    args = cPickle.load(f)
    data_loader = TextLoader(test_args, train=False)
    test_data = data_loader.read_dataset(test_args.test_file)

    test_args.word_vocab_size = data_loader.word_vocab_size
    print "Word vocab size: " + str(data_loader.word_vocab_size) + "\n"

    # Model
    lm_model = WordLM

    print "Begin testing..."
    # If using gpu:
    # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)
    # gpu_config = tf.ConfigProto(log_device_placement=False, gpu_options=gpu_options)
    # add parameters to the tf session -> tf.Session(config=gpu_config)
    with tf.Graph().as_default(), tf.Session() as sess:
        initializer = tf.random_uniform_initializer(-test_args.init_scale, test_args.init_scale)
        with tf.variable_scope("model", reuse=None, initializer=initializer):
            mtest = lm_model(test_args, is_training=False, is_testing=True)

        # save only the last model
        saver = tf.train.Saver(tf.all_variables())
        tf.initialize_all_variables().run()
        ckpt = tf.train.get_checkpoint_state(test_args.save_dir)
        if ckpt and ckpt.model_checkpoint_path:
	    print(".............restore %s"%ckpt)
            saver.restore(sess, ckpt.model_checkpoint_path)
        tf.initialize_all_variables().run()

	#reset w
	reset_w = tf.zeros([test_args.rnn_size, test_args.out_vocab_size])
	mtest.assign_w(sess, reset_w)
	print(mtest.output_weights.eval())
        #test_perplexity = run_epoch(sess, mtest, test_data, data_loader, tf.no_op())
        test_perplexity = run_epoch(sess, mtest, test_data, data_loader, tf.no_op())
        print("Test Perplexity: %.3f" % test_perplexity)
        print("Test time: %.0f" % (time.time() - start))


if __name__ == '__main__':
    main()
