#import xml.etree.ElementTree
import os
#import fnmatch
#import xml.dom
import tensorflow as tf
import numpy as np
#import collections
import sys
#from os import listdir
#from glob import glob
#from xml.dom import minidom
#from collections import defaultdict

class LSTMModel_old(object):

	def __init__ (self, vocab_size, batch_size, num_steps, hidden_size, num_layers, max_grad_norm, decay_rate, is_training):
		#print "* LSTMModel init()"
		tf.logging.set_verbosity(tf.logging.ERROR)
		self.batch_size = batch_size
		self.num_steps = num_steps
		self.hidden_size = hidden_size
		self.num_layers = num_layers

		# LSTM -- learning params
		self.max_grad_norm = max_grad_norm
		self.decay_rate = decay_rate

		# OLD WAY, which didn't crash
		self._input_data = tf.placeholder(tf.int32, [self.batch_size, self.num_steps])
		self._targets = tf.placeholder(tf.int32, [self.batch_size, self.num_steps])
		
 		# self._input_data = tf.placeholder(tf.int32, [self.num_steps, self.batch_size])
		# self._targets = tf.placeholder(tf.int32, [self.num_steps, self.batch_size])
		# TF 0.12
		lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(self.hidden_size, forget_bias = 0.0, state_is_tuple=False)
		cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell]*self.num_layers, state_is_tuple=False)

		#TF 1
		#lstm_cell = tf.contrib.rnn.BasicLSTMCell(self.hidden_size, forget_bias = 0.0, state_is_tuple=False)
		#cell = tf.contrib.rnn.MultiRNNCell([lstm_cell]*self.num_layers, state_is_tuple=False)
		self._initial_state = cell.zero_state(self.batch_size, tf.float32)

		#with tf.device("/" + str(dev_name) + ":0"):
		embedding = tf.get_variable("embedding", [vocab_size, self.hidden_size])
		#print "embedding: " + str(embedding)
		inputs = tf.nn.embedding_lookup(embedding, self._input_data)
		#print(inputs)

		# new way
		# inputs = [tf.squeeze(input_, [0]) for input_ in tf.split(0, self.num_steps, inputs)]

		# old batch way		
		# TF 1 has this commented out
		# inputs = [tf.squeeze(input_, [1]) for input_ in tf.split(inputs, self.num_steps, 1)]
		# outputs, state = tf.nn.dynamic_rnn(cell, inputs, initial_state = self._initial_state)
		# hidden_layer = tf.reshape(outputs, [-1, self.hidden_size])
		# TF 0.12 has this uncommented
		inputs = [tf.squeeze(input_, [1]) for input_ in tf.split(1, self.num_steps, inputs)]
		outputs, state = tf.nn.rnn(cell, inputs, initial_state = self._initial_state)
		hidden_layer = tf.reshape(tf.concat(1, outputs), [-1, self.hidden_size])

		self.hidden_layer = hidden_layer
		softmax_w = tf.get_variable("softmax_w", [self.hidden_size, vocab_size])
		softmax_b = tf.get_variable("softmax_b", [vocab_size])
		logits = tf.matmul(hidden_layer, softmax_w) + softmax_b

		self.softmax_w = softmax_w
		self.softmax_b = softmax_b
		self.logits = logits
		probDistr = tf.nn.softmax(logits)
		self.probDistr = probDistr

		# TF 1
		#loss = tf.contrib.seq2seq.sequence_loss(tf.reshape(logits,[-1, self.num_steps, vocab_size]), self._targets, tf.ones([self.batch_size, self.num_steps]), average_across_timesteps=False, average_across_batch=False)
		 
		# TF 0.12
		loss = tf.nn.seq2seq.sequence_loss_by_example([logits], [tf.reshape(self._targets, [-1])], [tf.ones([self.batch_size*self.num_steps])])

		# maybe this is what we tried to get TF 1 to work?
		#loss = tf.losses.sparse_softmax_cross_entropy(tf.reshape(self._targets, [-1]), logits)
		#self._cost = loss
		self._cost = tf.reduce_sum(loss)/self.batch_size
		self._final_state = state

		if not is_training:
			return

		self._lr = tf.Variable(0.0, trainable = False)
		tvars = tf.trainable_variables()
		grads, _ = tf.clip_by_global_norm(tf.gradients(self._cost, tvars), self.max_grad_norm)
		optimizer = tf.train.GradientDescentOptimizer(self._lr)
		self._train_op = optimizer.apply_gradients(zip(grads, tvars))

	def main():
		print "LSTMModel() in main()"

if __name__ == "__main__":
	main()