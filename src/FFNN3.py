import tensorflow as tf
import numpy as np
import random
import sys
import os
import time

from tensorflow.contrib.framework import deprecated
from tensorflow.contrib.losses.python.losses import loss_ops
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops as array_ops_
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn

from random import randint
from ECBHelper import ECBHelper
from sklearn import datasets
from sklearn.cross_validation import train_test_split


class FFNN3:
	def __init__(self, helper, vectorLength, model, baseOut, optimizers, hiddens, keep_inputs, subset, num_epochs, lrs, moms, nnmethod, subsample, penalty, activation):
		self.helper = helper
		self.vectorLength = vectorLength
		self.model = model
		self.baseOut = baseOut # fixed
		self.optimizers = optimizers
		self.hiddens = hiddens
		self.keep_inputs = keep_inputs
		self.subset = subset # fixed
		self.num_epochs = num_epochs
		self.lrs = lrs
		self.moms = moms
		self.nnmethod = nnmethod #"sub" or "full"
		self.RANDOM_SEED = 42
		#self.indexToHidden = None # will be set within trainAndTest()
		self.mention2Vec = None
		self.subSampleN = subsample # how many negative examples to have per positive (1 means equal amounts)
		self.penalty = penalty
		self.activation = activation

	def forwardprop(self, X, w_1, w_2, w_3, b_1, b_2, b_3, p_keep_input, p_keep_hidden):
		
		if self.activation == "relu":
			h1 = tf.nn.relu(tf.add(tf.matmul(tf.nn.dropout(X, p_keep_input), w_1), b_1))
			h1 = tf.nn.dropout(h1, p_keep_hidden)
			h2 = tf.nn.relu(tf.add(tf.matmul(h1, w_2), b_2))
			h2 = tf.nn.dropout(h2, p_keep_hidden)
			return tf.add(tf.matmul(h2, w_3), b_3)
		elif self.activation == "sigmoid":
			h1 = tf.nn.sigmoid(tf.add(tf.matmul(tf.nn.dropout(X, p_keep_input), w_1), b_1))
			h1 = tf.nn.dropout(h1, p_keep_hidden)
			h2 = tf.nn.sigmoid(tf.add(tf.matmul(h1, w_2), b_2))
			h2 = tf.nn.dropout(h2, p_keep_hidden)
			return tf.add(tf.matmul(h2, w_3), b_3)
		else:
			print "ERROR: wrong activation"
			exit(1)
			
		'''
		h1 = tf.nn.sigmoid(tf.matmul(tf.nn.dropout(X, p_keep_input), w_1))
		h1 = tf.nn.dropout(h1, p_keep_hidden)
		h2 = tf.nn.sigmoid(tf.matmul(h1, w_2))
		h2 = tf.nn.dropout(h2, p_keep_hidden)
		'''


	def loadVectorOnDemand(self, dm):
		m = self.helper.corpus.dmToMention[dm]

		# gets vector for the prev tokens
		prevTokens = self.helper.getMentionPrevTokenIndices(m, self.model.windowSize)
		prev_Vec = []
		for ti in prevTokens:
			cur_hidden = self.indexToHidden[ti]
			for i in range(len(cur_hidden)):
				prev_Vec.append(cur_hidden[i])
		# ensures vector is full size
		while len(prev_Vec) < self.model.windowSize*2*self.model.hidden_size:
			prev_Vec.insert(0, float(0))

		# gets vector for the mention
		avg_vec = [0]*2*self.model.hidden_size
		#print "mention: " + str(m)
		#print "\t" + str(m.corpusTokenIDs)
		for ti in m.corpusTokenIDs:
			cur_hidden = self.indexToHidden[ti]
			for i in range(len(cur_hidden)):
				avg_vec[i] = avg_vec[i] + cur_hidden[i]
		# normalizes
		for i in range(len(avg_vec)):
			avg_vec[i] = avg_vec[i] / len(m.corpusTokenIDs)

		# gets vector for the prev tokens
		nextTokens = self.helper.getMentionNextTokenIndices(m, self.model.windowSize)
		next_Vec = []
		for ti in nextTokens:
			if ti in self.indexToHidden.keys():
				cur_hidden = self.indexToHidden[ti]
				for i in range(len(cur_hidden)):
					next_Vec.append(cur_hidden[i])
			else:
				break

		# ensures vector is full size
		while len(next_Vec) < self.model.windowSize*2*self.model.hidden_size:
			next_Vec.append(float(0))
		print "prev: " + str(prev_Vec)
		print "men: " + str(avg_vec)
		print "next: " + str(next_Vec)
		return list(prev_Vec) + list(avg_vec) + list(next_Vec)

	def loadVectorPairsSubSampleLemma(self, dmPairs, method):
		start_time = time.time()
		dim = 2
		numPositives = 0
		numNegatives = 0
		for dmPair in dmPairs:
			label = self.helper.goldDMToTruth[dmPair]
			if label == 1:
				numPositives += 1
			elif label == 0:
				numNegatives += 1
		print "numP: " + str(numPositives)
		print "numN: " + str(numNegatives)
		num_examples = numPositives * (self.subSampleN + 1) # subSampleN negatives per 1 positive.

		x = np.ones((num_examples,dim))
		y = np.ndarray((num_examples,2)) # always a

		i = 0
		print "num_examples: " + str(num_examples)

		# populates the negatives
		num_filled = 0
		for dmPair in dmPairs:
			label = self.helper.goldDMToTruth[dmPair]
			if label == 0:
				if num_filled >= num_examples:
					break

				(dm1,dm2) = dmPair
				sameLemma = 0
				if self.helper.dmToLemma[dm1] == self.helper.dmToLemma[dm2]:
					sameLemma = 1

				sameLemma = self.helper.goldDMToTruth[dmPair]
				y[num_filled,0] = 1
				y[num_filled,1] = 0
				if sameLemma == 0:
					x[num_filled,0] = 1
					x[num_filled,1] = 0
				else:
					print "FOUND LEMMA"
					x[num_filled,0] = 0
					x[num_filled,1] = 1
				num_filled += 1

		# populates the positives
		positiveIndices = set()
		for dmPair in dmPairs:
			label = self.helper.goldDMToTruth[dmPair]
			if label == 1:
				(dm1,dm2) = dmPair
				sameLemma = 0
				if self.helper.dmToLemma[dm1] == self.helper.dmToLemma[dm2]:
					sameLemma = 1

				sameLemma = self.helper.goldDMToTruth[dmPair]
				randIndex = randint(0, num_examples-1)
				while randIndex in positiveIndices:
					randIndex = randint(0, num_examples-1)
				positiveIndices.add(randIndex)

				#print "randIndex: " + str(randIndex)
				y[randIndex,0] = 0
				y[randIndex,1] = 1
				if sameLemma == 0:
					x[randIndex,0] = 1
					x[randIndex,1] = 0
				else:
					x[randIndex,0] = 0
					x[randIndex,1] = 1

		print("*** LOADING VECTORS TOOK --- %s seconds ---" % (time.time() - start_time))
		return (x,y)

	def loadVectorPairsLemma(self, dmPairs, subset, method):
		start_time = time.time()
		dim = 2
		# gets num_examples
		num_examples = min(subset, len(dmPairs))
		if subset == -1:
			num_examples = len(dmPairs)

		x = np.ones((num_examples,dim))
		y = np.ndarray((num_examples,2)) # always a
		#print "making x size: " + str(num_examples) + " * " + str(dim+1)
		i = 0
		for dmPair in dmPairs:
			(dm1,dm2) = dmPair


			label = self.helper.goldDMToTruth[dmPair]
			if label == 0:
			    y[i,0] = 1
			    y[i,1] = 0
			elif label == 1:
			    y[i,0] = 0
			    y[i,1] = 1
			else:
			    print "ERROR: label was weird: " + str(label)
			    exit(1)
			
			sameLemma = 0
			if self.helper.dmToLemma[dm1] == self.helper.dmToLemma[dm2]:
				sameLemma = 1

			sameLemma = self.helper.goldDMToTruth[dmPair]
			if sameLemma == 0:
				x[i,0] = 1
				x[i,1] = 0
			else:
				x[i,0] = 0
				x[i,1] = 1

			i = i + 1
			if i == num_examples:
				break
		print("*** LOADING VECTORS TOOK --- %s seconds ---" % (time.time() - start_time))
		return (x,y)

	def loadVectorPairsSubSample(self, dmPairs, method):
		start_time = time.time()
		# gets dimension size
		dim = 0
		for dmPair in dmPairs:
			(dm1,dm2) = dmPair
			m1 = self.helper.corpus.dmToMention[dm1]
			vec1 = self.mention2Vec[m1]
			if method == "sub":
				dim = len(vec1)
			elif method == "full":
				dim = len(vec1)*2
			else:
				print "ERROR: not a valid method"
				exit(1)
			break

		numPositives = 0
		numNegatives = 0
		for dmPair in dmPairs:
			label = self.helper.goldDMToTruth[dmPair]
			if label == 1:
				numPositives += 1
			elif label == 0:
				numNegatives += 1
		print "numP: " + str(numPositives)
		print "numN: " + str(numNegatives)
		num_examples = numPositives * (self.subSampleN + 1) # subSampleN negatives per 1 positive.

		x = np.ones((num_examples,dim))
		y = np.ndarray((num_examples,2)) # always a
		#print "making x size: " + str(num_examples) + " * " + str(dim+1)
		i = 0
		print "num_examples: " + str(num_examples)

		# populates the negatives
		num_filled = 0
		for dmPair in dmPairs:
			label = self.helper.goldDMToTruth[dmPair]
			if label == 0:
				if num_filled >= num_examples:
					break
				(dm1,dm2) = dmPair
				m1 = self.helper.corpus.dmToMention[dm1]
				m2 = self.helper.corpus.dmToMention[dm2]
				vec1 = self.mention2Vec[m1]
				vec2 = self.mention2Vec[m2]
				y[num_filled,0] = 1
				y[num_filled,1] = 0

				if method == "sub":
					j = 0
					for i2 in range(len(vec1)):
						v = vec1[i2] - vec2[i2]
						x[num_filled,j] = v
						j = j + 1

				elif method == "full":
					j = 0
					for i2 in range(len(vec1)):
						x[num_filled,j] = vec1[i2]
						j = j + 1
					#start = len(vec1) # to save the time of looking up the length of vec1
					for i2 in range(len(vec2)):
						x[num_filled,j] = vec2[i2]
						j = j + 1

				num_filled += 1

		# populates the positives
		positiveIndices = set()
		for dmPair in dmPairs:
			label = self.helper.goldDMToTruth[dmPair]
			if label == 1:
				randIndex = randint(0, num_examples-1)
				while randIndex in positiveIndices:
					randIndex = randint(0, num_examples-1)
				positiveIndices.add(randIndex)

				#print "randIndex: " + str(randIndex)
				y[randIndex,0] = 0
				y[randIndex,1] = 1

				(dm1,dm2) = dmPair
				m1 = self.helper.corpus.dmToMention[dm1]
				m2 = self.helper.corpus.dmToMention[dm2]

				vec1 = self.mention2Vec[m1]
				vec2 = self.mention2Vec[m2]

				if method == "sub":
					j = 0
					for i2 in range(len(vec1)):
						v = vec1[i2] - vec2[i2]
						x[randIndex,j] = v
						j = j + 1

				elif method == "full":
					j = 0
					for i2 in range(len(vec1)):
						x[randIndex,j] = vec1[i2]
						j = j + 1
					start = len(vec1) # to save the time of looking up the length of vec1
					for i2 in range(len(vec2)):
						x[randIndex, start+j] = vec2[i2]
						j = j + 1
		print("*** LOADING VECTORS TOOK --- %s seconds ---" % (time.time() - start_time))
		'''
		print "# rand: " + str(positiveIndices)
		for p in positiveIndices:
			print p
		'''
		return (x,y)

	def loadVectorPairs(self, dmPairs, subset, method):
		start_time = time.time()

		# gets dimension size
		dim = 0
		for dmPair in dmPairs:
			(dm1,dm2) = dmPair
			m1 = self.helper.corpus.dmToMention[dm1]
			vec1 = self.mention2Vec[m1]
			if method == "sub":
				dim = len(vec1)
			elif method == "full":
				dim = len(vec1)*2
			else:
				print "ERROR: not a valid method"
				exit(1)
			break

		# gets num_examples
		num_examples = min(subset, len(dmPairs))
		if subset == -1:
			num_examples = len(dmPairs)

		x = np.ones((num_examples,dim))
		y = np.ndarray((num_examples,2)) # always a
		#print "making x size: " + str(num_examples) + " * " + str(dim+1)
		i = 0
		for dmPair in dmPairs:
			(dm1,dm2) = dmPair
			m1 = self.helper.corpus.dmToMention[dm1]
			m2 = self.helper.corpus.dmToMention[dm2]

			vec1 = self.mention2Vec[m1]
			vec2 = self.mention2Vec[m2]

			label = self.helper.goldDMToTruth[dmPair]
			if label == 0:
			    y[i,0] = 1
			    y[i,1] = 0
			elif label == 1:
			    y[i,0] = 0
			    y[i,1] = 1
			else:
			    print "ERROR: label was weird: " + str(label)
			    exit(1)

			if method == "sub":
				j = 0
				for i2 in range(len(vec1)):
					v = vec1[i2] - vec2[i2]
					x[i,j] = v
					j = j + 1

			elif method == "full":
				j = 0
				for i2 in range(len(vec1)):
					x[i,j] = vec1[i2]
					j = j + 1
				start = len(vec1) # to save the time of looking up the length of vec1
				for i2 in range(len(vec2)):
					x[i, start+j] = vec2[i2]
					j = j + 1

			else:
				print "ERROR: not a valid method"
				exit(1)

			i = i + 1
			if i == num_examples:
				break
		print("*** LOADING VECTORS TOOK --- %s seconds ---" % (time.time() - start_time))
		return (x,y)

	def calculateFeatures(self, dm1, dm2):
		v1 = self.loadVectorOnDemand(dm1)
		v2 = self.loadVectorOnDemand(dm2)
		if self.nnmethod == "sub":
			ret = (self.vectorLength + 1)*[1.0]
			for i in range(len(v1)):
				ret[i] = (float(v1[i]) - float(v2[i]))
			return ret
		elif self.nnmethod == "full":
			ret = np.ones(2*len(v1)+1)
			for i in range(len(v1)):
				ret[i] = float(v1[i])
			for i in range(len(v2)):
				ret[len(v1) + i] = float(v2[i])
			return ret
		else:
			print "ERROR: wrong nnmethod"
			exit(1)
	
	def init_weights(self, shape):
		""" Weight initialization """
		print shape
		weights = tf.random_normal(shape, stddev=0.1)
		return tf.Variable(weights)

	def calculateScore(self, gold, predictions):

		accuracy = np.mean(np.argmax(gold, axis=1) == predictions)
		golds_flat = np.argmax(gold, axis=1)

		num_predicted_true = 0
		num_predicted_false = 0
		num_golds_true = 0
		num_correct = 0
		for i in range(len(golds_flat)):
			if golds_flat[i] == 1:
				num_golds_true = num_golds_true + 1

		for i in range(len(predictions)):
			if predictions[i] == 1:
				num_predicted_true = num_predicted_true + 1
				if golds_flat[i] == 1:
					num_correct = num_correct + 1
			else:
				num_predicted_false += 1
		recall = float(num_correct) / float(num_golds_true)
		prec = 0
		if num_predicted_true > 0:
			prec = float(num_correct) / float(num_predicted_true)
		f1 = 0
		if prec > 0 or recall > 0:
			f1 = 2*float(prec * recall) / float(prec + recall)
		
		print "------"
		print "num_golds_true: " + str(num_golds_true) + "; num_predicted_false: " + str(num_predicted_false) + "; num_predicted_true: " + str(num_predicted_true) + " (of these, " + str(num_correct) + " actually were)"
		print "recall: " + str(recall) + "; prec: " + str(prec) + "; f1: " + str(f1) + "; accuracy: " + str(accuracy)		
		return (accuracy, f1)

	def softmax_weighted(tensor_in, labels, weights, biases, class_weight=None, name=None):
		logits = nn.xw_plus_b(tensor_in, weights, biases)
		if class_weight is not None:
			logits = math_ops.multiply(logits, class_weight)
		return nn.softmax(logits), loss_ops.softmax_cross_entropy(logits, labels)

	def engine(self, h1size, h2size, learning_rate, mom, num_epochs, keep_input, keep_hidden, optimizer, predictionsOut): 
		
		'''
		print "pred we will eventually write to (in train() now): " + str(predictionsOut)
		if os.path.isfile(predictionsOut):
			print("we already have predictions file:" + predictionsOut)
			exit(1)
		'''
		'''
		print h1size
		print h2size
		print learning_rate
		print mom
		print num_epochs
		print keep_input
		print keep_hidden
		print optimizer
		print "-------------------"
		print "CONFIG: " + str(predictionsOut)
		'''
		start_time = time.time()
		#with tf.device("/" + str(dev_name) + ":0"):

		#train_X, train_y = self.loadVectors(self.trainingDMPairs, self.subset, self.nnmethod)

		#x_size = train_X.shape[1]
		#y_size = train_y.shape[1]
		#print "x_dim: " + str(self.vectorLength)
		y_size = 2
		X = tf.placeholder("float", shape=[None, self.vectorLength]) #shape=[self.vectorLength + 1])
		y = tf.placeholder("float", shape=[None,2] ) #shape=[y_size])

		w_1 = self.init_weights((self.vectorLength, h1size))
		w_2 = self.init_weights((h1size, h2size))
		w_3 = self.init_weights((h2size, y_size))
		b_1 = self.init_weights((1,h1size))
		b_2 = self.init_weights((1,h2size))
		b_3 = self.init_weights((1,y_size))

		p_keep_input = tf.placeholder("float")
		p_keep_hidden = tf.placeholder("float")

		yhat = self.forwardprop(X, w_1, w_2, w_3, b_1, b_2, b_3, p_keep_input, p_keep_hidden)
		predict = tf.argmax(yhat, dimension=1)
		

		#print "yhat:"
		#yhat = tf.Print(yhat, [yhat], message="This is a: ")
		#cost = tf.reduce_mean()
		cost = tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(yhat, y, self.penalty))
		#cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(yhat, y))#+ 0.01*tf.nn.l2_loss(w_1) + 0.01*tf.nn.l2_loss(w_2) + 0.01*tf.nn.l2_loss(w_3)
		print "softmax: "
		#result = sess.run(tf.nn.softmax_cross_entropy_with_logits(yhat, y))
		#print(result)
		if optimizer == "gd":
			updates = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
		elif optimizer == "rms":
			updates = tf.train.RMSPropOptimizer(learning_rate, momentum=mom).minimize(cost)
		elif optimizer == "ada":
			updates = tf.train.AdagradOptimizer(learning_rate).minimize(cost)
		else:
			print "ERROR: optimzer not recognized"
			exit(1)

		# Run NN
		sess = tf.Session()
		init = tf.global_variables_initializer() #tf.initialize_all_variables()

		sess.run(init)
		print("* and awwwaaaaaay we go! (nn training)")
		
		# pre-load feature vectors way (shoudl be used when corpus is -3)
		(trainX, trainY) = self.loadVectorPairsSubSampleLemma(self.helper.trainingDMPairs, self.nnmethod)
		print "trainX subsampled!"
		print trainX.shape
		print trainY.shape
		print trainX
		print trainY
		(devX, devY) = self.loadVectorPairsLemma(self.helper.devDMPairs, -1, self.nnmethod)
		for epoch in range(num_epochs):
			e_time = time.time()

			sess.run(updates, feed_dict={X: trainX, y: trainY, p_keep_input: keep_input, p_keep_hidden: keep_hidden})
			trainPredictions = sess.run(predict, feed_dict={X: trainX, y: trainY, p_keep_input: 1.0, p_keep_hidden: 1.0})
			devPredictions = sess.run(predict, feed_dict={X: devX, y: devY, p_keep_input: 1.0, p_keep_hidden: 1.0})

			'''
			print "vecs"
			for xy in range(len(trainX)):
				print "xy: " + str(xy) + str(trainY[xy])
			print "trainY: " + str(trainY)
			print "devY: " + str(devY)
			print "trainPredictions: " + str(trainPredictions)
			print "devpred: " + str(devPredictions)
			'''
			print "TRAIN:"
			(train_accuracy, train_f1) = self.calculateScore(trainY, trainPredictions)
			print "\nDEV:"
			(dev_accuracy, dev_f1) = self.calculateScore(devY, devPredictions)
			print "epoch "+ str(epoch) + " took " + str(round(time.time() - e_time, 1)) + " secs -- f1 (dev set): " + str(dev_f1) + ", " + str(dev_accuracy) + " -- (train set): " + str(train_f1) + ", " + str(train_accuracy)
			sys.stdout.flush()
		sess.close()

		''' ON DEMAND WAY (shoudl be used when corpus is -1)
		for epoch in range(num_epochs):
			e_time = time.time()
			
			# train an epoch
			#print str(len(self.helper.trainingDMPairs))
			i = 0
			for dmPair in self.helper.trainingDMPairs:

				(dm1, dm2) = dmPair
				trainX = []
				trainX.append(self.calculateFeatures(dm1, dm2))
				trainY = []
				goldY = self.helper.goldDMToTruth[dmPair]
				if goldY == 0:
					trainY.append([1.0, 0.0])
				elif goldY == 1:
					trainY.append([0.0, 1.0])
				else:
					print "prediction value isn't 0 or 1"
					exit(1)

				sess.run(updates, feed_dict={X: trainX, y: trainY, p_keep_input: keep_input, p_keep_hidden: keep_hidden})
				i = i + 1

			# dev test
			num_correct = 0
			num_golds_true = 0
			num_predicted_true = 0
			i = 0
			for dmPair in self.helper.devDMPairs:
				(dm1, dm2) = dmPair
				# x
				trainX = []
				trainX.append(self.calculateFeatures(dm1, dm2))

				# y
				goldY = self.helper.goldDMToTruth[dmPair]
				predY = sess.run(predict, feed_dict={X: trainX, p_keep_input: 1.0, p_keep_hidden: 1.0})
				if goldY == 1:
					num_golds_true = num_golds_true + 1
				if predY == 1:
					if goldY == 1:
						num_correct = num_correct + 1
					num_predicted_true = num_predicted_true + 1
				i = i+1

			dev_recall = float(num_correct) / float(num_golds_true)
			dev_prec = 0
			if num_predicted_true > 0:
				dev_prec = float(num_correct) / float(num_predicted_true)
			dev_f1 = 0
			if dev_prec > 0 or dev_recall > 0:
				dev_f1 = 2*float(dev_prec * dev_recall) / float(dev_prec + dev_recall)
			print "epoch "+ str(epoch) + " took " + str(round(time.time() - e_time, 1)) + " secs -- f1 (dev set): " + str(dev_f1)
			sys.stdout.flush()
		
		# test set
		num_correct = 0
		num_incorrect = 0
		for dmPair in self.helper.testingDMPairs:
			(dm1, dm2) = dmPair

			# x
			trainX = []
			trainX.append(self.calculateFeatures(dm1, dm2))

			# y
			goldY = self.helper.goldDMToTruth[dmPair]
			predY = sess.run(predict, feed_dict={X: trainX, p_keep_input: 1.0, p_keep_hidden: 1.0})
			if goldY == 1:
				num_golds_true = num_golds_true + 1
			if predY == 1:
				if goldY == 1:
					num_correct = num_correct + 1
				num_predicted_true = num_predicted_true + 1
		test_recall = float(num_correct) / float(num_golds_true)
		test_prec = 0
		if num_predicted_true > 0:
			test_prec = float(num_correct) / float(num_predicted_true)
		test_f1 = 0
		if test_prec > 0 or test_recall > 0:
			test_f1 = 2*float(test_prec * test_recall) / float(test_prec + test_recall)
		print "(final) epoch " + str(epoch) + " took " + str(round(time.time() - start_time, 1)) + " secs -- f1 (test set):" + str(test_f1)
		print "CONFIG: " + str(predictionsOut)
		print "-------------------"
		sys.stdout.flush()
		sess.close()
		'''


	def trainAndTest(self):

		#self.indexToHidden = self.model.train() # trains the wordEmbeddings model
		self.mention2Vec = self.model.train()

		h1 = self.hiddens[0]
		h2 = self.hiddens[1]
		k1 = self.keep_inputs[0]
		k2 = self.keep_inputs[1]
		if self.optimizers == "rms":
			predictionsOut = self.baseOut + "_o" + str(self.optimizers) + "_h" + str(h1) + "_" + str(h2) + "_k" + str(k1) + "_" + str(k2) + "_ne" + str(self.num_epochs) + "_lr" + str(self.lrs) + "_mom" + str(self.moms) + ".predictions"
			self.engine(h1, h2, self.lrs, self.moms, self.num_epochs, k1, k2, self.optimizers, predictionsOut)
		elif self.optimizers == "gd" or self.optimizers == "ada":
			predictionsOut = self.baseOut + "_o" + str(self.optimizers) + "_h" + str(h1) + "_" + str(h2) + "_k" + str(k1) + "_" + str(k2) + "_ne" + str(self.num_epochs) + "_lr" + str(self.lrs) + ".predictions"
			self.engine(h1, h2, self.lrs, 0, self.num_epochs, k1, k2, self.optimizers, predictionsOut)
		else:
			print "ERROR: we dont recognize the opt"
			exit(1)
