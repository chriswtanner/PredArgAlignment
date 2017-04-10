import sys
sys.path.append('/gpfs/main/home/christanner/.local/lib/python2.7/site-packages/tensorflow/')
import tensorflow as tf
import numpy as np
import collections
import time
import operator
import math
from LSTMModel_old import LSTMModel_old

# HOW TO RUN:
# insantiate: model = LSTM()
#     (where tokens should be extracted from the Parser)

carry_state = np.array([0], dtype = np.int32)
carry_val = 0
class LSTM_old:
	def __init__(self, corpus, helper, hidden_size, num_steps, num_epochs, batch_size, learning_rate, windowSize, isVerbose):
		# global vars
		self.sanityCheck = False
		self.helper = helper
		self.globalIDsToType = corpus.globalIDsToType
		self.typeToGlobalID = corpus.typeToGlobalID
		self.numUniqueTypes = len(self.typeToGlobalID.keys())
		self.corpus = corpus
		self.isVerbose = isVerbose
		# these are the DMs listed in the gold file; so, only these do we want to export features for
		self.dmsInGold = helper.getGoldDMs()

		# the type IDs of every token in the
		# corpus (the same length as the corpus)
		self.corpusTypeIDs = corpus.corpusTypeIDs
		self.corpusTokens = corpus.corpusTokens # actually Tokens

		# LSTM -- architectural params
		self.batch_size = batch_size
		self.num_steps = num_steps
		self.hidden_size = hidden_size
		self.num_layers = 2

		# LSTM -- learning params
		self.max_grad_norm = 5
		self.decay_rate = 0.5
		self.learning_rate = learning_rate #0.8
		self.num_epochs = num_epochs #50 # the total number of epochs for training
		self.num_init_epoch = 15 # the number of epochs trained with the initial learning rate

		# temp, hard-coded vals
		self.windowSize = windowSize # 2 and 6

	def train(self):

		if self.isVerbose:
			print "*** corpus' # of unique types: " + str(self.numUniqueTypes)
			print "*** corpus' # of raw tokens: " + str(len(self.corpusTokens))
		with tf.Graph().as_default(), tf.Session() as session:
			initializer = tf.random_uniform_initializer(-0.10, 0.10)

			with tf.variable_scope("model", reuse = None, initializer = initializer):
				mtrain = LSTMModel_old(self.numUniqueTypes, self.batch_size, self.num_steps, self.hidden_size, self.num_layers, self.max_grad_norm, self.decay_rate, is_training = True)

			with tf.variable_scope("model", reuse = True, initializer = initializer): 
				mtest = LSTMModel_old(self.numUniqueTypes, 1, 1, self.hidden_size, self.num_layers, self.max_grad_norm, self.decay_rate, is_training = False)

			tf.global_variables_initializer().run()
			
			# Add ops to save and restore all the variables.
			saver = tf.train.Saver()
			start_time = time.time()
			for i in range(self.num_epochs):
				e_time = time.time()
				sys.stdout.flush()
				lr_decay = self.decay_rate ** max(i-self.num_init_epoch, 0.0)
				session.run(tf.assign(mtrain._lr, self.learning_rate*lr_decay))

				train_perplexity = run_epoch(self, session, mtrain, self.corpusTypeIDs, self.corpusTokens, mtrain._train_op, self.corpus, self.isVerbose, write=False)
				if self.isVerbose:
					print "(LSTM embedding training) epoch " + str(i) + " took " + str(round(time.time() - e_time, 1)) + " secs -- train_perplexity: " + str(train_perplexity)
				sys.stdout.flush()
			if self.isVerbose:
				print("*** LSTM training took a total of " + str(round(time.time() - start_time, 1)) + " secs")
			#test_perplexity, indexToHidden) = run_epoch(self, session, mtest, self.corpusTypeIDs, self.corpusTokens, tf.no_op(), self.corpus, self.isVerbose, write=True)
			(test_perplexity, mentionToVec) = run_epoch(self, session, mtest, self.corpusTypeIDs, self.corpusTokens, tf.no_op(), self.corpus, self.isVerbose, write=True)
			
			#print "size of mentionToVec: " + str(len(mentionToVec.keys()))
			if self.isVerbose:
				print "*** LSTM test perplexity: " + str(test_perplexity)
			#return indexToHidden
			return mentionToVec
	# returns cosine sim. b/w 2 vectors: a and b
	def getCosineSim(self, a, b):

		numerator = 0
		denomA = 0
		denomB = 0
		for i in range(len(a)):
			numerator = numerator + a[i]*b[i]
			denomA = denomA + (a[i]*a[i])
			denomB = denomB + (b[i]*b[i])	
		return float(numerator) / (float(math.sqrt(denomA)) * float(math.sqrt(denomB)))

		#exit(1)
		#print("# statesizes: " + str(len(state_sizes)))
		#print(state_sizes)
		#print("len saved states: " + str(len(saved_states)))
		#print("size of savedstates[0]" + str(len(saved_states[0])))			

def ptb_iterator(raw_data, batch_size, num_steps):
	# new way we're trying
	'''
	raw_data = np.array(raw_data, dtype=np.int32)
  	data_len = len(raw_data)
  	num_batches = data_len // batch_size
  	data = np.zeros([num_batches, batch_size], dtype=np.int32)

  	#print("rawdata len: " + str(len(raw_data)))
  	#print("batch size: " + str(batch_size))
  	#print("iterator num_batches:" + str(num_batches))

  	for i in range(num_batches):
		data[i] = raw_data[batch_size * i:batch_size * (i + 1)]
	epoch_size = (num_batches - 1) // num_steps
	#print "epoch_size:" + str(epoch_size)

	if epoch_size == 0:
		raise ValueError("epoch_size == 0, decrease batch_size or num_steps")

	for i in range(epoch_size):
		x = data[i*num_steps:(i+1)*num_steps, :]
		y = data[i*num_steps+1:(i+1)*num_steps+1, :]
		yield (x, y)
	'''

	# old way which doesn't crash
	raw_data = np.array(raw_data, dtype=np.int32)
  	data_len = len(raw_data)
  	num_batches = data_len // batch_size
  	data = np.zeros([batch_size, num_batches], dtype=np.int32)

  	for i in range(batch_size):
		data[i] = raw_data[num_batches * i:num_batches * (i + 1)]
	epoch_size = (num_batches - 1) // num_steps
	# print "epoch_size:" + str(epoch_size)

	if epoch_size == 0:
		raise ValueError("epoch_size == 0, decrease batch_size or num_steps")

	for i in range(epoch_size):
		x = data[:, i*num_steps:(i+1)*num_steps]
		y = data[:, i*num_steps+1:(i+1)*num_steps+1]
		yield (x, y)

def run_epoch(self, sess, model, tokenIDs, tokens, eval_op, corpus, isVerbose, write):
	global carry_state
	if carry_state.all() == 0:
		state = sess.run(model._initial_state)
	else:
		if write == True:
			#state = carry_state # i should comment this out
			state = carry_state[model.batch_size-1]
			state = np.reshape(state, (1,(2+model.num_layers)*model.hidden_size))
			#print("* WRITING out the vectors to: " + modelOutputFile)
		else:
			state = carry_state
	#print("staet: " + str(len(state)))
	#print(state)
	#print("hidden: " + str(model.hidden_layer))

	tmp_data = np.array(tokenIDs, dtype = np.int32)
	formatted_data = np.zeros([len(tmp_data), model.num_steps], dtype = np.int32)

	for i in range(len(tmp_data)):
		formatted_data[i] = tmp_data[i]

	''' # SWITCH: writes out the .allvectors
	if write:
		f = open(modelOutputFile, 'w')
	'''

	costs = 0.0
	iters = 0

	#print("tokenIDs: " + str(tokenIDs))
	tuples = ptb_iterator(tokenIDs, model.batch_size, model.num_steps)
	self.indexToHidden = {}
	for step, (x, y) in enumerate(tuples):
		'''
		print("x: " + str(x) + "; y: " + str(y))
		
		'''
		cost, state, hidden_layer, logits, probDistr, _ = sess.run([model._cost, model._final_state, model.hidden_layer, model.logits, model.probDistr, eval_op], 
				{model._input_data: x, model._targets: y, model._initial_state: state})

		costs += cost
		iters += model.num_steps

		if write:
			'''
			layerA=state[0][0:model.hidden_size]
			layerB=state[0][model.hidden_size:(2*model.hidden_size)]
			layerC=state[0][2*model.hidden_size:3*model.hidden_size]
			layerD=state[0][3*model.hidden_size:]
			'''
			layer1=state[0][model.hidden_size:(2*model.hidden_size)]
			layer2=state[0][3*model.hidden_size:]
			
			hidden_layers = list(layer1) + list(layer2)

			#print "step: [" + str(step) + "] gets the hidden's from the tokenIDs: " + str(x) + ", whose golden labels are " + str(y)
			
			self.indexToHidden[step] = hidden_layers
			outLine = ""
			for i in hidden_layers:
				outLine = outLine + str(i) + ","
			outLine = outLine[:-1]
			# f.write(outLine + "\n") SWITCH: writes out the .allvectors

		carry_state = state

	if write == False:
		return np.exp(costs/iters)
	else:

		if isVerbose and self.sanityCheck:
			mentionToVec = {}
			for dm in corpus.dmToREF.keys():
				if dm not in self.dmsInGold:
					print "skipping " + str(dm)
					continue
			#for m in corpus.mentions:
				m = corpus.dmToMention[dm]
				avg_vec = [0]*2*self.hidden_size
				for ti in m.corpusTokenIndices:
					cur_hidden = self.indexToHidden[ti]
					for i in range(len(cur_hidden)):
						avg_vec[i] = avg_vec[i] + cur_hidden[i]
				# normalizes
				for i in range(len(avg_vec)):
					avg_vec[i] = avg_vec[i] / len(m.corpusTokenIndices)
					mentionToVec[m] = avg_vec

			for m1 in mentionToVec.keys():
				distances = {}
				m1vec = mentionToVec[m1]
				print "MENTION: " + str(m1)
				for m2 in mentionToVec.keys():
					if m2 not in distances and m1 != m2 and m1.suffix == m2.suffix:
						m2vec = mentionToVec[m2]
						cosine = self.getCosineSim(m1vec, m2vec)
						distances[m2] = cosine

				'''
				print "\tWORST:"
				sorted_distances = sorted(distances.items(), key=operator.itemgetter(1))
				i = 0
				for d in sorted_distances:
					if i > 7:
						break
					m2 = d[0]
					cosine = d[1]
					print("\t" + str(m2) + " = " + str(cosine))
					i = i + 1
				'''
				print "\tBEST:"
				sorted_distances = sorted(distances.items(), key=operator.itemgetter(1), reverse=True)
				i = 0
				for d in sorted_distances:
					if i > 7:
						break
					m2 = d[0]
					cosine = d[1]
					print("\t" + str(m2) + " = " + str(cosine))
					#i = i + 1
			#print("total pairs: " + str(len(sorted_distances)))
		# f.close() SWITCH: writes out the .allvectors

		# returns the mentionToVec
		# (only the mentions that have REFs -- i.e., are co-refed -- and are in our goldTruth file,
		#  otherwise we are wasting memory and disk space)

		
		mentionToVec = {}
		for dm in corpus.dmToREF.keys():
			if dm not in self.dmsInGold:
				continue

			m = corpus.dmToMention[dm]

			# gets vector for the prev tokens
			prevTokens = self.helper.getMentionPrevTokenIndices(m, self.windowSize)
			prev_Vec = []
			for ti in prevTokens:
				cur_hidden = self.indexToHidden[ti]
				for i in range(len(cur_hidden)):
					prev_Vec.append(cur_hidden[i])

			# ensures vector is full size
			while len(prev_Vec) < self.windowSize*2*model.hidden_size:
				prev_Vec.insert(0, float(0))

			# gets vector for the mention
			avg_vec = [0]*2*model.hidden_size
			#print "mention: " + str(m)
			#print "\t" + str(m.corpusTokenIndices)
			for ti in m.corpusTokenIndices:
				cur_hidden = self.indexToHidden[ti]
				for i in range(len(cur_hidden)):
					avg_vec[i] = avg_vec[i] + cur_hidden[i]
			
			# normalizes
			for i in range(len(avg_vec)):
				avg_vec[i] = avg_vec[i] / len(m.corpusTokenIndices)

			# gets vector for the prev tokens
			nextTokens = self.helper.getMentionNextTokenIndices(m, self.windowSize)
			next_Vec = []
			for ti in nextTokens:

				if ti in self.indexToHidden.keys():
					cur_hidden = self.indexToHidden[ti]
					for i in range(len(cur_hidden)):
						next_Vec.append(cur_hidden[i])
				else:
					break

			# ensures vector is full size
			while len(next_Vec) < self.windowSize*2*model.hidden_size:
				next_Vec.append(float(0))

			mentionToVec[m] = list(prev_Vec) + list(avg_vec) + list(next_Vec)

			'''
			print "-----------"
			print m
			print "prev: " + str(prevTokens)
			for i in prevTokens:
				print self.corpusTokens[i]
			print m.corpusTokenIndices
			for i in m.corpusTokenIndices:
				print self.corpusTokens[i]
			print "next: " + str(nextTokens)
			for i in nextTokens:
				print self.corpusTokens[i]
			print "-----------"
			'''

		return (np.exp(costs/iters), mentionToVec)
		#return (np.exp(costs/iters), self.indexToHidden)