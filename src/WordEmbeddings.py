import sys
sys.path.append('/gpfs/main/home/christanner/.local/lib/python2.7/site-packages/tensorflow/')
import tensorflow as tf
import numpy as np
import collections
import time
import operator
import math

class WordEmbeddings:

	def __init__(self, corpus, helper, vectorFile, windowSize, dmsInGold):
		self.corpus = corpus
		self.helper = helper
		self.typeToVector = {}
		self.windowSize = windowSize
		self.dmsInGold = dmsInGold
		self.hidden_size = 0

		print("we care about " + str(len(self.dmsInGold)) + " dms")

		f = open(vectorFile, 'r')		
		for line in f:
			tokens = line.rstrip().split(" ")
			if len(tokens) < 2:
				continue
			word = tokens[0].lower()
			vec = []
			for v in tokens[1:]:
				if v[-1] == ",":
					vec.append(float(v[:-1]))
				else:
					vec.append(float(v))
			self.typeToVector[word] = vec
			self.hidden_size = len(vec)
		print str(len(self.typeToVector.keys())) + " unique types"

	def train(self):
		mentionToVec = {}
		for dm in self.corpus.dmToREF.keys():
			if dm not in self.dmsInGold:
				continue

			m = self.corpus.dmToMention[dm]
			
			#print "mention: " + str(m)
			
			# gets vector for the prev tokens
			prevTokens = self.helper.getMentionPrevTokenIndices(m, self.windowSize)
			prev_Vec = []
			for ti in prevTokens:

				curToken = self.corpus.corpusTokens[ti]
				token_text = curToken.text.lower()
				#print("\t " + str(token_text))
				if token_text not in self.typeToVector.keys():
					print("*** we don't have " + str(token_text))
				else:
					cur_hidden = self.typeToVector[token_text]
					for i in range(len(cur_hidden)):
						prev_Vec.append(cur_hidden[i])

			# ensures vector is full size
			while len(prev_Vec) < self.windowSize*self.hidden_size:
				prev_Vec.insert(0, float(0))

			# gets vector for the mention
			avg_vec = [0]*self.hidden_size
			num_found = 0
			for token in m.tokens:
				token_text = token.text.lower()
				#print("\t *" + str(token_text))
				if token_text not in self.typeToVector.keys():
					print("*** we don't have " + str(token_text))
				else:
					num_found = num_found + 1
					cur_hidden = self.typeToVector[token_text]
					for i in range(len(cur_hidden)):
						avg_vec[i] = avg_vec[i] + cur_hidden[i]
			# normalizes
			if num_found > 0:
				for i in range(len(avg_vec)):
					avg_vec[i] = avg_vec[i] / num_found

			# gets vector for the next tokens
			nextTokens = self.helper.getMentionNextTokenIndices(m, self.windowSize)
			next_Vec = []
			for ti in nextTokens:

				curToken = self.corpus.corpusTokens[ti]
				token_text = curToken.text.lower()
				#print("\t " + str(token_text))
				if token_text not in self.typeToVector.keys():
					print("*** we don't have" + str(token_text))
				else:
					cur_hidden = self.typeToVector[token_text]
					for i in range(len(cur_hidden)):
						next_Vec.append(cur_hidden[i])

			# ensures vector is full size
			while len(next_Vec) < self.windowSize*self.hidden_size:
				next_Vec.append(float(0))

			mentionToVec[m] = list(prev_Vec) + list(avg_vec) + list(next_Vec)
			#exit(1)
		return mentionToVec
