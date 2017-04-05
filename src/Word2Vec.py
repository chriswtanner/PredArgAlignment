import sys
sys.path.append('/gpfs/main/home/christanner/.local/lib/python2.7/site-packages/tensorflow/')
import tensorflow as tf
import numpy as np
import collections
import time
import operator
import math

class Word2Vec:

	def __init__(self, corpus, helper, vectorFile, dmsInGold):
		self.corpus = corpus
		self.typeToVector = {}
		self.hidden_size = 0

		self.dmsInGold = dmsInGold
		print("we care about " + str(len(self.dmsInGold)) + " dms")

		f = open(vectorFile, 'r')
		
		for line in f:
			tokens = line.rstrip().split(" ")
			if len(tokens) < 2:
				continue
			word = tokens[0].lower()
			vec = []
			for v in tokens[1:]:
				vec.append(float(v[:-1]))

			self.typeToVector[word] = vec
			self.hidden_size = len(vec)
		print str(len(self.typeToVector.keys())) + " unique types"

	def getVectors(self):
		mentionToVec = {}
		for dm in self.corpus.dmToREF.keys():
			if dm not in self.dmsInGold:
				continue
			m = self.corpus.dmToMention[dm]
			avg_vec = [0]*self.hidden_size
			num_found = 0
			for token in m.tokens:
				token_text = token.text.lower()
				if token_text not in self.typeToVector.keys():
					print "*** we don't have" + str(token_text)
				else:
					num_found = num_found + 1
					cur_hidden = self.typeToVector[token_text]
					for i in range(len(cur_hidden)):
						avg_vec[i] = avg_vec[i] + cur_hidden[i]
			# normalizes
			if num_found > 0:
				for i in range(len(avg_vec)):
					avg_vec[i] = avg_vec[i] / num_found
					mentionToVec[m] = avg_vec
		return mentionToVec
