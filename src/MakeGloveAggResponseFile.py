import sys
import numpy as np
import sklearn
from sklearn.cluster import AgglomerativeClustering
from ECBParser import ECBParser
from LSTM_old import LSTM_old
from collections import defaultdict

# PURPOSE: this branch from Test allows us to:
# 	(1) write out GloVe's agglomerative, such that the CoNLL scorer can evaluate.
#   NOTE: we deliberately cheat to use K=the ideal # of clusters per directory
# REQUIRES:
#	(1) HDDCRP's output for the entire corpus (that is, we'd have to run HDDCRP on all dirs as its test sets, then cat to make 1 big file, e.g., 'hddcrp_all_output.txt')
class MakeGloveAggResponseFile:

	if __name__ == "__main__":

		corpus = "test" # or test

		validDirs = range(23,26)
		if corpus == "test":
			validDirs = range(26,46)

		windowSize = 3
		goldEventsFile = "/Users/christanner/research/PredArgAlignment/data/goldEvents.txt" # used to map from HDDCRP's dm format to our DM format
		gloveFile = "/Users/christanner/research/PredArgAlignment/data/glove_nonstitched_400.txt"
		goldKeysFile = "/Users/christanner/research/PredArgAlignment/data/" + corpus + ".keys" # just to know the ordering of the output
		responseKeysfile = "/Users/christanner/research/PredArgAlignment/data/" + corpus + "_glove_agg_w" + str(windowSize) + ".response"

		# loads glove word embeddings
		typeToVector = {}
		wordTypes = set() # same as typeToVector.keys() but faster since iet's a set
		hidden_size = 0
		f = open(gloveFile, 'r')
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
			typeToVector[word] = vec
			hidden_size = len(vec)
			wordTypes.add(word)
		f.close()

		#iterate through goldEvents and populate dirToNumGoldClusters = {}, whereas the key will be dirNum_suffix (e.g., 1_ecbplus.xml)
		dirToGoldREFs = defaultdict(set)
		dirToNumGoldClusters = {}
		f = open(goldEventsFile, 'r')
		for line in f:
			(dirNum, ref, doc_id, m_id, lemma, head, sent_num, start_token, end_token, d, e) = line.rstrip().split(";")
			if int(dirNum) not in validDirs:
				continue
			suffix = doc_id[doc_id.index("ecb"):]
			key = dirNum + "_" + suffix
			dirToGoldREFs[key].add(ref)
		f.close()
		for k in dirToGoldREFs.keys():
			dirToNumGoldClusters[k] = len(dirToGoldREFs[k])

		currentHighestCluster = 0
		dmToAggClusterNum = {} # stores the unique cluster that the dm belongs to
		print "hidden size: " + str(hidden_size)
		for key in dirToNumGoldClusters.keys():
			f = open(goldEventsFile, 'r')
			mentionVectors = [] # represents all of the mentions for hte current dir
			DMs = [] # represents the correct order of the DMs for the current dir
			for line in f:
				(dirNum, ref, doc_id, m_id, lemma, head, sent_num, start_token, end_token, ids, tok) = line.rstrip().split(";")
				if int(dirNum) not in validDirs:
					continue
				suffix = doc_id[doc_id.index("ecb"):]
				current_key = dirNum + "_" + suffix
				if current_key != key:
					continue

				dm = doc_id + ";" + m_id
				tokens = tok.split(" ")
				m_ids = []
				for _ in ids.split(" "):
					m_ids.append(int(_))
				m_ids.sort()
				#print "m_ids: " + str(m_ids)
				m_low = m_ids[0]
				m_high = m_ids[-1]

				# prev tokens
				prev_tokens = []
				for i in range(windowSize):
					m = m_low - i - 1
					if m < 0 or m > len(tokens) - 1 or tokens[m] not in wordTypes:
						for _ in range(hidden_size):
							prev_tokens.append(float(0.0))
					else:
						for _ in typeToVector[tokens[m]]:
							prev_tokens.append(_)

				# sums the mention itself
				mention_tokens = [0]*hidden_size
				for m in m_ids:
					if m < 0 or m > len(tokens) - 1 or tokens[m] not in wordTypes:
						for _ in range(hidden_size):
							mention_tokens[_] += float(0.0)
					else:
						wordEmb = typeToVector[tokens[m]]
						for _ in range(len(wordEmb)):
							mention_tokens[_] += wordEmb[_]
				# avgs the mention vector
				if len(m_ids) > 1:
					for i in range(len(mention_tokens)):
						mention_tokens[i] = mention_tokens[i] / len(m_ids)
				# next tokens
				next_tokens = []
				for i in range(windowSize):
					m = m_high + i + 1
					if m < 0 or m > len(tokens) - 1 or tokens[m] not in wordTypes:
						for _ in range(hidden_size):
							next_tokens.append(float(0.0))
					else:
						for _ in typeToVector[tokens[m]]:
							next_tokens.append(_)
				
				all_tokens = list(prev_tokens) + list(mention_tokens) + list(next_tokens)
				DMs.append(dm)
				mentionVectors.append(all_tokens)
			f.close()
			print "# vectors " + str(len(mentionVectors)) + " should cluster to " + str(dirToNumGoldClusters[key]) + " clusters"				
			print "length of each mentionVectors: " + str(len(mentionVectors[0]))
			#print(mentionVectors)
			n = int(dirToNumGoldClusters[key])
			agg = AgglomerativeClustering(n_clusters=n).fit(mentionVectors)
			for i in range(len(DMs)):
				cur_dm = DMs[i]
				cur_label = agg.labels_[i]
				dmToAggClusterNum[cur_dm] = currentHighestCluster + cur_label
			currentHighestCluster += n

		# reads the .keys file so that we can:
		# (1) output our prediction in our DM format
		f = open(goldKeysFile, 'r')
		g = open(responseKeysfile, 'w')
		for line in f:
			line = line.rstrip()
			if line[0] == "#":
				g.write(line + "\n")
				continue
			(dirNum, dm, _) = line.split()
			g.write(str(dirNum) + "\t" + str(dm) + "\t(" + str(dmToAggClusterNum[dm]) + ")\n")
		f.close()
		g.close()

