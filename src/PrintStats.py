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
class PrintStats:

	if __name__ == "__main__":
		goldEventsFile = "/Users/christanner/research/PredArgAlignment/data/goldEvents.txt" # used to map from HDDCRP's dm format to our DM format

		dirToGoldREFs = defaultdict(set)
		dirToDMs = defaultdict(set)
		REFToDMs = defaultdict(set)
		DMToHead = {}
		f = open(goldEventsFile, 'r')
		for line in f:
			(dirNum, ref, doc_id, m_id, lemma, head, sent_num, start_token, end_token, d, e) = line.rstrip().split(";")
			suffix = doc_id[doc_id.index("ecb"):]
			key = dirNum + "_" + suffix
			dm = doc_id + "," + m_id
			if dirNum == "1":
				continue
			dirToGoldREFs[key].add(ref)
			dirToDMs[key].add(dm)
			REFToDMs[ref].add(dm)
			DMToHead[dm] = head
		f.close()
		numMentionsToREFs = defaultdict(set)
		for ref in REFToDMs.keys():
			numMentions = len(REFToDMs[ref])
			numMentionsToREFs[numMentions].add(ref)

		totalNumREFs = len(REFToDMs.keys())
		cumulCount = 0
		print "# mentions,# clusters w/ that # of mentions, cumulative # of representation"
		for n in sorted(numMentionsToREFs.keys()):
			numREFs = len(numMentionsToREFs[n])
			cumulCount += numREFs
			print str(n) + "," + str(numREFs) + "," + str(float(cumulCount) / totalNumREFs)
		exit(1)
		for d in dirToDMs.keys():
			print str(len(dirToDMs[d])) + "," + str(len(dirToGoldREFs[d]))
		exit(1)
		for d in sorted(dirToGoldREFs.keys()):
			print "dir: " + str(d) + " has # clusters: " + str(len(dirToGoldREFs[d]))
			headToREFs = defaultdict(set)
			for ref in dirToGoldREFs[d]:
				heads = ""
				for dm in REFToDMs[ref]:
					curHead = DMToHead[dm]
					heads += curHead + ","
					headToREFs[curHead].add(ref)
				heads = heads[:-1]
				#print "# dms: " + str(len(REFToDMs[ref])) + ": " + str(heads)
			'''
			for h in headToREFs.keys():
				if len(headToREFs[h]) > 1:
					print "*** head: " + str(h) + " belongs to " + str(headToREFs[h])
			'''