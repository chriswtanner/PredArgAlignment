import sys
from ECBParser import ECBParser
from LSTM_old import LSTM_old
from collections import defaultdict

# PURPOSE: this branch from Test allows us to:
# 	(1) write out HDDCRP's cluster predictions, such that the CoNLL scorer can evaluate
# REQUIRES:
#	(1) HDDCRP's output for the entire corpus (that is, we'd have to run HDDCRP on all dirs as its test sets, then cat to make 1 big file, e.g., 'hddcrp_all_output.txt')
class MakeSameLemmaResponseFile:

	# example run:
	# python Test.py /Users/christanner/research/PredArgAlignment/ 25 true true true cpu lstm 100 3 5 10 1.0
	if __name__ == "__main__":

		corpus = "dev" # or test

		goldEventsFile = "/Users/christanner/research/PredArgAlignment/data/goldEvents.txt" # used to map from HDDCRP's dm format to our DM format
		goldKeysFile = "/Users/christanner/research/PredArgAlignment/data/" + corpus + ".keys" # just to know the ordering of the output
		responseKeysfile = "/Users/christanner/research/PredArgAlignment/data/" + corpus + ".response"

		# reads goldEvent.txt to:
		# (1) map our DM format to its lemma
		# (2) map lemma to a unique cluster #
		dmToLemma = {}
		lemmaToClusterNum = {}
		f = open(goldEventsFile, 'r')
		for line in f:
			(dirNum, ref, doc_id, m_id, lemma, head, sent_num, start_token, end_token, d, e) = line.rstrip().split(";")
			dm = doc_id + ";" + m_id
			dmToLemma[dm] = lemma
			
			if lemma not in lemmaToClusterNum.keys():
				lemmaToClusterNum[lemma] = len(lemmaToClusterNum.keys())
		f.close()

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
			g.write(str(dirNum) + "\t" + str(dm) + "\t(" + str(lemmaToClusterNum[dmToLemma[dm]]) + ")\n")
		f.close()
		g.close()


