import sys
from ECBParser import ECBParser
from LSTM_old import LSTM_old
from collections import defaultdict

# PURPOSE: this branch from Test allows us to:
# 	(1) write out HDDCRP's cluster predictions, such that the CoNLL scorer can evaluate
# REQUIRES:
#	(1) HDDCRP's output for the entire corpus (that is, we'd have to run HDDCRP on all dirs as its test sets, then cat to make 1 big file, e.g., 'hddcrp_all_output.txt')
class MakeHDDCRPResponseFile:

	# example run:
	# python Test.py /Users/christanner/research/PredArgAlignment/ 25 true true true cpu lstm 100 3 5 10 1.0
	if __name__ == "__main__":

		corpus = "test" # or test

		theirFile = "/Users/christanner/research/PredArgAlignment/data/hddcrp_" + corpus + "_output2.txt"
		goldEventsFile = "/Users/christanner/research/PredArgAlignment/data/goldEvents.txt" # used to map from HDDCRP's dm format to our DM format
		goldKeysFile = "/Users/christanner/research/PredArgAlignment/data/" + corpus + ".keys" # just to know the ordering of the output
		responseKeysfile = "/Users/christanner/research/PredArgAlignment/data/" + corpus + ".response"

		# reads goldEvent.txt to:
		# (1) map our DM format to hddcrp format
		# (2) map hddcrp format to DM format
		dmToHKey = {}
		HKeyToDM = {}
		f = open(goldEventsFile, 'r')
		for line in f:
			(dirNum, ref, doc_id, m_id, lemma, head, sent_num, start_token, end_token, d, e) = line.rstrip().split(";")
			dm = doc_id + ";" + m_id
			hkey = doc_id + "," + sent_num + "," + start_token + "," + end_token + "," + head
			dmToHKey[dm] = hkey
			HKeyToDM[hkey] = dm
		f.close()

		# reads the hddcrp file so that we can:
		# (1) map hddcrp formats to cluster Num
		HKeyToClusterNum = {}
		f = open(theirFile, 'r')
		clusterNum = 0
		for line in f:
			tokens = line.rstrip().split(";")
			for t in tokens:
				HKeyToClusterNum[t] = clusterNum
			clusterNum += 1
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
			hkey = dmToHKey[dm]
			#print "dm: " + str(dm) + " => " + str(hkey)
			g.write(str(dirNum) + "\t" + str(dm) + "\t(" + str(HKeyToClusterNum[hkey]) + ")\n")

		f.close()
		g.close()


