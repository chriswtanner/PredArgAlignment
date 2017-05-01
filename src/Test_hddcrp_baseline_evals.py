import sys
sys.path.append('/gpfs/main/home/christanner/.local/lib/python2.7/site-packages/tensorflow/')
from ECBParser import ECBParser
from LSTM_old import LSTM_old
from LSTM import LSTM
from FeatureCreator import FeatureCreator
from collections import defaultdict

# PURPOSE: this branch from Test allows us to:
# 	(1) measure the HDDCRP performance
#   (2) measure samelemma and samehead preformance
#   by writing out the features files, which will then by EvaluatePredictions.java
class Test_hddcrp_baseline_evals:

	# example run:
	# python Test.py /Users/christanner/research/PredArgAlignment/ 25 true true true cpu lstm 100 3 5 10 1.0
	if __name__ == "__main__":
		print "* in main()"
		print "sys args: " + str(sys.argv)
		isVerbose = False
		
		basePath = sys.argv[1] # /Users/christanner/research/PredArgAlignment/ or /data/people/christanner/
		corpusDir = int(sys.argv[2]) # 25
		readPlus = sys.argv[3] # use ecbplus.xml (true) or ecbp.xml (false)
		stitchMentions = sys.argv[4] # true or false
		reverseCorpus = sys.argv[5] # true or false


		#corpusXMLFiles = basePath + "data/ECB+_LREC2014/ECB+/" + str(corpusDir) + "/"
		corpusXMLFiles = basePath + "data/ECB+_LREC2014/ECB+/"

		corpusFilter = ""
		
		suffix = ""
		if stitchMentions.lower() == 'true':
			suffix = "_stitched"
		else:
			suffix = "_nonstitched"

		if reverseCorpus.lower() == 'true':
			suffix = suffix + "_reverse"
		else:
			suffix = suffix + "_forward"

		if readPlus.lower() == 'true':
			suffix = suffix + "_ecbplus"
			corpusFilter = "ecbplus.xml"
		else:
			suffix = suffix + "_ecb"
			corpusFilter = "ecb.xml"

		corpus = ECBParser(corpusXMLFiles, ".xml", stitchMentions, reverseCorpus, isVerbose)
		
		# TMP: stores the conll file's lemmas
		docSentTokenToLemma = {}
		theirDocToHighestSentenceNum = defaultdict(int)

		f = open("/Users/christanner/research/HDPCoref/input/documents._auto_conll")
		curSentenceNum = 0
		for line in f:
			line = line.rstrip()
			tokens = line.split()

			if "#begin" in line or "#end document" in line:
				curSentenceNum = 0
			elif len(tokens) == 0:
				curSentenceNum = curSentenceNum + 1
			elif len(tokens) > 0 and ".xml" in tokens[0]:
				key = tokens[0] + "," + str(curSentenceNum) + "," + tokens[2]
				# print key + " ->" + tokens[6].lower()
				docSentTokenToLemma[key] = tokens[6].lower()

				if curSentenceNum > theirDocToHighestSentenceNum[tokens[0]]:
					theirDocToHighestSentenceNum[tokens[0]] = curSentenceNum
		f.close()
		
		# TMP:
		theirFormatToDM = {}
		for dm in corpus.dmToREF.keys():
			m = corpus.dmToMention[dm]
			sentenceNum = int(m.tokens[0].sentenceNum)

			if m.doc_id in theirDocToHighestSentenceNum.keys() and m.doc_id in corpus.docToHighestSentenceNum.keys():
				diff = corpus.docToHighestSentenceNum[m.doc_id] - theirDocToHighestSentenceNum[m.doc_id]
				sentenceNum = sentenceNum - int(diff)

				key = m.doc_id + "," + str(sentenceNum) + "," + str(m.tokens[0].tokenNum) + "," + str(m.tokens[-1].tokenNum) + ";"
				theirFormatToDM[key] = dm
			elif m.doc_id not in theirDocToHighestSentenceNum.keys() and m.doc_id in corpus.docToHighestSentenceNum.keys():
				print "** why don't we have the doc " + m.doc_id + " in their corpus?"

		theirFile = "/Users/christanner/research/HDPCoref/hddcrp_output.txt" # hddcrp_out.txt"
		f = open(theirFile, 'r')
		g = open("/Users/christanner/research/PredArgAlignment/data/goldTruth_events2.txt", 'w')
		theirDMToPseudoREF = {}
		lineNum = 0
		for line in f:
			line = line.rstrip()
			refs = set()
			for dm in line.split(";"):
				theirDM = dm[0:dm.rfind(',')] + ';'
				if len(theirDM) > 1:
					tokens = dm.split(",")
					doc_id = tokens[0]
					dir_num = doc_id[0:doc_id.find("_")]

					if True: # dir_num == str(corpusDir):
					
						sent_num = tokens[1]
						start_token = int(tokens[2])
						end_token = int(tokens[3])
						head_word = tokens[4]
						lemmas = ""
						for i in range(start_token, end_token + 1):
							key = doc_id + "," + sent_num + "," + str(i)
							if key not in docSentTokenToLemma:
								print "coudlnt find: " + key
								exit(1)
							lemmas = lemmas + str(docSentTokenToLemma[key]) + " "
						lemmas = lemmas.rstrip(" ")

						#print "looking up: " + theirDM + " and we have tokens: " + str(tokens)
						dm = theirFormatToDM[theirDM]
						m_id = dm[1]

						new_dm = str(doc_id) + "," + str(m_id)
						#print new_dm + ";"
						theirDMToPseudoREF[new_dm] = lineNum

						ref = corpus.dmToREF[dm]
						refs.add(ref)
						if (doc_id,m_id) not in corpus.dmToMention.keys():
							print "corpus doesn't have " + (doc_id,m_id)
							exit(1)
						else:
							print "have it"
						g.write(dir_num + ";" + ref + ";" + doc_id + ";" + str(m_id) + ";" + lemmas + ";" + head_word + "\n")
			lineNum = lineNum + 1

		#print theirDMToPseudoREF
		# makes a prediction file (which EvaluatePredictions.java will read)
		legendFile = open("/Users/christanner/research/PredArgAlignment/tmp/h300_ns3_ne40_bs5_lr1.0_nonstitched_forward_test.legend", 'r')
		predictionsOut = open("/Users/christanner/research/PredArgAlignment/tmp/hddcrp.predictions", 'w')
		predictionsOut.write("labels 0 1\n") # header
		for line in legendFile:
			tokens = line.rstrip().split(",")
			d1 = tokens[0]
			m1 = tokens[1]
			d2 = tokens[2]
			m2 = tokens[3]
			dm1 = d1 + "," + m1
			dm2 = d2 + "," + m2
			if theirDMToPseudoREF[dm1] == theirDMToPseudoREF[dm2]:
				predictionsOut.write("1 1 0\n")
			else:
				predictionsOut.write("0 0 1\n")
		legendFile.close()
		predictionsOut.close()
		#g.close()
		exit(1)