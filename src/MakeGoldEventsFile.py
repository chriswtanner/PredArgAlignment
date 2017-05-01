import sys
sys.path.append('/gpfs/main/home/christanner/.local/lib/python2.7/site-packages/tensorflow/')
from ECBParser import ECBParser
from LSTM_old import LSTM_old
from collections import defaultdict

# PURPOSE: this branch from Test allows us to:
# 	(1) create the goldTruth for just the Event Mentions
# REQUIRES:
#	(1) HDDCRP's output for the entire corpus (that is, we'd have to run HDDCRP on all dirs as its test sets, then cat to make 1 big file, e.g., 'hddcrp_all_output.txt')
class MakeGoldEventsFile:


	# example run:
	# python Test.py /Users/christanner/research/PredArgAlignment/ 25 true true true cpu lstm 100 3 5 10 1.0
	if __name__ == "__main__":

		isVerbose = False
		basePath = sys.argv[1] # /Users/christanner/research/PredArgAlignment/ or /data/people/christanner/
		stitchMentions = sys.argv[2] # true or false
		reverseCorpus = sys.argv[3] # true or false

		theirFile = "/Users/christanner/research/PredArgAlignment/data/hddcrp_all_output.txt"
		goldTruthOut = "/Users/christanner/research/PredArgAlignment/data/goldEvents.txt"
		dirNumToKeys = defaultdict(list)

		corpusXMLFiles = basePath + "data/ECB+_LREC2014/ECB+/"
		
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
				#print "key: " + str(key)
				# print key + " ->" + tokens[6].lower()
				docSentTokenToLemma[key] = tokens[6].lower()

				if curSentenceNum > theirDocToHighestSentenceNum[tokens[0]]:
					theirDocToHighestSentenceNum[tokens[0]] = curSentenceNum
		f.close()

		f = open(theirFile, 'r')
		#g = open("/Users/christanner/research/PredArgAlignment/data/goldTruth_events2.txt", 'w')
		theirKeyToPseudoREF = {}
		theirKeys = set()
		theirKeyToLemma = {}
		theirKeyToHeadWord = {}
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

					# TODO: turn this to if True
					if True: #dir_num == "23": #True: #doc_id == "23_1ecbplus.xml":
					
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

						key = doc_id + "," + str(sent_num) + "," + str(start_token) + "," + str(end_token)
						theirKeys.add(key)
						theirKeyToPseudoREF[key] = lineNum
						theirKeyToLemma[key] = lemmas
						theirKeyToHeadWord[key] = head_word
			lineNum = lineNum + 1
		f.close()

		# goes through our parsed corpus
		theirFormatToDM = {}
		numFound = 0
		for dm in corpus.dmToREF.keys():
			m = corpus.dmToMention[dm]
			sentenceNum = int(m.tokens[0].sentenceNum)

			if m.doc_id in theirDocToHighestSentenceNum.keys() and m.doc_id in corpus.docToHighestSentenceNum.keys():

				numFound += 1
				diff = corpus.docToHighestSentenceNum[m.doc_id] - theirDocToHighestSentenceNum[m.doc_id]
				sentenceNum = sentenceNum - int(diff)

				key = m.doc_id + "," + str(sentenceNum) + "," + str(m.tokens[0].tokenNum) + "," + str(m.tokens[-1].tokenNum)
				if key in theirKeys:
					dirNumToKeys[m.dirNum].append(key)
					theirFormatToDM[key] = dm
				
			elif m.doc_id not in theirDocToHighestSentenceNum.keys() and m.doc_id in corpus.docToHighestSentenceNum.keys():
				print "** why don't we have the doc " + m.doc_id + " in their corpus?"
			else:
				print "ERROR: check code"
				exit(1)
		print "numFound: " + str(numFound)

		print "# hddcrp keys:" + str(len(theirKeys))
		print "# of these which were found in our parsed corpus: " + str(len(theirFormatToDM))
		
		fout = open(goldTruthOut, 'w')
		for d, v in dirNumToKeys.iteritems():
			for key in dirNumToKeys[d]:
				#print key
			#theirFormatToDM.keys():
				dm = theirFormatToDM[key]
				mention = corpus.dmToMention[dm]
				
				firstToken = mention.tokens[0]
				globalSentenceNum = firstToken.globalSentenceNum
				sentTokens = corpus.globalSentenceNumToTokens[globalSentenceNum][1:] # removes the <start> token
				sentTokens = sentTokens[:-1]
				mentionTokenIndices = ""
				sent = ""
				for i in range(len(sentTokens)):
					sent += str(sentTokens[i].text) + " "
					if sentTokens[i] in mention.tokens:
						mentionTokenIndices += str(i) + " "
				mentionTokenIndices = mentionTokenIndices.rstrip()
				sent = sent.replace(";", "--")
				sent = sent.rstrip()

				if len(mentionTokenIndices) == 0:
					print "ERROR: we didn't find the Mention's tokens within the sentence"
					exit(1)

				(doc_id, sent_num, start_token, end_token) = key.split(",")
				fout.write(str(mention.dirNum) + ";" + str(corpus.dmToREF[dm]) + ";" + str(mention.doc_id) + ";" \
				+ str(mention.m_id) + ";" + str(theirKeyToLemma[key]) + ";" + str(theirKeyToHeadWord[key]) + ";" \
				+ str(sent_num) + ";" + str(start_token) + ";" + str(end_token) + ";" + str(mentionTokenIndices) + ";" + str(sent) + "\n")
		fout.close()
