import sys
import collections
sys.path.append('/gpfs/main/home/christanner/.local/lib/python2.7/site-packages/tensorflow/')
import tensorflow as tf
import numpy as np

from ECBParser import ECBParser
from ECBHelper import ECBHelper
from LSTM_old import LSTM_old
from Word2Vec import Word2Vec
from collections import defaultdict
from multilayer_perceptron import multilayer_perceptron

class Test:

	# example run:
	# python Test.py /Users/christanner/research/PredArgAlignment/ 25 true true true cpu lstm 100 3 5 10 1.0
	if __name__ == "__main__":
		#print tf.__version__
		tf.logging.set_verbosity(tf.logging.ERROR)
		isVerbose = False
		w2v_vectorFile = "/Users/christanner/research/PredArgAlignment/tmp/word2vec_output.txt"

		basePath = sys.argv[1] # /Users/christanner/research/PredArgAlignment/ or /data/people/christanner/
		corpusDir = int(sys.argv[2]) # 25  (-1 is the flag for running a global model across all dirs)
		readPlus = sys.argv[3] # use ecbplus.xml (true) or ecbp.xml (false)
		stitchMentions = sys.argv[4] # true or false
		reverseCorpus = sys.argv[5] # true or false
 		goldTruthFile = basePath + "data/goldTruth_events.txt"
		goldLegendFile = basePath + "data/gold_events_legend.txt"

		# model params
		model_type = sys.argv[6] # lstm, charcnn, word2vec
		hidden_size = int(sys.argv[7]) # 100
		num_steps = int(sys.argv[8]) # 3
		num_epochs = int(sys.argv[9]) # 30
		batch_size = int(sys.argv[10]) # 8
		learning_rate = float(sys.argv[11]) # 1.0
		windowSize = int(sys.argv[12])

		# NN classifier params
		params = {}
		params["nnmethod"] = sys.argv[13] # sub or full
		params["optimizer"] = sys.argv[14] # ["gd", "rms", "ada"]

		params["hidden1"] = int(sys.argv[15]) # 800-1000ish
		params["hidden2"] = int(sys.argv[16]) # 400-800ish
		params["p_keep_hidden1"] = float(sys.argv[17])
		params["p_keep_hidden2"] = float(sys.argv[18])
		params["num_epochs"] = int(sys.argv[19]) # was 25, 75
		
		params["batch_size"] = int(sys.argv[20]) # 100
		params["learning_rate"] = float(sys.argv[21])
		params["momentum"] = float(sys.argv[22]) # [0.0, 0.1, 0.9]
		params["subsample"] = int(sys.argv[23]) # e.g., 1 or 2
		params["penalty"] = int(sys.argv[24]) # 2
		params["activation"] = sys.argv[25] # e.g., activation or relu

		params["input_size"] = 2*hidden_size*(windowSize*2 + 1)
		params["output_size"] = 2

		if params["nnmethod"] == "full":
			params["input_size"] = 4*hidden_size*(windowSize*2 + 1)

		if corpusDir == -1:
			corpusXMLFiles = basePath + "data/ECB+_LREC2014/ECB+/"
		elif corpusDir == -2:
			corpusXMLFiles = basePath + "data/ECB+_LREC2014/SMALL_ECB/"
		elif corpusDir == -3:
			corpusXMLFiles = basePath + "data/ECB+_LREC2014/HALF_ECB/"
		else:
			corpusXMLFiles = basePath + "data/ECB+_LREC2014/ECB+/" + str(corpusDir) + "/"

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

		if corpusDir < 0:
			suffix = ""
			corpusFilter = ".xml"

		corpusTextFile = basePath + "data/ECB+_LREC2014/TXT/" + str(corpusDir) + str(suffix) + ".txt"

		corpus = ECBParser(corpusXMLFiles, corpusFilter, stitchMentions, reverseCorpus, isVerbose)
		helper = ECBHelper(corpus, goldTruthFile, goldLegendFile, isVerbose)
		#helper.printMentions(0,"false")
		#goldDMs = helper.getGoldDMs()

		'''
		for t in range(len(corpus.corpusTokens)):
			print str(t) + ": " + str(corpus.corpusTokens[t])
		'''

		# write the text file versions just once, then 'cat' them all to make a new 0.txt
		#corpus.writeToTextFile(corpusTextFile)
		#exit(1)

		# "/data/people/christanner/models/"
		#mentionToVec = {}
		model = None
		if model_type == 'lstm':
			config_name = str(stitchMentions) + "_ws" + str(windowSize) + "_h" + str(hidden_size) + "_ns" + str(num_steps) + "_ne" + str(num_epochs) + "_bs" + str(batch_size) + "_lr" + str(learning_rate) + str(suffix) + "_dir" + str(corpusDir)
			#modelOutputFile =  basePath + "models/" + str(model_type) + "/" + str(config_name) + ".allvectors"
			model = LSTM_old(corpus, helper, hidden_size, num_steps, num_epochs, batch_size, learning_rate, windowSize, isVerbose)
			#mentionToVec = model.train()
			#print "LSTM is returning # keys: " + str(len(mentionToVec.keys()))
		elif model_type == 'word2vec':
			config_name = "word2vec"
			model = Word2Vec(corpus, helper, w2v_vectorFile, helper.getGoldDMs())
			#mentionToVec = model.getVectors()
			#print(str(len(mentionToVec.keys())))

		#(trainingDMPairs, testingDMPairs) = helper.getDMTrainTestSets(mentionToVec, -1)
		#print "# training DMs: " + str(len(trainingDMPairs))
		#print "# testing DMs: " + str(len(testingDMPairs))
		
		nn = multilayer_perceptron(helper, model, params)
		#baseOut = basePath + "results/" + str(model_type) + "/" + str(config_name)
		
		# constructs the new goldTruth_new[dirnum].txt file, which we only need to do if we edit the original XML files
		# then we can 'cat' all of the dirnums together to make a new goldTruth
		''' 
		goldF = open(basePath + "data/goldTruth_new" + str(corpusDir) + ".txt", 'w')
		for dm in corpus.dmToREF.keys():
			(doc_id,m_id) = dm
			dirnum = doc_id[0:2]
			goldF.write(str(dirnum) + ";" + corpus.dmToREF[dm] + ";" + doc_id + ";" + str(m_id) + ";\n")
		goldF.close()
		'''