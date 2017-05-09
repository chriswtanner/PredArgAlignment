import sys
import collections
sys.path.append('/gpfs/main/home/christanner/.local/lib/python2.7/site-packages/tensorflow/')
import tensorflow as tf
import numpy as np

from sklearn.cluster import AgglomerativeClustering
from ECBParser import ECBParser
from ECBHelper import ECBHelper
from LSTM_old import LSTM_old
from WordEmbeddings import WordEmbeddings
from collections import defaultdict
from multilayer_perceptron import multilayer_perceptron

class Test:
	# example run:
	# python Test.py /Users/christanner/research/PredArgAlignment/ 25 true true true cpu lstm 100 3 5 10 1.0
	if __name__ == "__main__":
		#print tf.__version__
		tf.logging.set_verbosity(tf.logging.ERROR)
		isVerbose = True

		# scratch pad


		X = [] #np.empty() # array() #zeros(6)) #[[1, 2], [1, 4], [1, 0],[4, 2], [4, 4], [4, 0]])
		a = [float(1.0),float(2.0),float(1.0),float(2.0)]
		b = [float(2.0),float(1.0),float(1.0),float(2.0)]
		c = [float(1.0),float(3.0),float(1.0),float(2.0)]
		d = [float(4.0),float(2.0),float(1.0),float(2.0)]
		X.append(a)
		X.append(b)
		X.append(c)
		X.append(d)
		X.append(a)
		X.append(b)
		X.append(c)
		X.append(d)		
		print(type(X))
		print(X)
		print(len(X))
		agg = AgglomerativeClustering(n_clusters=3).fit(X)
		print(agg.labels_)
		exit(1)

		basePath = sys.argv[1] # /Users/christanner/research/PredArgAlignment/ or /data/people/christanner/
		corpusDir = int(sys.argv[2]) # 25  (-1 is the flag for running a global model across all dirs)
		readPlus = sys.argv[3] # use ecbplus.xml (true) or ecbp.xml (false)
		stitchMentions = sys.argv[4] # true or false
		reverseCorpus = sys.argv[5] # true or false
 		goldTruthFile = basePath + "data/goldTruth_events.txt"
		goldLegendFile = basePath + "data/gold_events_legend.txt"
		vectorFile = basePath + "data/word2vec_output.txt"

		# model params
		model_type = sys.argv[6] # lstm, charcnn, word2vec
		hidden_size = int(sys.argv[7]) # 100
		num_steps = int(sys.argv[8]) # 3
		num_epochs = int(sys.argv[9]) # 30
		batch_size = int(sys.argv[10]) # 8
		learning_rate = float(sys.argv[11]) # 1.0
		windowSize = int(sys.argv[12])

		if model_type == "gs100":
			vectorFile = basePath + "data/glove_stitched_100.txt"
		elif model_type == "gs400":
			vectorFile = basePath + "data/glove_stitched_400.txt"
		elif model_type == "gns100": # non-stitched 100-length vectors
			vectorFile = basePath + "data/glove_nonstitched_100.txt"
		elif model_type == "gns400":
			vectorFile = basePath + "data/glove_nonstitched_400.txt"

		print("model_type: " + str(model_type))
		# NN classifier params
		params = {}
		params["nnmethod"] = sys.argv[13] # sub or full
		params["optimizer"] = sys.argv[14] # ["gd", "rms", "ada"]

		params["hidden1"] = int(sys.argv[15]) # 800-1000ish
		params["hidden2"] = int(sys.argv[16]) # 400-800ish
		params["p_keep_input"] = float(sys.argv[17])
		params["p_keep_hidden1"] = float(sys.argv[18])
		params["p_keep_hidden2"] = float(sys.argv[19])
		params["num_epochs"] = int(sys.argv[20]) # was 25, 75
		
		params["batch_size"] = int(sys.argv[21]) # 100
		params["learning_rate"] = float(sys.argv[22])
		params["momentum"] = float(sys.argv[23]) # [0.0, 0.1, 0.9]
		params["subsample"] = int(sys.argv[24]) # e.g., 1 or 2
		params["penalty"] = int(sys.argv[25]) # 2
		params["activation"] = sys.argv[26] # e.g., activation or relu

		params["input_size"] = 2*hidden_size*(windowSize*2 + 1)
		params["output_size"] = 2

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

		#helper.printAllTokens()
		#helper.printMentionSentences()
		#exit(1)
		#helper.writeToTextFile(basePath + "data/allTokens.txt")

		model = None
		if model_type == 'lstm':
			model = LSTM_old(corpus, helper, hidden_size, num_steps, num_epochs, batch_size, learning_rate, windowSize, isVerbose)
		elif model_type == 'word2vec' or model_type[0:1] == 'g':
			model = WordEmbeddings(corpus, helper, vectorFile, windowSize, helper.getGoldDMs())
			params["input_size"] = (windowSize*2 + 1)*model.hidden_size
			sys.stdout.flush()
		if params["nnmethod"] == "full":
			params["input_size"] = 2*params["input_size"]
		
		mention2Vec = model.train()
		for mention in mention2Vec.keys():
			print(str(mention) + " has vec:")
			print("\t" + str(mention2Vec[mention]))
			exit(1)



		#nn = multilayer_perceptron(helper, model, params)