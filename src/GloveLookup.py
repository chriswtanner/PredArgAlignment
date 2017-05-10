import sys
from collections import defaultdict

# PURPOSE: allows for fetching glove embeddings on either:
#	(1) pre-trained -- gloveFile1
#	(2) corpus-trained -- gloveFile2
class GloveLookup:
	goldEventsFile = "/Users/christanner/research/PredArgAlignment/data/goldEvents.txt"
	gloveFile1 = "/Users/christanner/research/PredArgAlignment/data/glove.6B.300d.txt"
	gloveFile2 = "/Users/christanner/research/PredArgAlignment/data/glove_nonstitched_400.txt"

	# TODO:
	# self.pretrainedDMsToHead = {} maps DM -> word to look-up within gloveFile1
	# self.corpusDMsToHead = {} maps DM -> word to look-up within gloveFile2
	# self.pretrainedDMsToMentionWords = {} maps DM -> mention word to look-up within gloveFile1
	# self.corpusDMsToMentionHead = {} maps DM -> mention words to look-up within gloveFile2

	# order is (1) actual head token; (2) the formatted head word; (3) the headToFixed[] look-up
	f = open(goldEventsFile, 'r')
	headWords = set()
	headToFixed = {}
	headToFixed["takeing"] = "taking"
	headToFixed["show's"] = "show"
	headToFixed["riot\""] = "riot"
	headToFixed["riot'"] = "riot"
	headToFixed["i.r"] = "ir"
	headToFixed["marriage\""] = "marriage"
	headToFixed["microserver"] = "server"
	headToFixed["#oscars"] = "oscars"
	headToFixed["death?and"] = "death"
	headToFixed["support'"] = "support"
	headToFixed["wwdc12"] = "conference"
	headToFixed["arest"] = "arrest"
	headToFixed["gettingready"] = "ready"
	headToFixed["murder\""] = "murder"
	headToFixed["dwus"] = "dui"
	headToFixed["die'"] = "die"
	headToFixed["musicals\""] = "musicals"
	headToFixed["spiritual'"] = "spiritual"
	headToFixed["madness\""] = "madness"
	headToFixed["surgery-enhanced"] = "surgery"
	headToFixed["awards?"] = "awards"
	headToFixed["oscars?"] = "oscars"
	headToFixed["takeing"] = "taking"
	headToFixed["gathering'"] = "gathering"
	headToFixed["blindsided\""] = "blindsided"
	headToFixed["attacks\""] = "attacks"
	for line in f:
		(dirNum, ref, doc_id, m_id, lemma, head, sent_num, start_token, end_token, d, e, h, g) = line.rstrip().split(";")
		words = e.split(" ")
		head = words[int(h)]
		#print head
		if head in headToFixed.keys():
			headWords.add(headToFixed[head])
		else:
			headWords.add(head)

	print str(len(headWords)) + " headwords"
	f.close()
	f = open(gloveFile2)
	gloveWords = set()
	for line in f:
		tokens = line.rstrip().split(" ")
		w = tokens[0]
		if w in headWords:
			gloveWords.add(w)
	f.close()
	print str(len(gloveWords))
	for w in headWords:
		if w not in gloveWords:
			print w