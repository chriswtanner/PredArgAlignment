from collections import defaultdict
from Doc import Doc
from Mention import Mention
from Token import Token

class ECBHelper:

	def __init__(self, corpus, goldTruthFile, goldLegendFile, isVerbose):
		self.isVerbose = isVerbose

		self.corpus = corpus
		self.goldTruthFile = goldTruthFile
		self.refToDMs = defaultdict(list)
		self.dmToLemma = {}
		self.goldDMPairs = set() # the pairs we care about, per the legends file
		self.goldDMToTruth = {} # the gold truth

		self.trainingDirs = range(1,23)
		self.trainingDirs.remove(15)
		self.trainingDirs.remove(17)
		self.devDirs = range(23,26)
		self.testingDirs = range(26,46)

		f = open(goldTruthFile, 'r')
		for line in f:
			(dirNum, ref, doc_id, m_id, lemma, headword) = line.rstrip().split(";")
			key = (str(doc_id),int(m_id))
			self.refToDMs[ref].append(key)
			self.dmToLemma[key] = lemma
		f.close()

		f = open(goldLegendFile, 'r')
		for line in f:
			(dmPair, truth) = line.rstrip().split(" ")
			(d1,m1,d2,m2) = dmPair.split(",")
			dm1 = (str(d1),int(m1))
			dm2 = (str(d2),int(m2))
			self.goldDMPairs.add((dm1,dm2))
			self.goldDMToTruth[(dm1,dm2)] = int(truth)
		f.close()

		# loads the training and testing lists of DMs
		(trainingDMPairs, devDMPairs, testingDMPairs) = self.getDMPairsSplit()
		self.trainingDMPairs = trainingDMPairs
		self.devDMPairs = devDMPairs
		self.testingDMPairs = testingDMPairs


	# NEVER CALLED; but would be passed (basePath + "data/parserInput.txt")
	def writeToCharniakParserInputFormat(self, outputFile):
		fout = open(outputFile, 'w')
		for t in self.corpus.corpusTokens:
			if t.text == "<start>":
				fout.write("<s> ")
			elif t.text == "<end>":
				fout.write("</s>\n")
			else:
				fout.write(t.text + " ")

	# NEVER CALLED; but i once made it in Test.py
	# write the text file versions just once, then 'cat' them all to make a new 0.txt
	def writeToTextFile(self, outputFile):		
		print "writing out " + str(len(self.corpus.docTokens)) + " lines to " + str(outputFile)		
		fout = open(outputFile, 'w')
		for d in self.corpus.docTokens:
			tmpout = "" # allows for removing the trailing space
			for t in d:
				tmpout = tmpout + t.text + " "
			fout.write(tmpout.rstrip() + "\n")
		fout.close()

	# NEVER CALLED; but i once made it in Test.py
	# constructs the new goldTruth_new[dirnum].txt file, which we only need to do if we edit the original XML files
	# then we can 'cat' all of the dirnums together to make a new goldTruth
	def makeConciseGoldTruthFile(self, basePath):
		goldF = open(basePath + "data/goldTruth_new" + str(corpusDir) + ".txt", 'w')
		for dm in self.corpus.dmToREF.keys():
			(doc_id,m_id) = dm
			dirnum = doc_id[0:2]
			goldF.write(str(dirnum) + ";" + corpus.dmToREF[dm] + ";" + doc_id + ";" + str(m_id) + ";\n")
		goldF.close()

	def getDMPairs(self, dms):
		dmPairsSet = set() # just to 
		dmPairsList = []
		for dm1 in dms:
			m1 = self.corpus.dmToMention[dm1]
			for dm2 in dms:
				if dm2 != dm1 and (dm2,dm1) not in dmPairsSet:
					m2 = self.corpus.dmToMention[dm2]
					if m1.dirNum == m2.dirNum and m1.suffix == m2.suffix:
						if (dm1,dm2) in self.goldDMPairs:
							dmPairsList.append((dm1,dm2))
							dmPairsSet.add((dm1,dm2))
						elif (dm2,dm1) in self.goldDMPairs:
							dmPairsList.append((dm2,dm1))
							dmPairsSet.add((dm2,dm1))
						else:
							print "ERROR: dm1" + str(dm1) + " and dm2: " + str(dm2) + " not in goldDMPairs"
							exit(1)

		return dmPairsList

	def getDMPairsSplit(self):
		trainingDMs = []
		devDMs = []
		testingDMs = []
		goldDMs = self.getGoldDMs()
		foundDMs = set()
		for dm in self.corpus.dmToREF.keys():
			if dm in goldDMs:
				foundDMs.add(dm)
				dirNum = int(dm[0][0:dm[0].find("_")])
				if dirNum in self.trainingDirs:
					trainingDMs.append(dm)
				elif dirNum in self.devDirs:
					devDMs.append(dm)
				elif dirNum in self.testingDirs:
					testingDMs.append(dm)
				else:
					print "ERROR: dm: " + str(dm) + " doesn't seem to belong to either the training, dev, or testing dirs specified here in ECBHelper"
					exit(1)

		if self.isVerbose:
			print "goldDMs size (from goldTruth file): " + str(len(goldDMs)) + "; foundDMs (from the actual corpus): " + str(len(foundDMs))
			if len(goldDMs) != len(foundDMs):
				print "** ERROR: goldDMs didn't match 1-to-1"
			#exit(1) #TODO; this shouldn't be commented out

		# gets training pairs
		trainingDMPairs = self.getDMPairs(trainingDMs)
		devDMPairs = self.getDMPairs(devDMs)
		testingDMPairs = self.getDMPairs(testingDMs)
		if self.isVerbose:
			print "# total goldPairs (from legend txt file): " + str(len(self.goldDMPairs))
			print "# corpus-training pairs loaded: " + str(len(trainingDMPairs))
			print "# corpus-dev pairs loaded: " + str(len(devDMPairs))
			print "# corpus-testing pairs loaded: " + str(len(testingDMPairs))
		return (trainingDMPairs, devDMPairs, testingDMPairs)

	''' # DEFUNCT
	def getDMTrainTestSets(self, mentionToVec, training_subset_size=-1):
		trainingDMPairs = []
		testingDMPairs = []
		dmPairsSet = set() # just for speed

		train_size = 0
		for m1 in mentionToVec:
			dm1 = (m1.doc_id,m1.m_id)

			self.dmToMention[dm1] = m1
			for m2 in mentionToVec:

				if m1.dirNum == m2.dirNum and m1 != m2 and m1.suffix == m2.suffix:

					dm2 = (m2.doc_id,m2.m_id)
					self.dmToMention[dm2] = m2
					if (dm1,dm2) not in dmPairsSet and (dm2,dm1) not in dmPairsSet:

						# finds the correct ordering which matches the golds
						pair = (dm1,dm2)
						if pair in self.goldDMPairs:
							pair = (dm1,dm2)
						elif (dm2,dm1) in self.goldDMPairs:
							pair = (dm2,dm1)
						else:
							print "ERROR: the 2 dms are found in the gold!"
							exit(1)
							
						dmPairsSet.add(pair)
						
						if m1.dirNum in self.trainingDirs:
							if training_subset_size == -1 or train_size < training_subset_size:
								trainingDMPairs.append(pair)
								train_size = train_size + 1
						elif m1.dirNum in self.testingDirs:
							testingDMPairs.append(pair)
						else:
							print "ERROR: dir: " + str(m1.dirNum) + " is not a training or testing dir"

		missing = set()
		for dmPair in self.goldDMPairs:
			if dmPair not in dmPairsSet:
				missing.add(dmPair)
		print "we are missing " + str(len(missing)) + " pairs from the legend file:"
		#for m in missing:
		#	print m
		return (trainingDMPairs, testingDMPairs)
	'''

	def getGoldDMs(self):
		dms = set()
		for dm in self.dmToLemma.keys():
			dms.add(dm)
		return dms

	def printMentions(self, windowSize, includeMiddleOfMention):
		n = 3
		for ref in self.refToDMs.keys():
			#print "REF: " + str(ref)
			for dm in self.refToDMs[ref]:
				if dm not in self.corpus.dmToMention.keys():
					continue
				m = self.corpus.dmToMention[dm]
				print "\t" + str(dm) + ": " + str(m)
				tokensWeCareAbout = self.getMentionContextIndices(m, windowSize, includeMiddleOfMention)
				print tokensWeCareAbout

	# returns the corpusTokens' index to the mention and its 'windowSize' words on each side
	def getMentionContextIndices(self, mention, windowSize, includeMiddleOfMention):
		ret = []
		startingIndex = mention.corpusTokenIndices[0]
		endingIndex = mention.corpusTokenIndices[-1]
		start = max(0, startingIndex - windowSize)
		for i in range(start, startingIndex):
			if i not in ret:
				ret.append(i)

		if includeMiddleOfMention.lower() == "true":
			for i in range(startingIndex, endingIndex + 1):
				if i not in ret:
					ret.append(i)
		else:
			for i in mention.corpusTokenIndices:
				if i not in ret:
					ret.append(i)

		end = min(endingIndex + windowSize, len(self.corpus.corpusTokens) - 1)
		for i in range(endingIndex + 1, end + 1):
			if i not in ret:
				ret.append(i)
		return ret

	def getMentionPrevTokenIndices(self, mention, windowSize):
		ret = []
		startingIndex = mention.corpusTokenIndices[0]
		start = max(0, startingIndex - windowSize)
		for i in range(start, startingIndex):
			if i not in ret:
				ret.append(i)
		return ret

	def getMentionNextTokenIndices(self, mention, windowSize):
		ret = []
		endingIndex = mention.corpusTokenIndices[-1]
		end = min(endingIndex + windowSize, len(self.corpus.corpusTokens) - 1)
		for i in range(endingIndex + 1, end + 1):
			if i not in ret:
				ret.append(i)
		return ret


