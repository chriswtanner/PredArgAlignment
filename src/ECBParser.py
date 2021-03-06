import sys
import collections
import re
import os
import fnmatch
from glob import glob
from collections import defaultdict

from Doc import Doc
from Mention import Mention
from Token import Token

# PURPOSE: to create a standardized way of parsing the ECB+ corpus and getting its tokens
# HOW TO RUN: this should be instantiated from another class, via
# ECBParser p = ECBParser(1,2)
# params:
# 	1 - /Users/christanner/research/PredArgAlignment/data/ECB+_LREC2014/FIXED/
# 	2 - 25 or 0 (for everyone, which i'll manually hand-curate)
###############
class ECBParser:
	def __init__(self, corpusXMLFiles, corpusFilter, stitchTogether, reverse, isVerbose):
		self.parse(corpusXMLFiles, corpusFilter, stitchTogether, reverse, isVerbose)

	def getGlobalTypeID(self, wordType):
		if wordType in self.typeToGlobalID.keys():
			return self.typeToGlobalID[wordType]
		else:
			newID = len(self.typeToGlobalID.keys())
			self.typeToGlobalID[wordType] = newID
			self.globalIDsToType[newID] = wordType
			return newID

	def parse(self, corpusXMLFiles, corpusFilter, stitchMentions, reverseCorpus, isVerbose):
		self.corpusFilter = corpusFilter
		self.stitchMentions = stitchMentions
		self.reverseCorpus = reverseCorpus

		self.numCorpusTokens = 0 # used so we don't have to do len(corpusTokens) all the time
		self.corpusTokens = []
		self.corpusTokensToCorpusIndex = {}

		self.mentions = []
		self.dmToMention = {} # (doc_id,m_id) -> Mention
		self.dmToREF = {}
		self.refToDMs = defaultdict(list)
		
		# same tokens as corpusTokens, just made into lists according
		# to each doc.  (1 doc = 1 list of tokens); used for printing corpus to .txt file 
		self.docTokens = []
		
		self.typeToGlobalID = {}
		self.globalIDsToType = {}
		self.corpusTypeIDs = []
		
		self.docToHighestSentenceNum = defaultdict(int) # tmp, just for creating an aligned goldTruth from HDDCRP

		# added so that we can easily parse the original sentence which contains each Mention
		self.globalSentenceNumToTokens = defaultdict(list)

		numFound = 0

		files = []
		for root, dirnames, filenames in os.walk(corpusXMLFiles):
			for filename in fnmatch.filter(filenames, '*' + self.corpusFilter):
				files.append(os.path.join(root, filename))
		
		globalSentenceNum = 0
		for f in files:
			doc_id = f[f.rfind("/") + 1:]
			dir_num = int(doc_id.split("_")[0])
			#print "parsing: " + doc_id + " (file " + str(files.index(f) + 1) + " of " + str(len(files)) + ")"
			if isVerbose:
				print "parsing: " + doc_id + " (file " + str(files.index(f) + 1) + " of " + str(len(files)) + ")"
			sys.stdout.flush()
	        # DOC attributes
			# tokenIDsToToken = defaultdict()
			tmpDocTokens = [] # we will optionally flip these and optionally stitch Mention tokens together
			tmpDocTokenIDsToTokens = {}
			docTokenIDToCorpusIndex = {}

			with open (f, 'r') as myfile:
				fileContents=myfile.read().replace('\n',' ')

	        # reads <tokens>
			it = tuple(re.finditer(r"<token t\_id=\"(\d+)\" sentence=\"(\d+)\" number=\"(\d+)\".*?>(.*?)</(.*?)>", fileContents))
			lastSentenceNum = -1


			tokenNum = -1 # numbers every token in each given sentence, starting at 1 (each sentence starts at 1)
			firstToken = True
			for match in it:
				t_id = match.group(1)
				sentenceNum = int(match.group(2))
				#tokenNum = int(match.group(3))
				tokenText = match.group(4).lower()

				#print tokenText

				# TMP
				if sentenceNum > self.docToHighestSentenceNum[doc_id]:
					self.docToHighestSentenceNum[doc_id] = sentenceNum

				if sentenceNum > 0 or "plus" not in doc_id:

					# we are starting a new sentence
					if sentenceNum != lastSentenceNum:
						
						#print "** sent not equal to last"
						# we are possibly ending the prev sentence
						if not firstToken:

							endToken = Token("-1", lastSentenceNum, globalSentenceNum, tokenNum, "<end>")
							tmpDocTokens.append(endToken)
							globalSentenceNum = globalSentenceNum + 1

						tokenNum = -1
						startToken = Token("-1", sentenceNum, globalSentenceNum, tokenNum, "<start>")
						#startToken = Token("-1", sentenceNum, -1, "<start>")
						tmpDocTokens.append(startToken)
						tokenNum = tokenNum + 1

					# adds token
					curToken = Token(t_id, sentenceNum, globalSentenceNum, tokenNum, tokenText)
					tmpDocTokenIDsToTokens[t_id] = curToken
					firstToken = False
					tmpDocTokens.append(curToken)
					tokenNum = tokenNum + 1
				
				lastSentenceNum = sentenceNum

			endToken = Token("-1", lastSentenceNum, globalSentenceNum, tokenNum, "<end>")
			globalSentenceNum = globalSentenceNum + 1
			#endToken = Token("-1", lastSentenceNum, -1, "<end>")
			tmpDocTokens.append(endToken)

			if self.reverseCorpus.lower() == 'true':
				tmpDocTokens.reverse()

			if isVerbose:
				print "\t" + str(len(tmpDocTokens)) + " doc tokens"
				print "\t# unique token ids: " + str(len(tmpDocTokenIDsToTokens))

			tokenToStitchedToken = {} # will assign each Token to its stitchedToken (e.g., 3 -> [3,4], 4 -> [3,4])
			stitchedTokens = []
			# reads <markables> 1st time
			regex = r"<([\w]+) m_id=\"(\d+)?\".*?>(.*?)?</.*?>"
			markables = fileContents[fileContents.find("<Markables>")+11:fileContents.find("</Markables>")]
			it = tuple(re.finditer(regex, markables))
			for match in it:
				# gets the token IDs
				regex2 = r"<token_anchor t_id=\"(\d+)\".*?/>"
				it2 = tuple(re.finditer(regex2, match.group(3)))
				tmpCurrentMentionSpanIDs = []
				text = []
				hasAllTokens = True
				for match2 in it2:
					tokenID = match2.group(1)
					tmpCurrentMentionSpanIDs.append(int(tokenID))
					if tokenID not in tmpDocTokenIDsToTokens.keys():
						hasAllTokens = False

				# we should only have incomplete Mentions for our hand-curated, sample corpus, 
				# for we do not want to have all mentions, so we curtail the sentences of tokens
				if hasAllTokens and self.stitchMentions.lower() == 'true' and len(tmpCurrentMentionSpanIDs) > 1:
					tmpCurrentMentionSpanIDs.sort()
					#print "tmpCurrentMentionSpanIDs: " + str(tmpCurrentMentionSpanIDs)
					spanRange = 1 + max(tmpCurrentMentionSpanIDs) - min(tmpCurrentMentionSpanIDs)
					if isVerbose and spanRange != len(tmpCurrentMentionSpanIDs):
						print "*** WARNING: the mention's token range seems to skip over a token id! " + str(tmpCurrentMentionSpanIDs)
					else:

						#print "tmpCurrentMentionSpanIDs:" + str(tmpCurrentMentionSpanIDs)
						# makes a new stitched-together Token
						tokens_stitched_together = []
						for token_id in tmpCurrentMentionSpanIDs:
							cur_token = tmpDocTokenIDsToTokens[str(token_id)]
							tokens_stitched_together.append(cur_token)

						# this was comprised in order (e.g., 3,4), so we want to reverse it
						if self.reverseCorpus.lower() == 'true':
							tokens_stitched_together.reverse()
						
						stitched_token = Token(-2, -2, -2, -2, "", True, tokens_stitched_together)
						stitchedTokens.append(stitched_token)

						# points the constituent Tokens to its new stitched-together Token
						for token_id in tmpCurrentMentionSpanIDs:
							cur_token = tmpDocTokenIDsToTokens[str(token_id)]

							if cur_token in tokenToStitchedToken.keys():
								if isVerbose:
									print "ERROR: OH NO, the same token id (" +  str(token_id) + ") is used in multiple Mentions!"
								#print tokenToStitchedToken[cur_token]
								#print tmpCurrentMentionSpanIDs
								#exit(1)
							else:
								# print "pointing token id: " + str(token_id) + " to " + str(stitched_token)
								tokenToStitchedToken[cur_token] = stitched_token

			if isVerbose:
				print "# stitched tokens: " + str(len(stitchedTokens))
				for st in stitchedTokens:
					print st
				print "-----------"

			# ADDS to the corpusTokens in the correct, optionally reversed, optionally stitched, manner
			# puts stitched tokens in the right positions
			# appends to the docTokens[] (each entry is a list of the current doc's tokens);
			# only used for writing out the .txt files corpus
			curDocTokens = []
			#self.docTokens.append(tmpDocTokens)
			if self.stitchMentions.lower() == 'true':
				completedTokens = set()
				for t in tmpDocTokens:
					if t not in completedTokens:
						if t in tokenToStitchedToken.keys():

							stitched_token = tokenToStitchedToken[t]
							self.corpusTokens.append(stitched_token)
							curDocTokens.append(stitched_token)

							#print "constitutents: " + str(stitched_token.tokens)
							for const_token in stitched_token.tokens:
								completedTokens.add(const_token)
								# assigns all constituent Tokens to have the same index, because they do (w/ the stitchedToken)
								self.corpusTokensToCorpusIndex[const_token] = self.numCorpusTokens

							self.numCorpusTokens = self.numCorpusTokens + 1
						else:
							self.corpusTokens.append(t)
							curDocTokens.append(t)

							self.corpusTokensToCorpusIndex[t] = self.numCorpusTokens
							self.numCorpusTokens = self.numCorpusTokens + 1
			else:
				for t in tmpDocTokens:
					self.corpusTokens.append(t)
					curDocTokens.append(t)
					self.corpusTokensToCorpusIndex[t] = self.numCorpusTokens
					self.numCorpusTokens = self.numCorpusTokens + 1

			self.docTokens.append(curDocTokens)

			# reads <markables> 2nd time
			regex = r"<([\w]+) m_id=\"(\d+)?\".*?>(.*?)?</.*?>"
			markables = fileContents[fileContents.find("<Markables>")+11:fileContents.find("</Markables>")]
			it = tuple(re.finditer(regex, markables))
			for match in it:
				isPred = False
				if "ACTION" in match.group(1):
					isPred = True
				m_id = int(match.group(2))
				
				# gets the token IDs
				regex2 = r"<token_anchor t_id=\"(\d+)\".*?/>"
				it2 = tuple(re.finditer(regex2, match.group(3)))
				tmpMentionCorpusIndices = []
				tmpTokens = [] # can remove after testing if our corpus matches HDDCRP's
				text = []
				hasAllTokens = True
				for match2 in it2:
					tokenID = match2.group(1)

					if tokenID in tmpDocTokenIDsToTokens.keys():
						
						cur_token = tmpDocTokenIDsToTokens[tokenID]
						tmpTokens.append(cur_token)
						text.append(cur_token.text)

						# gets corpusIndex (we ensure we don't add the same index twice, which would happen for stitched tokens)
						tmpCorpusIndex = self.corpusTokensToCorpusIndex[cur_token]
						if tmpCorpusIndex not in tmpMentionCorpusIndices: 
							tmpMentionCorpusIndices.append(tmpCorpusIndex)
					else:
						hasAllTokens = False

				# we should only have incomplete Mentions for our hand-curated, sample corpus, 
				# for we do not want to have all mentions, so we curtail the sentences of tokens
				if hasAllTokens:

					tmpMentionCorpusIndices.sort() # regardless of if we reverse the corpus or not, these indices should be in ascending order
					if self.reverseCorpus.lower() == 'true':
						text.reverse()

					curMention = Mention(dir_num, doc_id, m_id, tmpTokens, tmpMentionCorpusIndices, text, isPred)
					self.dmToMention[(doc_id,m_id)] = curMention
					self.mentions.append(curMention)

					'''
					if doc_id == "23_1ecbplus.xml":
						print "parsed: " + str(curMention)
						print "\thas tokens: " + str(curMention.tokens[0])
						numFound += 1
					'''
			# reads <relations>
			relations = fileContents[fileContents.find("<Relations>"):fileContents.find("</Relations>")]
			regex = r"<CROSS_DOC_COREF.*?note=\"(.+?)\".*?>(.*?)?</.*?>"
			it = tuple(re.finditer(regex, relations))
			for match in it:
				ref_id = match.group(1)
				regex2 = r"<source m_id=\"(\d+)\".*?/>"
				it2 = tuple(re.finditer(regex2, match.group(2)))

				for match2 in it2:
					m_id = int(match2.group(1))
					if (doc_id,m_id) not in self.dmToMention.keys():
						continue
					self.dmToREF[(doc_id,m_id)] = ref_id
					self.refToDMs[ref_id].append((doc_id,m_id))

		# (1) sets the corpus type-based variables AND
		# (2) sets the globalSentenceTokens so that we can parse the original sentences
		for t in self.corpusTokens:
			# (1)
			g_id = self.getGlobalTypeID(t.text)
			self.corpusTypeIDs.append(g_id)
			
			# (2)
			self.globalSentenceNumToTokens[t.globalSentenceNum].append(t)


		# prints sentences
		'''
		for k, v in self.globalSentenceNumToTokens.iteritems():
			out = str(k) + ":"
			for t in v:
				out += " " + t.text
			print out
		'''
		
		'''
		if isVerbose:
			i=0
			for i in range(len(self.corpusTokens)):
				print str(i) + ": " + str(self.corpusTokens[i])
				i = i + 1
			print "total # of tokens:" + str(len(self.corpusTokens))
			
					
			for m in self.dmToMention.keys():
				print str(m) + " -> " + str(self.dmToMention[m])

			print "# mentions: " + str(len(self.mentions))
			print "# REFS: " + str(len(self.refToDMs))
			for ref in self.refToDMs:
				print str(ref) + ": "
				for dm in self.refToDMs[ref]:
					print "\t" + str(self.dmToMention[dm])
		'''