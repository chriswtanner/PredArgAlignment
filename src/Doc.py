from Mention import Mention
from collections import defaultdict

# used for the LSTMFeatureMaker.py file
class Doc:

	def __init__(self, doc_id, mentions, tokens, globalTokenIDs):
		self.doc_id = doc_id
		self.mentions = mentions
		self.tokens = tokens
		self.globalTokenIDs = globalTokenIDs

	def printGlobalTokenIDs(self):
		return self.globalTokenIDs
		
	def __str__(self):
		return "DOC: " + str(self.doc_id) + " (encompasses " + str(len(self.mentions)) + " mentions); TOKENS: " + str([x.text for x in self.tokens])
