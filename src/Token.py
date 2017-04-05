class Token:
	def __init__(self, tokenID, sentenceNum, tokenNum, text, stitchedTogether=False, tokens=[]):
		self.stitchedTogether = stitchedTogether
		self.tokens = tokens
		self.tokenNum = tokenNum
		if self.stitchedTogether == True:
			self.sentenceNum = self.tokens[0].sentenceNum
			text = ""
			tokenID = ""
			for t in tokens:
				text = text + t.text + "_"
				tokenID = tokenID + t.tokenID + ","
			text = text[:-1]
			tokenID = tokenID[:-1]
			self.text = text
			self.tokenID = tokenID
		else:
			self.tokenID = tokenID
			self.sentenceNum = sentenceNum
			self.text = text

	def __str__(self):
		return "TOKEN: ID:" + str(self.tokenID) + "; SENTENCE#:" + str(self.sentenceNum) + "; TEXT:" + str(self.text)