import sys
import editdistance
from collections import defaultdict

# PURPOSE: annotate our goldEvents file with POS tags
class AddPOSTagsAndHeadIndexToGoldEventsFile:
	goldEventsFile = "/Users/christanner/research/PredArgAlignment/data/goldEvents.txt"
	goldEventsFile2 = "/Users/christanner/research/PredArgAlignment/data/goldEvents2.txt"
	posFile = "/Users/christanner/research/PredArgAlignment/data/all_sentences_pos.txt.xml"

	f = open(goldEventsFile, 'r')
	g = open(posFile, 'r')
	fout = open(goldEventsFile2, 'w')
	numSkipped = 0
	lineNum = 0

	for line in f:
		#print "lineNum:" + str(lineNum)
		lineNum += 1
		(dirNum, ref, doc_id, m_id, lemma, head, sent_num, start_token, end_token, d, e) = line.rstrip().split(";")
		curWords = e.split(" ")
		#print curWords
		curPOS = []
		for i in range(len(curWords)):
			word = curWords[i]
			if len(word) > 2 and "'s" == word[-2:]:
				word = word[:word.index("'s")]
				#print "A"
			if "n't" in word and word != "n't":
				word = word[:word.index("n't")]
				#print "B"
			if word == "i'm":
				word = "i"
			if len(word) > 2 and word[-1] == "'":
				word = word[:-1]
			if len(word) > 2 and word[-1] == "\"":
				word = word[:-1]
			if word == "''the":
				word = "the"
			if len(word) > 2 and "'" == word[0] and word != "'15" and word != "'50s" and word != "'re" and word != "'ve" and word != "'09":
				word = word[1:]
				#print "C"
			if word == "2,292nd":
				word = "2,292"
			if word == "#1":
				word = "1"
			if len(word) > 1 and word[0] == "$":
				word = word[1:]
			if len(word) > 1 and "\"" == word[0]:
				word = word[1:]
			if len(word) > 2 and word == "awards\xc2\xae":
				word = "awards"
			if len(word) > 1 and word[-1] == "%":
				word = word[:-1]
			if word == "-and-run":
				word = "and-run"
			if word == "two\xe2\x80\x93":
				word = "two"
			if word == "\xe2\x82\xac2" or word == "\xe2\x82\xac2m":
				word = "2"
			if word == "\xe2\x82\xac2m":
				word = "2m"
			if word == "\xc2\xa31":
				word = "1"
			if word == "4\xc3\x974":
				word = "4"
			if len(word) > 2 and word[len(word)-2:] == "'d":
				word = word[0:-2]
			if word == "emmy\xc2\xae":
				word = "emmy"
			if word == "oscars\xc2\xae":
				word = "oscars"
			if word == "&amp--":
				word = "&amp;"
			if word == "\"it\\'s":
				#print "E"
				word = "it"
			if word == "we'll":
				word = "we"
			if word == "we've":
				word = "we"
			if word == "\xe2\x80\x9c":
				word = "``"
			if word == "\xe2\x80\xa6":
				word = "..."
			if word == "1/2":
				curPOS.append("CD")
				continue
			if len(word) > 2 and "\"" == word[0]:
				word = word[1:]
				#print "D"
			if len(word) > 2 and word[-1] == "'":
				word = word[:-1]
			if word == "\xe2\x80\x98":
				word = "`"
			if len(word) > 2 and word[0] == "`":
				word = word[1:]
			if word == "\"" or word == "\xe2\x80\x9d":
				word = "''"
			if len(word) > 2 and word[-2:] == "'l":
				word = word[:-2]
			if word == "projects\xe2\x80\x94such":
				word = "projects"
			if word == "dams\xe2\x80\x94would":
				word = "dams"
			if word == "mines\xe2\x80\x94would":
				word = "mines"
			if word == "c&amp--g":
				word = "c"
			if word == "death\xe2\x80\x94and":
				word = "death"
			if word == "'a":
				word = "a"
			if word == "q&amp--a":
				word = "q"
			if word == "-magnitude":
				word = "magnitude"
			if word == "2.30am":
				word = "2.30"
			if word == "3.45pm":
				word = "3.45"
			if word == "blackberry\xc2\xae" or word == "blackberry\xef\xbf\xbd":
				word = "blackberry"
			if word == "us$334":
				word = "$"
			if word == "5.4bn":
				word = "5.4"
			if word == "ka'loni":
				word = "ka"
			if len(word) > 3 and word[-3:] == "'re" or word[-3:] == "'ll":
				word = word[:-3]
			if len(word) > 2 and word[-1] == "-":
				word = word[:-1]
			if word == "dec.14":
				word = "dec."
			if word == "(":
				word = "-LRB-"
			if word == ")":
				word = "-RRB-"
			if word == "[":
				word = "-LSB-"
			if word == "]":
				word = "-RSB-"
			if word == "\xe2\x80\x94" or word == "\xe2\x80\x93":
				word = "--"
			if word == "i'd":
				word = "i"
			if word == ".":
				curPOS.append(".")
				#print "added POS ."
				continue

			#print "word: " + str(word)
			# reads through the .xml until we find the target word
			while True:
				xmlLine = g.readline().lstrip().rstrip()
				#print xmlLine
				if "<word>" in xmlLine:
					foundWord = xmlLine[6:xmlLine.index("</word>")]
					if foundWord == word:
						break
					elif foundWord == "'" and word == "\xe2\x80\x99":
						#print "WHOA!"
						break
					elif foundWord == "..." and word == ".":
						break
					elif foundWord == "2m" and word == "2":
						break
					elif foundWord[0] == "2" and word == "2":
						break
					else:
						#print "skipping word: " + str(foundWord)
						numSkipped += 1
						if numSkipped == 5:
							print "EXITING!"
							exit(1)
			
			# reads through the .xml until we find the POS
			while True:
				xmlLine = g.readline().lstrip().rstrip()
				if "<POS>" in xmlLine:
					numSkipped = 0
					foundPOS = xmlLine[5:xmlLine.index("</POS>")]
					curPOS.append(foundPOS)
					#print "\t* setting " + str(word) + " = " + str(foundPOS)
					break
		#print curPOS
		if len(curWords) != len(curPOS):
			print "UNEQUAL!"
		else:
			# find the head word
			headToken = ""
			shortestDist = 9999
			mentionWords = []
			headIndex = -1
			for i in range(int(end_token) - int(start_token) + 1):
				curWord = curWords[int(start_token) + i]
				#print str(i) + " => " + str(curWord)
				mentionWords.append(curWord)
				dist = int(editdistance.eval(curWord, head))
				if dist < shortestDist:
					shortestDist = dist
					headToken = curWord
					headIndex = int(start_token) + i
			if str(curWords[headIndex]) != str(headToken):
				print "ERROR!!  heads don't match"
				exit(1)
			fout.write(line.rstrip() + ";")
			fout.write(str(headIndex) + ";" + curPOS[headIndex] + "\n")
			#fout.write()
	f.close()
	g.close()
	fout.close()