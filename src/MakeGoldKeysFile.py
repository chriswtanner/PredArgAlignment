from collections import defaultdict

# PURPOSE: to score w/ the CoNll metrics, we use the CoNll-provided scorer.pl
#          which requires 2 files: .key (the gold truth) and .response
#          this script makes the .key file for either the dev set or gold set
# NOTE: to run on dev, replace the 'testingDirs' text w/ 'devDirs' in line 23, and outputfile
class MakeGoldKeysFile:
	goldTruth = "/Users/christanner/research/PredArgAlignment/data/goldEvents.txt"
	output = "/Users/christanner/research/PredArgAlignment/data/test.keys"

	devDirs = range(23,26)
	testingDirs = range(26,46)

	dirToREFS = defaultdict(set)
	refsToDMs = defaultdict(set)

	f = open(goldTruth, 'r')
	fout = open(output, 'w')

	for line in f:
		print line
		#print line.rstrip().split(";")
		(dirNum, ref, doc_id, m_id, lemma, head, a, b, c, d, e) = line.rstrip().split(";")
		dirNum = int(dirNum)
		if dirNum in testingDirs:
			dm = doc_id + ";" + m_id
			dirToREFS[dirNum].add(ref)
			refsToDMs[ref].add(dm)

	refToNum = {}
	fout.write("#begin document (t);\n")
	#for d, v in dirNumToKeys.iteritems():
	print dirToREFS.keys()
	for d in sorted(dirToREFS.keys()):
		for ref in dirToREFS[d]:
			for dm in refsToDMs[ref]:
				refNum = -1
				if ref in refToNum:
					refNum = refToNum[ref]
				else:
					refNum = len(refToNum.keys())
					refToNum[ref] = refNum
					print "setting " + str(ref) + " to " + str(refNum)
				fout.write(str(d) + "\t" + str(dm) + "\t(" + str(refNum) + ")\n")
	fout.write("#end document\n")
	fout.close()
