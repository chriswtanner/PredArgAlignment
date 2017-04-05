import collections
import sys

from collections import defaultdict
## PURPOSE: constructs train & test files for SVM from passed-in feature(s) files
## EXAMPLE:
#   python SVMPreprocessor.py
#   /Users/christanner/research/PredArgAlignment/tmp/gold_legend.txt
#   /Users/christanner/research/PredArgAlignment/tmp/orig_8.features,/Users/christanner/research/PredArgAlignment/tmp/h150_ns3_ne20_bs5_lr1.0_stitched_forward_all.features
#   /Users/christanner/research/PredArgAlignment/tmp/svm
#
# (it will make svm_train.legend  svm_train.features   svm_test.legend   svm_test.features)
class SVMPreprocessor:

	if __name__ == "__main__":

		goldLegendFile = sys.argv[1]
		filesToStitchTogether = sys.argv[2]
		outputBase = sys.argv[3]
		print("outbase: " + outputBase)

		trainingDirs = range(1,26)
		trainingDirs.remove(15)
		trainingDirs.remove(17)
		testingDirs = range(26,46)

		print "training: " + str(trainingDirs)
		print "testing: " + str(testingDirs)
		# loads gold legend to a map
		goldLegend = {}
		goldDMs = set()
		dmList = [] # so that we can maintain order between runs
		f = open(goldLegendFile, 'r')
		for line in f:
			tokens = line.split(" ")
			dm = tokens[0]
			truth = tokens[1]
			goldLegend[dm] = truth
			goldDMs.add(dm)
			dmList.append(dm)

		f.close()

		# makes a combined features file
		featureFiles = filesToStitchTogether.split(",")
		allFeatures = defaultdict(list)
		for f in featureFiles:
			
			print("reading features: " + str(f))
			sys.stdout.flush()
			g = open(f, 'r')
			dmsFound = set() # the dms that belong to goldLegend which we actually found in the current features file

			for line in g:
				tokens = line.split(",")
				dm1 = tokens[0] + "," + tokens[1]
				dm2 = tokens[2] + "," + tokens[3]

				forward = dm1 + "," + dm2
				backwards = dm2 + "," + dm1
				correct = ""
				# adhere to the format of the goldLegend file for the ordering of the DM
				if forward in goldDMs:
					correct = forward
				elif backwards in goldDMs:
					correct = backwards

				# we found a dmPair that we actually care about (the features file may contain dm pairs we don't care about)
				if correct != "":
					dmsFound.add(correct)

					features = tokens[4:]
					for feat in features:
						allFeatures[correct].append(float(feat))
			if len(dmsFound) != len(goldDMs):
				print "ERROR: MISSING SOME DMs!!"
				exit(1)
			else:
				print "\t** successfully fetched all DMs listed in the goldLegend file: " + str(goldLegendFile)

		print "done reading features"
		sys.stdout.flush()

		# writes out the file for SVM to use
		outTrainFeaturesFile = outputBase + "_train.features"
		outTrainLegendFile = outputBase + "_train.legend"
		outTestFeaturesFile = outputBase + "_test.features"
		outTestLegendFile = outputBase + "_test.legend"
		trainF = open(outTrainFeaturesFile, 'w')
		trainL = open(outTrainLegendFile, 'w')
		testF = open(outTestFeaturesFile, 'w')
		testL = open(outTestLegendFile, 'w')
		for dmPair in dmList:

			#print(dmPair)
			dirNum = int(dmPair[0:dmPair.find("_")])
			
			outLine = goldLegend[dmPair].rstrip() # writes the golden truth 1st (i.e., 0 or 1)
			i = 1
			for f in allFeatures[dmPair]:
				outLine = outLine + " " + str(i) + ":" + str(f)
				i = i + 1

			if dirNum in trainingDirs:
				trainF.write(outLine + "\n")
				trainL.write(dmPair + "\n")
			elif dirNum in testingDirs:
				testF.write(outLine + "\n")
				testL.write(dmPair + "\n")
			else:
				print str(dirNum) + '.'
				print str(trainingDirs)
				print("ERROR: the goldLegend (and features) contains dms for a dir that is not listed in this script's declared arrays for training/testing")
				exit(1)

		trainF.close()
		trainL.close()
		testF.close()
		testL.close()