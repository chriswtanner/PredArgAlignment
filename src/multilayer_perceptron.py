from __future__ import print_function
#from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import time
import sys
import numpy as np
from random import randint
class multilayer_perceptron:

    def __init__(self, helper, model, params):

        self.isVerbose = True

        # essentials
        self.helper = helper
        self.model = model
        self.params = params
        self.mention2Vec = self.model.train()

        # current class parameters
        self.display_step = 1
        self.nnmethod = params["nnmethod"]
        self.optimizer = params["optimizer"]

        self.n_hidden_1 = params["hidden1"]
        self.n_hidden_2 = params["hidden2"]
        self.n_hidden_3 = 800
        self.p_keep1 = params["p_keep_hidden1"]
        self.p_keep2 = params["p_keep_hidden2"]
        self.p_keep3 = 1.0
        self.batch_size = params["batch_size"]

        self.training_epochs = params["num_epochs"]
        self.learning_rate = params["learning_rate"]
        self.momentum = params["momentum"]
        self.subsample = params["subsample"]
        self.penalty = params["penalty"]
        self.activation = params["activation"]

        self.n_input = params["input_size"] # 2 for samelemma test
        self.n_classes = params["output_size"] # 2 for samelemma test
        
        if self.isVerbose:
            print(params)
        # tf Graph input
        self.x = tf.placeholder("float", [None, self.n_input])
        self.y = tf.placeholder("float", [None, self.n_classes])
        self.p_keep_input = tf.placeholder("float")
        self.p_keep_hidden1 = tf.placeholder("float")
        self.p_keep_hidden2 = tf.placeholder("float")

        #mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)
        
        # bindings
        self.direct = self.directLink()
        self.logits = self.inference()
        self.loss_val = self.get_loss()
        self.train_op = self.train()

        self.predict = tf.argmax(self.logits, dimension=1)
        self.sameLemma = tf.argmax(self.direct, dimension=1)

        # Initializing the variables
        init = tf.global_variables_initializer()

        #(trainX, trainY) = self.loadVectorPairsSubSampleLemma(self.helper.trainingDMPairs, self.nnmethod)
        #(devX, devY) = self.loadVectorPairsLemma(self.helper.devDMPairs, -1, self.nnmethod)
        (trainX, trainY) = self.loadVectorPairsSubSample(self.helper.trainingDMPairs, self.nnmethod)
        (devX, devY) = self.loadVectorPairs(self.helper.devDMPairs, -1, self.nnmethod)

        # Launch the graph
        with tf.Session() as sess:
            sess.run(init)

            # Training cycle
            for epoch in range(self.training_epochs):
                e_time = time.time()
                avg_cost = 0.
                num_batches = int(len(trainX)/self.batch_size)#int(mnist.train.num_examples/self.batch_size)
                #num_batches = int(mnist.train.num_examples/self.batch_size)
                # Loop over all batches
                for i in range(num_batches):

                    batch_x = trainX[i*self.batch_size:(i+1)*self.batch_size]
                    batch_y = trainY[i*self.batch_size:(i+1)*self.batch_size] #mnist.train.next_batch(self.batch_size)
                    #batch_x, batch_y = mnist.train.next_batch(self.batch_size)

                    # Run optimization op (backprop) and cost op (to get loss value)
                    _, c = sess.run([self.train_op, self.loss_val], feed_dict={self.x: batch_x, self.y: batch_y, self.p_keep_input:0.9, self.p_keep_hidden1: self.p_keep1, self.p_keep_hidden2: self.p_keep2})
                    #print("epoch: " + str(epoch) + "batcH: " + str(i) + "; cost: " + str(c))
                    # Compute average loss
                    avg_cost += c / num_batches

                # Display logs per epoch step
                if epoch % self.display_step == 0:
                    
                    trainPredictions = sess.run(self.predict, feed_dict={self.x: trainX, self.y: trainY, self.p_keep_input:1.0, self.p_keep_hidden1: 1.0, self.p_keep_hidden2: 1.0})
                    devPredictions = sess.run(self.predict, feed_dict={self.x: devX, self.y: devY, self.p_keep_input:1.0, self.p_keep_hidden1: 1.0, self.p_keep_hidden2: 1.0})
                    
                    (train_accuracy, train_f1) = self.calculateScore(trainY, trainPredictions)
                    (dev_accuracy, dev_f1) = self.calculateScore(devY, devPredictions)
                    if self.isVerbose == True:
                        print("EPOCH " + str(epoch) + "; TRAIN f1: " + str(train_f1) + "; TRAIN accuracy: " + str(train_accuracy))
                        print("EPOCH " + str(epoch) + "; DEV f1: " + str(dev_f1) + " ; DEV accuracy: " + str(dev_accuracy))
                    #print("\nepoch "+ str(epoch) + " took " + str(round(time.time() - e_time, 1)) + " secs -- f1 (dev set): " + str(dev_f1) + ", " + str(dev_accuracy) + " -- (train set): " + str(train_f1) + ", " + str(train_accuracy))
                    sys.stdout.flush()

                if epoch > 1 and epoch % 5 == 0:
                    allTestGold = []
                    allTestPred = []
                    for dirNum in self.helper.testingDirs:
                        (testX, testY) = self.loadVectorPairs(self.helper.testingDMPairs, dirNum, self.nnmethod)
                        if len(testX) < 1:
                            continue
                        testPredictions = sess.run(self.predict, feed_dict={self.x: testX, self.y: testY, self.p_keep_input:1.0, self.p_keep_hidden1: 1.0, self.p_keep_hidden2: 1.0})
                        (test_accuracy, test_f1) = self.calculateScore(testY, testPredictions)
                        for i in range(len(testY)):
                            allTestGold.append(testY[i])
                            allTestPred.append(testPredictions[i])
                    (all_test_accuracy, all_test_f1) = self.calculateScore(allTestGold, allTestPred)
                    print("EPOCH " + str(epoch) + ": TEST_F1: " + str(all_test_f1) + "  TEST_accuracy: " + str(all_test_accuracy))
                    allTestGold = [] # saves memory
                    allTestPred = [] # saves memory

            allTestGold = []
            allTestPred = []
            for dirNum in self.helper.testingDirs:
                (testX, testY) = self.loadVectorPairs(self.helper.testingDMPairs, dirNum, self.nnmethod)
                if len(testX) < 1:
                    continue
                testPredictions = sess.run(self.predict, feed_dict={self.x: testX, self.y: testY, self.p_keep_input:1.0, self.p_keep_hidden1: 1.0, self.p_keep_hidden2: 1.0})
                (test_accuracy, test_f1) = self.calculateScore(testY, testPredictions)
                if self.isVerbose == True:
                    print("* TEST DIR " + str(dirNum) + ": f1:" + str(test_f1))
                for i in range(len(testY)):
                    allTestGold.append(testY[i])
                    allTestPred.append(testPredictions[i])
            (all_test_accuracy, all_test_f1) = self.calculateScore(allTestGold, allTestPred)
            print("EPOCH " + str(epoch) + ": FINAL_TEST_F1: " + str(all_test_f1) + "  FINAL_TEST_accuracy: " + str(all_test_accuracy))

    def directLink(self):
        return self.x

    # Create model
    def inference(self):
        # Store layers weight & bias
        weights = {}
        weights['h1'] = init_weight([self.n_input, self.n_hidden_1])
        weights['h2'] = init_weight([self.n_hidden_1, self.n_hidden_2])
        weights['h3'] = init_weight([self.n_hidden_2, self.n_hidden_3])

        weights['out'] = init_weight([self.n_hidden_2, self.n_classes])
        #weights['out'] = init_weight([self.n_hidden_3, self.n_classes])
        biases = {}
        biases['b1'] = init_weight([self.n_hidden_1])
        biases['b2'] = init_weight([self.n_hidden_2])
        biases['b3'] = init_weight([self.n_hidden_3])

        biases['out'] = init_weight([self.n_classes])
        
        if self.activation == "relu":
            layer_1 = tf.nn.relu(tf.add(tf.matmul(tf.nn.dropout(self.x, self.p_keep_input), weights['h1']), biases['b1']))
            layer_1 = tf.nn.dropout(layer_1, self.p_keep_hidden1)
            layer_2 = tf.nn.relu(tf.add(tf.matmul(layer_1, weights['h2']), biases['b2']))
            layer_2 = tf.nn.dropout(layer_2, self.p_keep_hidden2)
            out_layer = tf.matmul(layer_2, weights['out']) + biases['out']

            '''
            layer_3 = tf.nn.relu(tf.add(tf.matmul(layer_2, weights['h3']), biases['b3']))
            layer_3 = tf.nn.dropout(layer_3, self.p_keep3)
            out_layer = tf.matmul(layer_3, weights['out']) + biases['out']
            '''

            return out_layer
        elif self.activation == "sigmoid":
            layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(self.x, weights['h1']), biases['b1']))
            layer_1 = tf.nn.dropout(layer_1, self.p_keep_hidden1)
            layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['h2']), biases['b2']))
            layer_2 = tf.nn.dropout(layer_2, self.p_keep_hidden2)
            out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
            return out_layer

    def get_loss(self):
        cost = tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(targets=self.y, logits=self.logits, pos_weight=self.penalty))
        #cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.y))
        return cost

    def train(self):
        if self.optimizer == "gd":
            return tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate).minimize(self.loss_val)
        elif self.optimizer == "rms":
            return tf.train.RMSPropOptimizer(learning_rate=self.learning_rate, momentum=self.momentum).minimize(self.loss_val)
        elif self.optimizer == "adam":
            #updates = tf.train.AdagradOptimizer(learning_rate).minimize(cost)
            return tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss_val)
        else:
            print("ERROR: optimzer not recognized")
            exit(1)

        return tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss_val)
  
    def calculateScore(self, gold, predictions):

        accuracy = np.mean(np.argmax(gold, axis=1) == predictions)
        golds_flat = np.argmax(gold, axis=1)

        num_predicted_true = 0
        num_predicted_false = 0
        num_golds_true = 0
        num_correct = 0
        for i in range(len(golds_flat)):
            if golds_flat[i] == 1:
                num_golds_true = num_golds_true + 1

        for i in range(len(predictions)):
            if predictions[i] == 1:
                num_predicted_true = num_predicted_true + 1
                if golds_flat[i] == 1:
                    num_correct = num_correct + 1
            else:
                num_predicted_false += 1
        recall = float(num_correct) / float(num_golds_true)
        prec = 0
        if num_predicted_true > 0:
            prec = float(num_correct) / float(num_predicted_true)
        f1 = 0
        if prec > 0 or recall > 0:
            f1 = 2*float(prec * recall) / float(prec + recall)
        if self.isVerbose == True:
            print("------")
            print("num_golds_true: " + str(num_golds_true) + "; num_predicted_false: " + str(num_predicted_false) + "; num_predicted_true: " + str(num_predicted_true) + " (of these, " + str(num_correct) + " actually were)")
            print("recall: " + str(recall) + "; prec: " + str(prec) + "; f1: " + str(f1) + "; accuracy: " + str(accuracy))
        return (accuracy, f1)


    def loadVectorPairsSubSample(self, dmPairs, method):
        start_time = time.time()
        # gets dimension size
        dim = 0
        for dmPair in dmPairs:
            (dm1,dm2) = dmPair
            m1 = self.helper.corpus.dmToMention[dm1]
            vec1 = self.mention2Vec[m1]
            if method == "sub":
                dim = len(vec1)
            elif method == "full":
                dim = len(vec1)*2
            else:
                print("ERROR: not a valid method")
                exit(1)
            break

        numPositives = 0
        numNegatives = 0
        for dmPair in dmPairs:
            label = self.helper.goldDMToTruth[dmPair]
            if label == 1:
                numPositives += 1
            elif label == 0:
                numNegatives += 1

        if self.isVerbose == True:
            print("numP: " + str(numPositives))
            print("numN: " + str(numNegatives))
        num_examples = numPositives * (self.subsample + 1) # subSampleN negatives per 1 positive.

        x = np.ones((num_examples,dim))
        y = np.ndarray((num_examples,2)) # always a
        #print "making x size: " + str(num_examples) + " * " + str(dim+1)
        i = 0
        if self.isVerbose == True:
            print("(train) all examples: " + str(numPositives + numNegatives))
            print("(train) num_examples (actually using): " + str(num_examples))

        # populates the negatives
        num_filled = 0
        for dmPair in dmPairs:
            label = self.helper.goldDMToTruth[dmPair]
            if label == 0:
                if num_filled >= num_examples:
                    break
                (dm1,dm2) = dmPair
                m1 = self.helper.corpus.dmToMention[dm1]
                m2 = self.helper.corpus.dmToMention[dm2]
                vec1 = self.mention2Vec[m1]
                vec2 = self.mention2Vec[m2]
                y[num_filled,0] = 1
                y[num_filled,1] = 0

                if method == "sub":
                    j = 0
                    for i2 in range(len(vec1)):
                        v = vec1[i2] - vec2[i2]
                        x[num_filled,j] = v
                        j = j + 1

                elif method == "full":
                    j = 0
                    for i2 in range(len(vec1)):
                        x[num_filled,j] = vec1[i2]
                        j = j + 1
                    #start = len(vec1) # to save the time of looking up the length of vec1
                    for i2 in range(len(vec2)):
                        x[num_filled,j] = vec2[i2]
                        j = j + 1

                num_filled += 1

        # populates the positives
        positiveIndices = set()
        for dmPair in dmPairs:
            label = self.helper.goldDMToTruth[dmPair]
            if label == 1:
                randIndex = randint(0, num_examples-1)
                while randIndex in positiveIndices:
                    randIndex = randint(0, num_examples-1)
                positiveIndices.add(randIndex)

                #print "randIndex: " + str(randIndex)
                y[randIndex,0] = 0
                y[randIndex,1] = 1

                (dm1,dm2) = dmPair
                m1 = self.helper.corpus.dmToMention[dm1]
                m2 = self.helper.corpus.dmToMention[dm2]

                vec1 = self.mention2Vec[m1]
                vec2 = self.mention2Vec[m2]

                if method == "sub":
                    j = 0
                    for i2 in range(len(vec1)):
                        v = vec1[i2] - vec2[i2]
                        x[randIndex,j] = v
                        j = j + 1

                elif method == "full":
                    j = 0
                    for i2 in range(len(vec1)):
                        x[randIndex,j] = vec1[i2]
                        j = j + 1
                    for i2 in range(len(vec2)):
                        x[randIndex, j] = vec2[i2]
                        j = j + 1
        #print("*** LOADING VECTORS TOOK --- %s seconds ---" % (time.time() - start_time))
        '''
        print "# rand: " + str(positiveIndices)
        for p in positiveIndices:
            print p
        '''
        return (x,y)

    def loadVectorPairs(self, dmPairs, dirNum, method):
        start_time = time.time()

        newDMPairs = []
        if dirNum != -1:
            for dmPair in dmPairs:
                (dm1,dm2) = dmPair
                m1 = self.helper.corpus.dmToMention[dm1]    
                if m1.dirNum == dirNum:
                    newDMPairs.append(dmPair)
            dmPairs = newDMPairs
            #print("* dir " + str(dirNum) + " made " + str(len(dmPairs)) + " pairs")

        # gets dimension size
        dim = 0
        for dmPair in dmPairs:
            (dm1,dm2) = dmPair
            m1 = self.helper.corpus.dmToMention[dm1]
            vec1 = self.mention2Vec[m1]
            if method == "sub":
                dim = len(vec1)
            elif method == "full":
                dim = len(vec1)*2
            else:
                print("ERROR: not a valid method")
                exit(1)
            break

        #print("(dev) num_examples: " + str(len(dmPairs)))

        # gets num_examples
        num_examples = len(dmPairs)

        x = np.ones((num_examples,dim))
        y = np.ndarray((num_examples,2)) # always a
        #print "making x size: " + str(num_examples) + " * " + str(dim+1)
        i = 0
        for dmPair in dmPairs:
            (dm1,dm2) = dmPair
            m1 = self.helper.corpus.dmToMention[dm1]
            m2 = self.helper.corpus.dmToMention[dm2]

            vec1 = self.mention2Vec[m1]
            vec2 = self.mention2Vec[m2]

            label = self.helper.goldDMToTruth[dmPair]
            if label == 0:
                y[i,0] = 1
                y[i,1] = 0
            elif label == 1:
                y[i,0] = 0
                y[i,1] = 1
            else:
                print("ERROR: label was weird: " + str(label))
                exit(1)

            if method == "sub":
                j = 0
                for i2 in range(len(vec1)):
                    v = vec1[i2] - vec2[i2]
                    x[i,j] = v
                    j = j + 1

            elif method == "full":
                j = 0
                for i2 in range(len(vec1)):
                    x[i,j] = vec1[i2]
                    j = j + 1
                for i2 in range(len(vec2)):
                    x[i, j] = vec2[i2]
                    j = j + 1

            else:
                print("ERROR: not a valid method")
                exit(1)

            i = i + 1
            if i == num_examples:
                break
        #print("*** LOADING VECTORS TOOK --- %s seconds ---" % (time.time() - start_time))
        return (x,y)




    def loadVectorPairsSubSampleLemma(self, dmPairs, method):
        start_time = time.time()
        dim = 2
        numPositives = 0
        numNegatives = 0
        for dmPair in dmPairs:
            label = self.helper.goldDMToTruth[dmPair]
            if label == 1:
                numPositives += 1
            elif label == 0:
                numNegatives += 1
        if self.isVerbose == True:
            print("numP: " + str(numPositives))
            print("numN: " + str(numNegatives))
        num_examples = numPositives * (self.subsample + 1) # subSampleN negatives per 1 positive.

        x = np.ones((num_examples,dim))
        y = np.ndarray((num_examples,2)) # always a

        i = 0
        
        if self.isVerbose == True:
            print("num_examples: " + str(num_examples))

        # populates the negatives
        num_filled = 0
        for dmPair in dmPairs:
            label = self.helper.goldDMToTruth[dmPair]
            if label == 0:
                if num_filled >= num_examples:
                    break

                (dm1,dm2) = dmPair
                sameLemma = 0
                if self.helper.dmToLemma[dm1] == self.helper.dmToLemma[dm2]:
                    sameLemma = 1

                #sameLemma = self.helper.goldDMToTruth[dmPair]
                y[num_filled,0] = 1
                y[num_filled,1] = 0
                if sameLemma == 0:
                    x[num_filled,0] = 1
                    x[num_filled,1] = 0
                else:
                    x[num_filled,0] = 0
                    x[num_filled,1] = 1
                num_filled += 1

        # populates the positives
        positiveIndices = set()
        for dmPair in dmPairs:
            label = self.helper.goldDMToTruth[dmPair]
            if label == 1:
                (dm1,dm2) = dmPair
                sameLemma = 0
                if self.helper.dmToLemma[dm1] == self.helper.dmToLemma[dm2]:
                    sameLemma = 1

                #sameLemma = self.helper.goldDMToTruth[dmPair]
                randIndex = randint(0, num_examples-1)
                while randIndex in positiveIndices:
                    randIndex = randint(0, num_examples-1)
                positiveIndices.add(randIndex)

                #print "randIndex: " + str(randIndex)
                y[randIndex,0] = 0
                y[randIndex,1] = 1
                if sameLemma == 0:
                    x[randIndex,0] = 1
                    x[randIndex,1] = 0
                else:
                    x[randIndex,0] = 0
                    x[randIndex,1] = 1

        if self.isVerbose == True:
            print("*** LOADING VECTORS TOOK --- %s seconds ---" % (time.time() - start_time))
        return (x,y)

    def loadVectorPairsLemma(self, dmPairs, subset, method):
        start_time = time.time()
        dim = 2
        # gets num_examples
        num_examples = min(subset, len(dmPairs))
        if subset == -1:
            num_examples = len(dmPairs)

        x = np.ones((num_examples,dim))
        y = np.ndarray((num_examples,2)) # always a
        #print "making x size: " + str(num_examples) + " * " + str(dim+1)
        i = 0
        for dmPair in dmPairs:
            (dm1,dm2) = dmPair


            label = self.helper.goldDMToTruth[dmPair]
            if label == 0:
                y[i,0] = 1
                y[i,1] = 0
            elif label == 1:
                y[i,0] = 0
                y[i,1] = 1
            else:
                print("ERROR: label was weird: " + str(label))
                exit(1)
            
            sameLemma = 0
            if self.helper.dmToLemma[dm1] == self.helper.dmToLemma[dm2]:
                sameLemma = 1

            #sameLemma = self.helper.goldDMToTruth[dmPair]
            if sameLemma == 0:
                x[i,0] = 1
                x[i,1] = 0
            else:
                x[i,0] = 0
                x[i,1] = 1

            i = i + 1
            if i == num_examples:
                break
        if self.isVerbose == True:
            print("*** LOADING VECTORS TOOK --- %s seconds ---" % (time.time() - start_time))
        return (x,y)

def init_weight(shape):
    return tf.Variable(tf.random_normal(shape, stddev=1))

