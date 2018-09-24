from batch_neural_network import *
import numpy as np
import itertools
import pickle
import argparse

def reshapeInstance(t_input, t_label, numElements):

  #reshape and scale // make it a list
  t_input = np.reshape(t_input,(numElements,1))
  t_input = (t_input + 1)/257.
  t_input = t_input.tolist()
  t_input = list(itertools.chain.from_iterable(t_input))

  #make it a list
  t_label = t_label.tolist()
  t_label = list(itertools.chain.from_iterable(t_label))

  return (t_input, t_label)

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--error-file', type=str, default='error_batch.pickle')
  parser.add_argument('--test-error-file', type=str, default='test_error_batch.pickle')
  parser.add_argument('--test-error-on', type=int, default=0);
  parser.add_argument('--NN-setup', type=int, default = 0)

  parser.add_argument('--num-epochs', type=int, default=3, help='no help for you!')
  parser.add_argument('--alpha', type=float, default=.03)
  parser.add_argument('--batch-size', type=int, default=2, help='number of examples per batch, -1 is full batch')
  args = parser.parse_args()
  error_file = args.error_file
  test_error_file = args.test_error_file;
  test_error_on = args.test_error_on;
  nn_setup = args.NN_setup;

  #config parameters
  numEpochs = args.num_epochs
  alpha = args.alpha
  batchSize = args.batch_size

  # loading in that good good data
  trainImages = np.load('data/train_images.npy')
  trainLabels = np.load('data/train_labels.npy')
  
  testImages = np.load('data/test_images.npy')
  testLabels = np.load('data/test_labels.npy')

  imSize = np.shape(trainImages[0])
  inputSize = imSize[0]*imSize[1]
  numLabels = len(trainLabels[0])

  #print '# features in:',inputSize
  #print '# possibe labels out:',numLabels

  #Neural Network Definition
  hiddenLayerDimensionList = [[350,175,85,40,20,numLabels], [300,150,75,40,numLabels], [280,130,60,numLabels], [250,100,numLabels], [175,numLabels]];
  hiddenLayerDimensions = hiddenLayerDimensionList[nn_setup];
  # hiddenLayerDimensions = [ 300, 30, numLabels ]

  NN = BatchNeuralNetwork(hiddenLayerDimensions, inputSize,alpha)
  correctTrain = 0.0
  error_over_iters = np.array([], dtype='float64')
  test_error_over_iters = np.array([], dtype='float64')

  if batchSize == -1: batchSize = len(trainLabels)

  for j in range(0,numEpochs):
    i_last = 0
    for i in range(0,len(trainLabels),batchSize): 

      upperBound = i+batchSize
      if upperBound > len(trainLabels) - 1:
        upperBound = len(trainLabels) - 1

      inputs = trainImages[i:upperBound]
      labels = trainLabels[i:upperBound]

      train_inputs = [];
      train_labels = [];
      #in case the last batch isn't full size
      bsBatchSize = upperBound - i 
      for k in range(0, bsBatchSize):
        (train_input, train_label) = reshapeInstance(inputs[k], labels[k], inputSize)
        train_inputs.append(train_input)
        train_labels.append(train_label)

      y_hats = NN.feedForward(train_inputs)
      outputDeltas = NN.computeDError(train_labels, y_hats)
      NN.feedBack(outputDeltas)

      # print y_hats
      for a in range(0,len(y_hats)):
        y_hat = y_hats[a]
        train_label = train_labels[a]
        same = (np.argmax(np.asarray(y_hat)) == np.argmax(np.asarray(train_label)))
        correctTrain += float(same)

      if (i % 1000 == 0 and i != 0):
        correctTest = 0.0;
        denom = i - i_last
        i_last = i
        train_accuracy = 100 - (100 * (float(correctTrain)/float(denom)))
        print 'Epoch//Sample:',j,'//',i
        print 'Train Error:', int(train_accuracy),'%'
        correctTrain = 0
        error_over_iters = np.append(error_over_iters, train_accuracy)
        
        if test_error_on:
          correct = 0;
          for idx in range(0,len(testLabels),batchSize):

            upperBound = idx+batchSize
            if upperBound > len(trainLabels) - 1:
              upperBound = len(trainLabels) - 1

            inputs = testImages[idx:upperBound]
            labels = testLabels[idx:upperBound]

            # print labels

            test_inputs = [];
            test_labels = [];
            #in case the last batch isn't full size
            bsBatchSize = upperBound - idx 
            for k in range(0, bsBatchSize):
              (test_input, test_label) = reshapeInstance(inputs[k], labels[k], inputSize)
              test_inputs.append(test_input)
              test_labels.append(test_label)

            y_hats = NN.feedForward(test_inputs)
            
            for a in range(0,len(y_hats)):
              y_hat = y_hats[a]
              test_label = test_labels[a]

              same = (np.argmax(np.asarray(y_hat)) == np.argmax(np.asarray(test_label)))
              correctTest += float(same)

          test_error =  100 - (100 * (float(correctTest)/float(idx+1)));
          test_error_over_iters = np.append(test_error_over_iters, test_error);
          print 'Test Error:', int(test_error), '%'

  with open(error_file, 'wb') as f:
    pickle.dump(error_over_iters, f)
  
  if test_error_on:
    with open(test_error_file, 'wb') as f:
      pickle.dump(test_error_over_iters, f)

if __name__ == '__main__':
  main()
