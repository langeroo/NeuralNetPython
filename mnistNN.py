from neural_network import *
import mnist
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
  parser.add_argument('--error-file', type=str, default = 'error_online.pickle')
  parser.add_argument('--num-epochs', type=int, default = 3)
  parser.add_argument('--alpha', type=float, default = .1)
  args = parser.parse_args()
  error_file = args.error_file

  #config parameters
  numEpochs = args.num_epochs
  alpha = args.alpha

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
  hiddenLayerDimensions = [ 300, 100, numLabels ]
  NN = NeuralNetwork(hiddenLayerDimensions, inputSize,alpha)
  correctTest = 0.0
  error_over_iterations = np.array([], dtype='float64')

  for j in range(0,numEpochs):
    for i in range(0,len(trainLabels)): 

      (train_input, train_label) = reshapeInstance(trainImages[i], trainLabels[i], inputSize)

      y_hat = NN.feedForward(train_input)
      delta = NN.computeDError(np.array(train_label), y_hat)
      NN.feedBack(delta)

      same = (np.argmax(np.asarray(y_hat)) == np.argmax(np.asarray(train_label)))
      correctTest += float(same)

      if (i % 1000 == 0 and i != 0):
        train_accuracy = 100 - (100 * (float(correctTest)/float(1000)))
        #print 'epoch//sample:',j,'//',i
        #print 'Train Accuracy:', int(train_accuracy),'%'
        correctTest = 0
        error_over_iterations = np.append(error_over_iterations, train_accuracy)
        '''
        correct = 0
        for idx in range(0,len(testLabels)):
          (test_input, test_label) = reshapeInstance(testImages[idx], testLabels[idx], inputSize)
          y_hat = NN.feedForward(test_input)
          
          same = (np.argmax(np.asarray(y_hat)) == np.argmax(np.asarray(test_label)))
          correct += float(same)
          
        print 'Test Accuracy:', int(100 * (float(correct)/float(idx+1)) ),'%'
        '''
  with open(error_file, 'wb') as f:
    pickle.dump(error_over_iterations, f)

if __name__ == '__main__':
  main()

