from neural_network import *;
import numpy as np;

def main():

  #config parameters
  numEpochs = 50;
  alpha = 0.1;

  train_input = [.4,.4,.2,1];
  train_label = [.9,.2,.1,.551,.4];

  #Neural Network Definition
  hiddenLayerDimensions = [ 100, len(train_label) ];
  NN = NeuralNetwork(hiddenLayerDimensions, len(train_input),alpha)

  for j in range(0,numEpochs):

    print train_input
    print train_label

    y_hat = NN.feedForward(train_input)
    delta = NN.computeDError(np.array(train_label), y_hat)
    NN.feedBack(delta)
    print y_hat

if __name__ == '__main__':
  main()
