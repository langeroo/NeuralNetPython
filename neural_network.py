import numpy as np
from neural_layer import NeuralLayer

class NeuralNetwork:
  def __init__(self, shape, num_inputs,alpha):
    self.num_inputs = num_inputs
    self.num_layers = len(shape)
    self.network = [NeuralLayer(num_inputs, shape[0])]
    self.alpha = alpha

    for layer in range(1, self.num_layers):
      self.network.append(NeuralLayer(shape[layer-1], shape[layer]))

  def feedForward(self, inputs):
    for layer in range(self.num_layers):
      self.network[layer].setPotentials(inputs)
      inputs = self.network[layer].getActivations()
    return inputs

  def feedBack(self, delta):
    #may need to change to not affect 0-layer
    for layer in range(self.num_layers-1, -1, -1):
      # print 'delta',layer,':',delta
      
      delta = self.network[layer].backPropogate(delta,self.alpha)
      delta = delta[1:len(delta)]

  def computeDError(self, expected, returned):
    return (returned - expected)
