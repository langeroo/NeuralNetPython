import numpy as np
from batch_neural_layer import BatchNeuralLayer

class BatchNeuralNetwork:
  def __init__(self, shape, num_inputs,alpha):
    self.num_inputs = num_inputs
    self.num_layers = len(shape)
    self.network = [BatchNeuralLayer(num_inputs, shape[0])]
    self.alpha = alpha

    for layer in range(1, self.num_layers):
      self.network.append(BatchNeuralLayer(shape[layer-1], shape[layer]))

  def feedForward(self, inputs):
    outputs = []
    for i in range(0,len(inputs)):
      input_vect = inputs[i]

      for layer in range(self.num_layers):
        self.network[layer].setPotentials(input_vect)
        input_vect = self.network[layer].getActivations()
      outputs.append(input_vect)

    return outputs

  def feedBack(self, deltas):
    #may need to change to not affect 0-layer

    for i in range(0,len(deltas)):
      delta = deltas[i]

      for layer in range(self.num_layers-1, -1, -1):

        delta = self.network[layer].backPropogate(delta,self.alpha)        
        delta = delta[1:len(delta)]

    for layer in range(self.num_layers-1, -1, -1):

      self.network[layer].weights = self.network[layer].weights - self.alpha * self.network[layer].Delta
      self.network[layer].newBatch()


  def computeDError(self, expected, returned):
    returned = np.asarray(returned)
    expected = np.asarray(expected)
    result = (returned - expected)

    return result.tolist()
