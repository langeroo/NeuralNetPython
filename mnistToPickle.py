import mnist;
import numpy as np;

def oneHot(labels,maxNum):
  oneHotList = []
  for label in labels:
    oh = np.zeros((maxNum,1));
    oh[label] = 1;
    oneHotList.append(oh);
  return oneHotList;

def main():
  print "Loading pictures and labels..."
  # (im_train,lab_train) = mnist.load_mnist(dataset="training", digits=np.arange(10), path=".");
  (im_train,lab_train) = mnist.load_mnist(dataset="testing", digits=np.arange(10), path=".");
  print "Pictures and labels loaded!\n"

  #parameters of inputs/outputs
  numLabels = 10;
  imShape = np.shape(im_train[0]);
  numInputs = imShape[0]*imShape[1]; #multiply by both dimensions of the image

  inputs = im_train;
  print "Formatting label vectors..."
  
  trainLabels = oneHot(lab_train, numLabels);
  print "Labels formatted!"
  print "Saving files..."
  
  # np.save('data/train_images.npy', inputs)
  np.save('data/test_images.npy', inputs)

  print "Images saved!\nWorking on labels..."
  
  # np.save('data/train_labels.npy', trainLabels)
  np.save('data/test_labels.npy', trainLabels)
  print "Labels saved!\nExiting..."

if __name__ == '__main__':
  main()
