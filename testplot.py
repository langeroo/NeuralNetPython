import matplotlib.pyplot as plt
import numpy as np
import pickle as cucumber
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--filename', type=str, default = '-1')
args = parser.parse_args()
infilename = args.filename
output_filename  = infilename[11:-7] 

training_error_file = infilename
test_error_file = 'ttpickle/test' + infilename[16:]
title = 'Test and Train Error On Config '+ infilename[17:-7]

with open(training_error_file,'r') as f:
  train_error = cucumber.load(f);

with open(test_error_file,'r') as f:
  test_error = cucumber.load(f);

train_error_fixed = train_error[0];
test_error_fixed = test_error[0]
for i in range(1,len(train_error)):
  if (i%59):
    train_error_fixed = np.append(train_error_fixed, train_error[i]);
    test_error_fixed = np.append(test_error_fixed, test_error[i]);

x_vals = range(0,len(train_error_fixed));

train = plt.plot(x_vals, train_error_fixed, label="Train Error")
test = plt.plot(x_vals, test_error_fixed, label="Test Error")
legend = plt.legend(loc='center right', shadow=True, fontsize='large')
legend.get_frame().set_facecolor('#CCCCCC')


axes = plt.gca()
axes.set_xlim([0,len(x_vals)])
axes.set_ylim([0,100])

plt.xlabel('Training Examples (x1000)')
plt.ylabel('Error')
plt.title(title)
plt.grid(False)
plt.savefig("outputPlots/" + output_filename)
